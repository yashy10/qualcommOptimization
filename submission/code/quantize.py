"""Static QDQ quantization of an fp32 ONNX model with ONNX Runtime (plan.md, Section 4).

    python quantize.py --name q1_perchannel_minmax_s8s8 --per-channel --calib-method minmax --types s8s8
    python quantize.py --name q0_pertensor_minmax_u8s8 --per-tensor --calib-method minmax --types u8s8

Steps: quant_pre_process the input model (saved separately), calibrate on the fixed
calibration indices, quantize_static in QDQ format, save models/<name>.onnx with a
models/<name>.json sidecar of the exact settings, then run check_quant.py on the result.
Never dynamic quantization.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader, Subset

from check_quant import check
from common import ROOT, load_dataset

MODELS_DIR = ROOT / "models"
DEFAULT_INPUT = MODELS_DIR / "student_fp32.onnx"
DEFAULT_CALIBRATION = ROOT / "subsets" / "calibration_indices_300_seed42.json"
INPUT_NAME = "input"
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}
# activation type, weight type
DATA_TYPES = {"s8s8": (QuantType.QInt8, QuantType.QInt8), "u8s8": (QuantType.QUInt8, QuantType.QInt8)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FixedIndexReader(CalibrationDataReader):
    """Feeds the fixed calibration images one at a time, with the packet's exact transform."""

    def __init__(self, calibration_file, size=None):
        spec = json.loads(Path(calibration_file).read_text())
        indices = spec["indices"]
        if size and size < len(indices):
            # The file is sorted by class, so take evenly spaced entries to keep every class
            indices = indices[:: len(indices) // size][:size]
        elif size and size > len(indices):
            sys.exit(f"{calibration_file} has only {len(indices)} indices; --calib-size {size} needs a larger file")
        self.indices = indices
        self.loader = iter(DataLoader(Subset(load_dataset(), indices), batch_size=1))
        self.seen = 0

    def get_next(self):
        batch = next(self.loader, None)
        if batch is None:
            return None
        self.seen += 1
        return {INPUT_NAME: batch[0].numpy()}


def chunked_histogram_collection(chunk):
    """Make ORT's histogram calibrator (Entropy/Percentile) build its histograms chunk images at a time.

    ORT keeps every activation of every calibration image in memory before building the
    histograms, which runs out of memory around 1000 images. The collector merges new data
    into existing histograms, so feeding it in chunks gives the same kind of result.
    """
    from onnxruntime.quantization import calibrate

    def collect_data(self, data_reader):
        output_names = [arg.name for arg in self.infer_session.get_outputs()]

        def flush(batch):
            merged = {}
            for outputs in batch:
                for name, value in zip(output_names, outputs):
                    if name in self.tensors_to_calibrate:
                        merged.setdefault(name, []).append(value)
            if not self.collector:
                self.collector = calibrate.HistogramCollector(
                    method=self.method, symmetric=self.symmetric, num_bins=self.num_bins,
                    num_quantized_bins=self.num_quantized_bins, percentile=self.percentile, scenario=self.scenario,
                )
            self.collector.collect(merged)

        batch, seen = [], 0
        while (inputs := data_reader.get_next()):
            batch.append(self.infer_session.run(None, inputs))
            seen += 1
            if len(batch) == chunk:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
        if seen == 0:
            raise ValueError("No data is collected.")

    calibrate.HistogramCalibrater.collect_data = collect_data


def first_and_last_nodes(model_path):
    """The first Conv (sees raw pixels) and the final Gemm (makes the decision)."""
    graph = onnx.load(model_path).graph
    convs = [node.name for node in graph.node if node.op_type == "Conv"]
    gemms = [node.name for node in graph.node if node.op_type in ("Gemm", "MatMul")]
    return [convs[0], gemms[-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="output name: models/<name>.onnx")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="fp32 ONNX model to quantize")
    granularity = parser.add_mutually_exclusive_group()
    granularity.add_argument("--per-channel", dest="per_channel", action="store_true", default=True,
                             help="one scale per output channel for weights (default)")
    granularity.add_argument("--per-tensor", dest="per_channel", action="store_false",
                             help="one scale per weight tensor")
    parser.add_argument("--calib-method", choices=CALIBRATION_METHODS, default="minmax")
    parser.add_argument("--percentile", type=float, default=99.999, help="for --calib-method percentile")
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--calib-size", type=int, help="use only this many of the calibration images")
    parser.add_argument("--calib-chunk", type=int, help="collect calibration statistics this many images at a time "
                        "(histogram methods keep every activation in memory otherwise; needed above ~300 images)")
    parser.add_argument("--types", choices=DATA_TYPES, default="s8s8", help="activation and weight types")
    parser.add_argument("--exclude-first-last", action="store_true",
                        help="keep the first Conv and the final Gemm in fp32")
    args = parser.parse_args()
    args.input, args.calib_file = args.input.resolve(), args.calib_file.resolve()

    out = MODELS_DIR / f"{args.name}.onnx"
    if out.exists():
        sys.exit(f"{out.relative_to(ROOT)} already exists; pick another --name (results are never overwritten)")

    preprocessed = MODELS_DIR / f"{args.input.stem}_preprocessed.onnx"
    print(f"[{args.name}] quant_pre_process -> {preprocessed.relative_to(ROOT)}")
    quant_pre_process(str(args.input), str(preprocessed))

    reader = FixedIndexReader(args.calib_file, args.calib_size)
    activation_type, weight_type = DATA_TYPES[args.types]
    nodes_to_exclude = first_and_last_nodes(preprocessed) if args.exclude_first_last else []
    extra_options = {}
    if args.calib_method == "percentile":
        extra_options["CalibPercentile"] = args.percentile
    if args.calib_chunk:
        extra_options["CalibMaxIntermediateOutputs"] = args.calib_chunk  # honoured by MinMax
        if args.calib_method != "minmax":
            chunked_histogram_collection(args.calib_chunk)  # Entropy/Percentile need the patch
    # Activations stay asymmetric (ORT default): post-ReLU6 values are never negative, so an
    # asymmetric range uses all 256 levels. Weights are symmetric per ORT's default.

    print(f"[{args.name}] quantize_static: QDQ, {'per-channel' if args.per_channel else 'per-tensor'}, "
          f"{args.calib_method}, {args.types}, {len(reader.indices)} calibration images"
          + (f", fp32 nodes {nodes_to_exclude}" if nodes_to_exclude else ""))
    quantize_static(
        str(preprocessed),
        str(out),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        reduce_range=False,
        activation_type=activation_type,
        weight_type=weight_type,
        nodes_to_exclude=nodes_to_exclude or None,
        calibrate_method=CALIBRATION_METHODS[args.calib_method],
        extra_options=extra_options or None,
    )
    print(f"[{args.name}] saved {out.relative_to(ROOT)} ({out.stat().st_size / 2**20:.2f} MB), "
          f"{reader.seen} images calibrated")

    settings = {
        "name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_model": str(args.input.relative_to(ROOT)),
        "input_model_sha256": sha256(args.input),
        "preprocessed_model": str(preprocessed.relative_to(ROOT)),
        "quant_format": "QDQ",
        "per_channel": args.per_channel,
        "reduce_range": False,
        "calibration_method": args.calib_method,
        "percentile": args.percentile if args.calib_method == "percentile" else None,
        "calibration_file": str(args.calib_file.relative_to(ROOT)),
        "calibration_file_sha256": sha256(args.calib_file),
        "calibration_images": reader.seen,
        "activation_type": activation_type.name,
        "weight_type": weight_type.name,
        "extra_options": extra_options,
        "nodes_to_exclude": nodes_to_exclude,
        "onnxruntime_version": onnxruntime.__version__,
        "output_model_sha256": sha256(out),
    }
    print(f"[{args.name}] sanity checks")
    settings["checks"] = check(out)
    (MODELS_DIR / f"{args.name}.json").write_text(json.dumps(settings, indent=2))
    if not settings["checks"]["passed"]:
        sys.exit(f"[{args.name}] sanity checks FAILED; see models/{args.name}.json")


if __name__ == "__main__":
    main()
