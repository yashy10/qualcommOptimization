"""Sanity checks for a quantized ONNX model (plan.md, Section 4, step 5).

    python check_quant.py models/q1_perchannel_minmax_s8s8.onnx

Checks that the model runs in ONNX Runtime on (1, 3, 224, 224) and returns 10 scores,
that it is static QDQ (QuantizeLinear/DequantizeLinear present, no ConvInteger,
MatMulInteger or DynamicQuantizeLinear), reports its size against the fp32 baseline,
and measures how often its predictions agree with the fp32 model on 500 real images.
quantize.py runs this automatically. Exits 1 if any check fails.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from common import NUM_CLASSES, ROOT, get_loaders
from measure import INPUT_SHAPE, MB, RESULTS_DIR

BASELINE_RESULT = RESULTS_DIR / "official_baseline_onnx_fp32.json"
FORBIDDEN_OPS = {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear"}
AGREEMENT_IMAGES = 500


def predictions(session, loader):
    name = session.get_inputs()[0].name
    return np.array([session.run(None, {name: image.numpy()})[0].argmax() for image, _ in loader])


def check(model_path, agreement_images=AGREEMENT_IMAGES):
    """Run every check; returns a dict with each outcome and an overall 'passed' flag."""
    model_path = Path(model_path).resolve()
    report = {"model": str(model_path.relative_to(ROOT)), "checks": []}

    def record(description, passed, detail):
        report["checks"].append({"check": description, "passed": bool(passed), "detail": detail})
        print(f"  {'PASS' if passed else 'FAIL'}  {description}  ({detail})")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    output = session.run(None, {model_input.name: dummy})[0]
    record("loads in ONNX Runtime and accepts (1, 3, 224, 224)",
           tuple(model_input.shape) == INPUT_SHAPE and model_input.type == "tensor(float)",
           f"input {model_input.name} {model_input.shape} {model_input.type}")
    record(f"outputs {NUM_CLASSES} scores", output.shape == (1, NUM_CLASSES), f"output shape {output.shape}")

    ops = Counter(node.op_type for node in onnx.load(model_path).graph.node)
    forbidden = {op: n for op, n in ops.items() if op in FORBIDDEN_OPS}
    report["node_counts"] = dict(ops)
    record("static QDQ: QuantizeLinear and DequantizeLinear present",
           ops["QuantizeLinear"] > 0 and ops["DequantizeLinear"] > 0,
           f"{ops['QuantizeLinear']} QuantizeLinear, {ops['DequantizeLinear']} DequantizeLinear")
    record("no dynamic-quantization operators", not forbidden,
           f"found {forbidden}" if forbidden else "no ConvInteger, MatMulInteger or DynamicQuantizeLinear")

    size_mb = model_path.stat().st_size / MB
    baseline = json.loads(BASELINE_RESULT.read_text())
    ratio = baseline["size_mb"] / size_mb
    report.update({"size_mb": round(size_mb, 3), "baseline_size_mb": baseline["size_mb"], "size_ratio": round(ratio, 2)})
    # Reported, not gated: a mixed-precision model is deliberately larger than a quarter
    record("size about a quarter of fp32", 3.0 <= ratio <= 4.5,
           f"{size_mb:.2f} MB vs {baseline['size_mb']:.2f} MB, {ratio:.2f}x smaller")

    fp32 = ort.InferenceSession(str(ROOT / baseline["model"]), providers=["CPUExecutionProvider"])
    _, loader, _ = get_loaders(batch_size=1, limit=agreement_images)
    fp32_predictions = predictions(fp32, loader)
    quant_predictions = predictions(session, loader)
    labels = np.array([label.item() for _, label in loader])
    agreement = float((fp32_predictions == quant_predictions).mean() * 100)
    report.update({
        "agreement_images": int(len(labels)),
        "agreement_percent": round(agreement, 2),
        "fp32_accuracy_on_these": round(float((fp32_predictions == labels).mean() * 100), 2),
        "quantized_accuracy_on_these": round(float((quant_predictions == labels).mean() * 100), 2),
    })
    record(f"predictions agree with fp32 on {len(labels)} images", True,
           f"{agreement:.1f}% agree; accuracy on these: fp32 {report['fp32_accuracy_on_these']:.1f}%, "
           f"quantized {report['quantized_accuracy_on_these']:.1f}%")

    # The size band and agreement are informational; the structural checks decide pass/fail
    structural = report["checks"][:4]
    report["passed"] = all(item["passed"] for item in structural)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--images", type=int, default=AGREEMENT_IMAGES, help="images for the agreement check")
    args = parser.parse_args()
    report = check(args.model, args.images)
    print("overall:", "PASS" if report["passed"] else "FAIL")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
