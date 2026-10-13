"""Measure one model (.pt or .onnx) with the Section 5 methods and save the result.

    python measure.py student_mobilenetv2_distilled.pt --name official_baseline_pytorch
    python measure.py models/student_fp32.onnx --name official_baseline_onnx_fp32
    python measure.py models/student_fp32.onnx --name try --limit 200         # smoke test

Each run writes results/<name>.json and refreshes results/dashboard.html; compare.py
tabulates the results against the baseline in the terminal.
Accuracy is scored on all of EuroSAT, per judge instruction (no val_split). Smoke tests
(--limit) are written to results/smoke/ so they are never mistaken for full results. An
existing result is not replaced unless --overwrite is given.
Latency, peak RAM and energy are measured in a fresh process, so the accuracy pass
(dataset, DataLoader) does not inflate the memory figures.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
INPUT_SHAPE = (1, 3, 224, 224)
# The judges instructed teams to score on all 27,000 EuroSAT images, with no val_split
SPLIT_LABEL = "full_dataset_judge_instruction"
WARMUP_RUNS = 10
RSS_SAMPLE_INTERVAL_S = 0.01
MB = 1024 * 1024


def make_runner(model_path, arch, threads, process):
    """Return a function that runs one batch-1 inference, and a description of the runtime."""
    if model_path.suffix == ".onnx":
        import numpy as np
        import onnxruntime as ort

        ram_before_model = process.memory_info().rss
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        return lambda: session.run(None, {input_name: dummy}), {
            "runtime": f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
            "thread_setting": f"intra_op_num_threads={threads}" if threads
            else "onnxruntime default (intra_op_num_threads=0)",
            "ram_before_model_mb": round(ram_before_model / MB, 2),
        }

    import torch

    from common import load_model

    ram_before_model = process.memory_info().rss
    if threads:
        torch.set_num_threads(threads)
    model = load_model(model_path, arch)
    dummy = torch.randn(*INPUT_SHAPE)

    def run():
        with torch.no_grad():
            model(dummy)

    return run, {
        "runtime": f"pytorch {torch.__version__} (CPU)",
        "thread_setting": f"torch.set_num_threads({threads})" if threads
        else f"pytorch default ({torch.get_num_threads()} intra-op threads)",
        "ram_before_model_mb": round(ram_before_model / MB, 2),
    }


def runtime_metrics(model_path, arch, runs, energy_seconds, threads):
    """Batch-1 latency, peak RSS and energy per inference (Sections 5.3 to 5.5)."""
    import psutil

    process = psutil.Process(os.getpid())
    run, metrics = make_runner(model_path, arch, threads, process)
    for _ in range(WARMUP_RUNS):
        run()

    # Sample RSS continuously: before/after readings miss the peak in between
    peak_rss = process.memory_info().rss
    stop_sampling = threading.Event()

    def sample_rss():
        nonlocal peak_rss
        while not stop_sampling.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(RSS_SAMPLE_INTERVAL_S)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    stop_sampling.set()
    sampler.join()
    timings.sort()

    mean = sum(timings) / runs
    metrics.update({
        "latency_ms": round(mean * 1000, 3),
        "latency_std_ms": round((sum((t - mean) ** 2 for t in timings) / runs) ** 0.5 * 1000, 3),
        "latency_median_ms": round(timings[runs // 2] * 1000, 3),
        "latency_p95_ms": round(timings[int(runs * 0.95) - 1] * 1000, 3),
        "peak_ram_mb": round(peak_rss / MB, 2),
    })
    try:
        # Lifetime cross-check: bytes on macOS, kilobytes on Linux, unavailable on Windows
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["peak_ram_maxrss_mb"] = round(max_rss / (MB if sys.platform == "darwin" else 1024), 2)
    except ImportError:
        pass

    # Imported only now so codecarbon's own memory stays out of the RAM figures
    from codecarbon import OfflineEmissionsTracker

    tracker = OfflineEmissionsTracker(
        country_iso_code="USA", measure_power_secs=1, log_level="error", save_to_file=False
    )
    tracker.start()
    inferences = 0
    start = time.perf_counter()
    while time.perf_counter() - start < energy_seconds:
        run()
        inferences += 1
    tracker.stop()
    energy = tracker.final_emissions_data

    # codecarbon estimates from CPU load or a TDP constant when it cannot read power sensors
    cpu_modes = {getattr(hardware, "_mode", None) for hardware in tracker._hardware}
    estimated = bool(cpu_modes & {"cpu_load", "constant"})
    metrics.update({
        "joules_per_inference": round(energy.energy_consumed * 3.6e6 / inferences, 5),
        "energy_source": "estimate" if estimated else "sensor",
        "codecarbon_mode": "tdp_estimate" if estimated else "measured",
        "codecarbon_hardware": tracker._conf.get("hardware"),
        "energy_inferences": inferences,
    })
    return metrics


def runtime_metrics_in_fresh_process(args):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "runtime.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), str(args.model),
            "--arch", args.arch, "--runs", str(args.runs),
            "--energy-seconds", str(args.energy_seconds), "--runtime-out", str(out),
        ]
        if args.threads:
            command += ["--threads", str(args.threads)]
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode != 0:
            sys.exit(f"runtime measurement failed:\n{child.stderr}")
        return json.loads(out.read_text())


def torch_static_metrics(model):
    """Parameters and MACs from thop, as in Section 5.2."""
    import copy

    import torch
    from thop import profile

    # thop attaches counters to the modules it profiles, so give it a copy
    macs, params = profile(copy.deepcopy(model), inputs=(torch.randn(*INPUT_SHAPE),), verbose=False)
    return {"params_million": round(params / 1e6, 4), "macs_giga": round(macs / 1e9, 4), "count_method": "thop"}


def onnx_static_metrics(model_path):
    """Parameters and MACs counted from the graph's Conv, Gemm and MatMul nodes.

    thop cannot read ONNX. Weights are followed through DequantizeLinear, so QDQ-quantized
    models are counted the same way as float ones. Compare ONNX figures with ONNX figures:
    export folds BatchNorm into the convolutions, which thop counts separately.
    """
    import onnx

    graph = onnx.shape_inference.infer_shapes(onnx.load(model_path)).graph
    initializer_dims = {tensor.name: list(tensor.dims) for tensor in graph.initializer}
    output_shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in [*graph.value_info, *graph.output]
    }
    dequantized = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "DequantizeLinear"}

    def weight_dims(name):
        return initializer_dims.get(dequantized.get(name, name))

    params = macs = 0
    for node in graph.node:
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue
        weight = weight_dims(node.input[1])
        if weight is None:
            continue
        params += sum(prod(weight_dims(name) or [0]) for name in node.input[1:])
        if node.op_type == "Conv":
            # output elements per image x (input channels per group x kernel area)
            macs += prod(output_shapes[node.output[0]][1:]) * prod(weight[1:])
        else:
            macs += prod(weight)
    return {"params_million": round(params / 1e6, 4), "macs_giga": round(macs / 1e9, 4), "count_method": "onnx graph"}


def accuracy_and_static_metrics(model_path, arch, limit):
    """Overall and per-class accuracy over the dataset, plus parameters, MACs and size."""
    import torch

    from common import get_loaders, load_model

    if model_path.suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        # A model exported with a fixed batch dimension must be fed exactly that many images
        batch_size = model_input.shape[0] if isinstance(model_input.shape[0], int) else 32

        def predict(images):
            logits = session.run(None, {model_input.name: images.numpy()})[0]
            return torch.from_numpy(logits).argmax(dim=1)

        metrics = onnx_static_metrics(model_path)
    else:
        model = load_model(model_path, arch)
        batch_size = 32

        def predict(images):
            with torch.no_grad():
                return model(images).argmax(dim=1)

        metrics = torch_static_metrics(model)

    _, eval_loader, classes = get_loaders(batch_size, limit)
    n = len(classes)
    confusion = torch.zeros(n, n, dtype=torch.long)
    for images, labels in eval_loader:
        confusion += torch.bincount(labels * n + predict(images), minlength=n * n).view(n, n)
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    per_class = {name: round(value.item() * 100, 2) for name, value in zip(classes, recall)}
    worst_class = min(per_class, key=per_class.get)

    metrics.update({
        "size_mb": round(model_path.stat().st_size / MB, 3),
        "accuracy": round(confusion.diag().sum().item() / confusion.sum().item() * 100, 2),
        "worst_class": worst_class,
        "worst_class_accuracy": per_class[worst_class],
        "per_class_accuracy": per_class,
        "split": SPLIT_LABEL if limit is None else "smoke_subset",
        "images_evaluated": int(confusion.sum()),
    })
    return metrics


def machine_description():
    import psutil

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu = brand.stdout.strip() or cpu
    ram_gb = round(psutil.virtual_memory().total / 2**30)
    return f"{cpu}, {psutil.cpu_count(logical=False)} cores, {ram_gb} GB RAM, {platform.system()} {platform.release()}, CPU only"


def measurement_conditions():
    """What else the machine was doing: background load and power source skew latency and energy."""
    import psutil

    battery = psutil.sensors_battery()
    return {
        "background_cpu_percent": psutil.cpu_percent(interval=1.0),
        "on_ac_power": battery.power_plugged if battery else None,
    }


def environment():
    from importlib.metadata import version

    packages = ["torch", "torchvision", "onnx", "onnxruntime", "thop", "codecarbon", "psutil", "numpy"]
    return {"python": platform.python_version(), **{package: version(package) for package in packages}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help=".pt checkpoint or .onnx file")
    parser.add_argument("--name", help="result name (default: the model file's stem)")
    parser.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"],
                        help="architecture a .pt state dict belongs to")
    parser.add_argument("--limit", type=int, help="smoke test: score only this many images")
    parser.add_argument("--runs", type=int, default=100, help="timed inferences for latency")
    parser.add_argument("--energy-seconds", type=float, default=15, help="how long to run inference under codecarbon")
    parser.add_argument("--threads", type=int, help="CPU threads for inference (default: the runtime's own default)")
    parser.add_argument("--what", help='what was done to this model, shown on the dashboard, '
                                       'e.g. "structured pruning 30%% + static int8, 300 calibration images"')
    parser.add_argument("--timing", choices=["provisional", "milestone"], default="provisional",
                        help="milestone = clean session, nothing else running (default: provisional)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing result of the same name")
    parser.add_argument("--runtime-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.runtime_out:
        metrics = runtime_metrics(args.model, args.arch, args.runs, args.energy_seconds, args.threads)
        args.runtime_out.write_text(json.dumps(metrics))
        return

    name = args.name or args.model.stem
    smoke_test = args.limit is not None
    out = (RESULTS_DIR / "smoke" if smoke_test else RESULTS_DIR) / f"{name}.json"
    if out.exists() and not (smoke_test or args.overwrite):
        sys.exit(f"{out.relative_to(ROOT)} already exists; choose another --name or pass --overwrite")

    result = {
        "name": name,
        "what_we_did": args.what,
        "result_type": "local measurement (ours, not an organizer-provided number)",
        "smoke_test": smoke_test,
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "format": "onnx" if args.model.suffix == ".onnx" else "pytorch",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": measurement_conditions(),
        "timing_status": args.timing,
    }
    # Runtime first, while the machine is still cool from idle
    print(f"[{name}] latency, peak RAM and energy ...")
    result.update(runtime_metrics_in_fresh_process(args))
    print(f"[{name}] accuracy over the dataset ...")
    result.update(accuracy_and_static_metrics(args.model, args.arch, args.limit))
    result["method"] = {
        "input": f"one image per inference, shape {INPUT_SHAPE}; random values for timing, real images for accuracy",
        "accuracy": "top-1 over one pass of the evaluation set, Section 4 preprocessing",
        "params_and_macs": result["count_method"],
        "latency": f"mean of {args.runs} timed inferences after {WARMUP_RUNS} warm-up inferences",
        "peak_ram": f"highest process RSS, sampled every {RSS_SAMPLE_INTERVAL_S * 1000:g} ms during the timed inferences (psutil)",
        "energy": f"codecarbon over {args.energy_seconds:g} s of back-to-back inferences, divided by the inference count",
    }
    result["machine"] = machine_description()
    result["environment"] = environment()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"  accuracy     {result['accuracy']:.2f}%  (worst class {result['worst_class']}: "
          f"{result['worst_class_accuracy']:.2f}%, {result['images_evaluated']} images, split: {result['split']})")
    print(f"  params       {result['params_million']:.3f} M | MACs {result['macs_giga']:.3f} G "
          f"({result['count_method']}) | size {result['size_mb']:.2f} MB")
    print(f"  latency      {result['latency_ms']:.2f} ms mean (std {result['latency_std_ms']:.2f}) "
          f"| peak RAM {result['peak_ram_mb']:.1f} MB | {result['runtime']}, {result['thread_setting']}")
    print(f"  energy       {result['joules_per_inference']:.4f} J/inference ({result['energy_source']})")
    print(f"  saved to     {out.relative_to(ROOT)}")

    # The measurement is already saved, so a dashboard problem must not look like a failed run
    try:
        from dashboard import build_dashboard

        print(f"  dashboard    {build_dashboard().relative_to(ROOT)}")
    except Exception as error:
        print(f"  dashboard    not updated: {error!r}")


if __name__ == "__main__":
    main()
