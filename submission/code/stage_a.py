"""Stage A of PLAN_PRUNING.md: cheap pruning sweeps, no fine-tuning.

    python stage_a.py a1        # block scan: remove each residual block alone
    python stage_a.py a2        # block stack: remove the 1..6 least damaging blocks
    python stage_a.py a3        # global channel sweep 0.1 / 0.2 / 0.3 / 0.4
    python stage_a.py a4        # uniform channel sweep (control)
    python stage_a.py a5        # criterion check at 0.3 global
    python stage_a.py all

Each candidate: prune -> adaptive BN -> quick-eval on the 10% subset (provisional) ->
export check (fp32 ONNX, agreement with PyTorch on 500 images) -> row in results/stageA/.
Models go to models/stageA/, configs to configs/. Nothing here is timed.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import ROOT, get_loaders
from export_onnx import export
from prune import CONFIGS_DIR, build, count, evaluate, make_config, residual_block_ids, save

STAGE_DIR = ROOT / "results" / "stageA"
MODELS_STAGE_DIR = ROOT / "models" / "stageA"
AGREEMENT_IMAGES = 500
MB = 1024 * 1024


def export_check(model, onnx_path):
    """Export to fp32 ONNX and compare predictions with the PyTorch model on 500 images."""
    export(None, onnx_path, model=model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, loader, _ = get_loaders(batch_size=1, limit=AGREEMENT_IMAGES)
    agree = total = 0
    with torch.no_grad():
        for image, _ in loader:
            torch_prediction = model(image).argmax(1).item()
            onnx_prediction = session.run(None, {"input": image.numpy()})[0].argmax()
            agree += int(torch_prediction == onnx_prediction)
            total += 1
    model_input = session.get_inputs()[0]
    return {"ok": agree == total and tuple(model_input.shape) == (1, 3, 224, 224),
            "agreement": f"{agree}/{total}", "size_mb": round(onnx_path.stat().st_size / MB, 3),
            "onnx": str(onnx_path.relative_to(ROOT))}


def run_candidate(name, what, remove_blocks=(), ratio=0.0, global_pruning=False, criterion="magnitude", group=""):
    row_path = STAGE_DIR / f"{name}.json"
    if row_path.exists():
        print(f"[{name}] already done, skipping")
        return json.loads(row_path.read_text())
    start = time.perf_counter()
    config = make_config(name, list(remove_blocks), ratio, global_pruning, criterion)
    model = build(config)
    params, macs = count(model)
    result = evaluate(model)
    MODELS_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_STAGE_DIR / f"{name}.pt"
    torch.save(model, model_path)
    config = {**config, "created_at": datetime.now().isoformat(timespec="seconds"),
              "model": str(model_path.relative_to(ROOT)),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    CONFIGS_DIR.mkdir(exist_ok=True)
    (CONFIGS_DIR / f"{name}.json").write_text(json.dumps(config, indent=2))
    check = export_check(model, MODELS_STAGE_DIR / f"{name}_fp32.onnx")
    row = {
        "name": name, "stage": "A", "group": group, "what_we_did": what,
        "result_type": "provisional (quick-eval on the 10% subset, no fine-tuning)",
        "split": "quickeval_subset_10", "images_evaluated": result["images"],
        "accuracy": result["accuracy"], "worst_class": result["worst_class"],
        "worst_class_accuracy": result["worst_class_accuracy"], "per_class_accuracy": result["per_class_accuracy"],
        "params_million": params, "macs_giga": macs, "count_method": "thop on the PyTorch model",
        "size_mb": check["size_mb"], "export_ok": check["ok"], "export_agreement": check["agreement"],
        "model": str(model_path.relative_to(ROOT)), "onnx": check["onnx"], "config": f"configs/{name}.json",
        "remove_blocks": config["remove_blocks"], "width": config["width"],
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - start),
    }
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(row, indent=2))
    print(f"[{name}] {result['accuracy']:.2f}% (worst {result['worst_class']} {result['worst_class_accuracy']:.1f}) | "
          f"params {params:.3f} M | MACs {macs:.3f} G | {check['size_mb']:.2f} MB | export {'OK' if check['ok'] else 'FAIL'} "
          f"{check['agreement']} | {row['seconds']}s")
    return row


def reference():
    return run_candidate("a0_reference_adaptive_bn", "A0: unpruned student, adaptive BN only (reference for the quick-eval subset)", group="A0")


def a1():
    rows = [reference()]
    for i in residual_block_ids(build({"remove_blocks": [], "width": None, "adaptive_bn": False})):
        rows.append(run_candidate(f"a1_block{i}", f"A1: remove features[{i}] (residual block) alone, adaptive BN", remove_blocks=[i], group="A1"))
    return rows


def block_ranking():
    ref = json.loads((STAGE_DIR / "a0_reference_adaptive_bn.json").read_text())["accuracy"]
    rows = [json.loads(p.read_text()) for p in sorted(STAGE_DIR.glob("a1_block*.json"))]
    ranked = sorted(rows, key=lambda r: ref - r["accuracy"])
    return [(r["remove_blocks"][0], round(ref - r["accuracy"], 2)) for r in ranked]


def a2():
    ranking = block_ranking()
    print("A1 ranking (block, accuracy drop):", ranking)
    rows = []
    for k in range(1, 7):
        blocks = sorted(b for b, _ in ranking[:k])
        rows.append(run_candidate(f"a2_blocks{k}", f"A2: remove the {k} least damaging blocks {blocks}, adaptive BN",
                                  remove_blocks=blocks, group="A2"))
    return rows


def a3():
    return [run_candidate(f"a3_global_r{int(r * 100)}", f"A3: global channel pruning ratio {r}, group magnitude p=2, isomorphic, round_to 8, adaptive BN",
                          ratio=r, global_pruning=True, group="A3") for r in (0.1, 0.2, 0.3, 0.4)]


def a4():
    return [run_candidate(f"a4_uniform_r{int(r * 100)}", f"A4: uniform channel pruning ratio {r}, group magnitude p=2, round_to 8, adaptive BN",
                          ratio=r, global_pruning=False, group="A4") for r in (0.1, 0.2, 0.3, 0.4)]


def a5():
    rows = []
    for criterion in ("bnscale", "taylor"):
        rows.append(run_candidate(f"a5_global_r30_{criterion}", f"A5: global channel pruning ratio 0.3 with {criterion} importance, adaptive BN",
                                  ratio=0.3, global_pruning=True, criterion=criterion, group="A5"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=["a1", "a2", "a3", "a4", "a5", "all"])
    args = parser.parse_args()
    parts = {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5}
    for part in (parts if args.part == "all" else [args.part]):
        print(f"=== {part.upper()} ===")
        parts[part]()
    try:
        from dashboard import build_dashboard

        build_dashboard()
    except Exception as error:
        print(f"dashboard not updated: {error!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
