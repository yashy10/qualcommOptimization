"""Check that an ONNX export still behaves like the PyTorch model it came from.

    python verify_export.py                    # student checkpoint vs models/student_fp32.onnx
    python verify_export.py --images 1000

Feeds real images through both and compares the raw outputs (logits).
Exits non-zero on any failure, which is the signal to re-run export_onnx.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import NUM_CLASSES, ROOT, get_loaders, load_model
from measure import INPUT_SHAPE

# Largest allowed gap between any PyTorch logit and its ONNX counterpart. Float rounding
# after export is around 1e-5; anything near 1e-3 would mean the graphs really differ.
TOLERANCE = 1e-3


def verify(checkpoint, onnx_path, arch, n_images):
    checks = []

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_input, model_output = session.get_inputs()[0], session.get_outputs()[0]
    checks.append((
        f"ONNX input is one float32 RGB image of shape {INPUT_SHAPE}",
        tuple(model_input.shape) == INPUT_SHAPE and model_input.type == "tensor(float)",
        f"shape {model_input.shape}, {model_input.type}",
    ))
    checks.append((
        f"ONNX output has shape (1, {NUM_CLASSES})",
        tuple(model_output.shape) == (1, NUM_CLASSES),
        f"shape {model_output.shape}",
    ))

    model = load_model(checkpoint, arch)
    # batch_size=1 because the export has a fixed batch dimension of one image
    _, eval_loader, _ = get_loaders(batch_size=1, limit=n_images)
    max_difference = 0.0
    agreeing = total = 0
    classes_seen = set()
    for image, label in eval_loader:
        with torch.no_grad():
            torch_logits = model(image).numpy()
        onnx_logits = session.run(None, {model_input.name: image.numpy()})[0]
        max_difference = max(max_difference, float(np.abs(torch_logits - onnx_logits).max()))
        agreeing += int(torch_logits.argmax() == onnx_logits.argmax())
        total += 1
        classes_seen.add(label.item())

    checks.append((
        f"PyTorch accepts {INPUT_SHAPE} and returns (1, {NUM_CLASSES})",
        tuple(image.shape) == INPUT_SHAPE and torch_logits.shape == (1, NUM_CLASSES),
        f"input {tuple(image.shape)}, output {torch_logits.shape}",
    ))
    checks.append((
        f"logits match within {TOLERANCE:g} on {total} real images",
        max_difference <= TOLERANCE,
        f"largest difference {max_difference:.2e}",
    ))
    checks.append((
        "both models pick the same class for every image",
        agreeing == total,
        f"{agreeing}/{total} agree, {len(classes_seen)} of {NUM_CLASSES} classes covered",
    ))

    for description, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {description}  ({detail})")
    return all(passed for _, passed, _ in checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=ROOT / "student_mobilenetv2_distilled.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models" / "student_fp32.onnx")
    parser.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--images", type=int, default=500, help="real images to compare on")
    args = parser.parse_args()
    sys.exit(0 if verify(args.checkpoint, args.onnx, args.arch, args.images) else 1)


if __name__ == "__main__":
    main()
