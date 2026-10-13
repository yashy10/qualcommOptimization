"""Export a PyTorch checkpoint to ONNX with the Section 6.2 settings.

    python export_onnx.py                                      # student -> models/student_fp32.onnx
    python export_onnx.py pruned.pt -o models/pruned_fp32.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import ROOT, load_model
from measure import INPUT_SHAPE, MB


def export(checkpoint, output, arch="mobilenet_v2", model=None):
    """Export a checkpoint path, or an already-built model when `model` is given."""
    model = model.eval() if model is not None else load_model(checkpoint, arch)
    dummy_input = torch.randn(*INPUT_SHAPE)
    output.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False keeps the TorchScript exporter, which honours opset_version=13
    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))

    # The exported graph must reproduce the PyTorch outputs
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(dummy_input).numpy()
    actual = session.run(None, {"input": dummy_input.numpy()})[0]
    print(f"{output}: {output.stat().st_size / MB:.2f} MB, max |onnx - torch| = {np.abs(actual - expected).max():.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=ROOT / "student_mobilenetv2_distilled.pt")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "models" / "student_fp32.onnx")
    parser.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.arch)


if __name__ == "__main__":
    main()
