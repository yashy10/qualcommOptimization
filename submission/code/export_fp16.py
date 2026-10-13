"""Convert the fp32 ONNX model to fp16 weights and compute (plan.md run Q7).

    python export_fp16.py                       # models/student_fp32.onnx -> models/q7_fp16.onnx

The graph input and output stay float32 (keep_io_types), so the model still accepts
(1, 3, 224, 224) float32 like every other submission candidate. A sidecar JSON records
the settings. Not quantization: this is the cheap middle point on the tradeoff curve.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16

from common import ROOT
from measure import INPUT_SHAPE, MB


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "models" / "student_fp32.onnx")
    parser.add_argument("--name", default="q7_fp16")
    args = parser.parse_args()
    out = ROOT / "models" / f"{args.name}.onnx"
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; pick another --name")

    model = convert_float_to_float16(onnx.load(args.input), keep_io_types=True)
    onnx.save(model, out)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    fp32 = ort.InferenceSession(str(args.input), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    difference = float(np.abs(session.run(None, {"input": dummy})[0] - fp32.run(None, {"input": dummy})[0]).max())
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / MB:.2f} MB, max |fp16 - fp32| logit difference {difference:.3e}")

    (ROOT / "models" / f"{args.name}.json").write_text(json.dumps({
        "name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "onnxruntime.transformers.float16.convert_float_to_float16, keep_io_types=True",
        "input_model": str(args.input.relative_to(ROOT)),
        "input_model_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "onnxruntime_version": ort.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
