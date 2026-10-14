"""Classify EuroSAT images with an ONNX model and print the prediction, truth and time per image.

    python predict.py student_model_int8.onnx                 # 10 random images
    python predict.py student_model_int8.onnx --count 25 --seed 3
    python predict.py student_model_int8.onnx --image path/to/tile.jpg
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import load_dataset, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", type=Path, help="classify one image file instead of dataset samples")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dataset = load_dataset()
    classes = dataset.classes
    print(f"model {args.model.name} ({args.model.stat().st_size / 2**20:.2f} MB), input {session.get_inputs()[0].shape}\n")

    if args.image:
        samples = [(transform(Image.open(args.image).convert("RGB")), None, str(args.image))]
    else:
        rng = random.Random(args.seed)
        picks = rng.sample(range(len(dataset)), args.count)
        samples = [(dataset[i][0], dataset[i][1], f"image #{i}") for i in picks]

    session.run(None, {name: np.zeros((1, 3, 224, 224), np.float32)})  # warm-up
    correct = 0
    for tensor, label, title in samples:
        x = tensor.unsqueeze(0).numpy()
        start = time.perf_counter()
        logits = session.run(None, {name: x})[0][0]
        ms = (time.perf_counter() - start) * 1000
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        predicted = int(logits.argmax())
        truth = "" if label is None else f"  truth {classes[label]:20s} {'OK' if predicted == label else 'WRONG'}"
        correct += int(predicted == label)
        print(f"{title:12s} -> {classes[predicted]:20s} {probabilities[predicted] * 100:5.1f}%  {ms:5.2f} ms{truth}")
    if not args.image:
        print(f"\n{correct}/{len(samples)} correct")


if __name__ == "__main__":
    main()
