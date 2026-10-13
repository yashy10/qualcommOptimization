"""Write the fixed image subsets used for quantization calibration and fine-tuning.

    python make_subsets.py

Both files list indices into torchvision's EuroSAT ordering (the same indices
common.load_dataset() uses). They are drawn class by class with one fixed seed, so
rerunning this script reproduces them exactly.
"""
import json
import random
from collections import Counter

from common import ROOT, load_dataset

SEED = 42
CALIBRATION_PER_CLASS = 30  # 10 classes x 30 = 300 images
FINETUNE_FRACTION = 0.2
SUBSETS_DIR = ROOT / "subsets"


def draw(targets, per_class_count):
    """Shuffle each class with the fixed seed and keep the first per_class_count(class size) images."""
    rng = random.Random(SEED)
    chosen = []
    for label in sorted(set(targets)):
        indices = [i for i, target in enumerate(targets) if target == label]
        rng.shuffle(indices)
        chosen += indices[: per_class_count(len(indices))]
    return sorted(chosen)


def write(name, purpose, indices, dataset):
    per_class = Counter(dataset.targets[i] for i in indices)
    out = SUBSETS_DIR / name
    out.write_text(json.dumps({
        "purpose": purpose,
        "dataset": f"torchvision EuroSAT, {len(dataset)} images, indices in torchvision's order",
        "seed": SEED,
        "stratified": True,
        "count": len(indices),
        "per_class": {dataset.classes[label]: per_class[label] for label in sorted(per_class)},
        "indices": indices,
    }))
    print(f"{out.relative_to(ROOT)}: {len(indices)} indices")


def main():
    dataset = load_dataset()
    SUBSETS_DIR.mkdir(exist_ok=True)
    calibration = draw(dataset.targets, lambda class_size: CALIBRATION_PER_CLASS)
    finetune = draw(dataset.targets, lambda class_size: round(class_size * FINETUNE_FRACTION))
    write(
        f"calibration_indices_{len(calibration)}_seed{SEED}.json",
        "calibration images for static quantization",
        calibration,
        dataset,
    )
    calibration_1000 = draw(dataset.targets, lambda class_size: 100)
    write(
        f"calibration_indices_{len(calibration_1000)}_seed{SEED}.json",
        "larger calibration set for the Q6 stability check (100 per class)",
        calibration_1000,
        dataset,
    )
    write(
        f"finetune_indices_{FINETUNE_FRACTION:.0%}_seed{SEED}.json".replace("%", "pct"),
        "training subset for short recovery fine-tuning",
        finetune,
        dataset,
    )


if __name__ == "__main__":
    main()
