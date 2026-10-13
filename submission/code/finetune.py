"""Stage B recovery fine-tune of a pruned model (PLAN_PRUNING.md, Section 8).

    python finetune.py --cache-teacher                                   # once: teacher logits on the 20% subset
    python finetune.py models/stageA/a3_global_r30.pt --name b1_r30_kd --epochs 2
    python finetune.py models/stageA/a3_global_r30.pt --name b1_r30_hard --epochs 2 --no-kd
    python finetune.py ... --probe                                       # A6: time one epoch, save nothing

Recipe: subsets/finetune_subset_20.json (seed 0), no augmentation, SGD momentum 0.9,
lr 0.005 with cosine decay to 0, weight decay 4e-5, batch 32, seed 0.
Loss: (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T)),
T = 4, alpha = 0.7; with --no-kd it is plain cross-entropy. Teacher logits come from
cache/teacher_logits_ft20.pt. After training: eval mode, quick-eval on the 10% subset,
accuracy on the unseen 80% (every image not in the fine-tune subset), save models/<name>.pt
whole plus configs/<name>.json. Official 27,000-image accuracy is measure.py's job.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from common import ROOT, load_dataset, load_model
from prune import CONFIGS_DIR, FINETUNE_SUBSET, MODELS_DIR, count, evaluate, seed_everything

TEACHER_CHECKPOINT = ROOT / "teacher_resnet50_eurosat.pt"
TEACHER_CACHE = ROOT / "cache" / "teacher_logits_ft20.pt"
UNSEEN_FILE = ROOT / "subsets" / "unseen_80_derived.json"


def finetune_indices():
    return json.loads(FINETUNE_SUBSET.read_text())["indices"]


def unseen_file():
    """Indices of every image not in the fine-tune subset (derived once, then fixed)."""
    if not UNSEEN_FILE.exists():
        seen = set(finetune_indices())
        indices = [i for i in range(len(load_dataset())) if i not in seen]
        UNSEEN_FILE.write_text(json.dumps({"purpose": "all images not in finetune_subset_20 (honesty check)",
                                           "count": len(indices), "indices": indices}))
    return UNSEEN_FILE


def cache_teacher_logits():
    teacher = load_model(TEACHER_CHECKPOINT, arch="resnet50")
    indices = finetune_indices()
    loader = DataLoader(Subset(load_dataset(), indices), batch_size=64)
    logits, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits.append(teacher(x))
            labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    TEACHER_CACHE.parent.mkdir(exist_ok=True)
    torch.save({"logits": logits, "labels": labels, "indices": indices,
                "teacher_sha256": hashlib.sha256(TEACHER_CHECKPOINT.read_bytes()).hexdigest()}, TEACHER_CACHE)
    accuracy = (logits.argmax(1) == labels).float().mean().item() * 100
    print(f"cached {len(labels)} teacher logits -> {TEACHER_CACHE.relative_to(ROOT)} in {time.perf_counter() - start:.0f}s "
          f"(teacher accuracy on them {accuracy:.2f}%)")


class WithPosition(Dataset):
    """(image, label, position) so cached teacher logits can be looked up per sample."""

    def __init__(self, indices):
        self.dataset, self.indices = load_dataset(), indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.dataset[self.indices[i]]
        return image, label, i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="?", help="pruned .pt to fine-tune")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--no-kd", action="store_true", help="hard labels only (KD comparison)")
    parser.add_argument("--probe", action="store_true", help="A6: time one epoch and exit without saving")
    parser.add_argument("--cache-teacher", action="store_true")
    parser.add_argument("--no-unseen", action="store_true", help="skip the unseen-80%% check (time pressure)")
    args = parser.parse_args()
    if args.cache_teacher:
        cache_teacher_logits()
        return
    if not args.model or not (args.name or args.probe):
        parser.error("give a model and --name (or --probe)")
    out = MODELS_DIR / f"{args.name}.pt" if args.name else None
    if out and out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; pick another --name")

    seed_everything(0)
    indices = finetune_indices()
    teacher = None
    if not args.no_kd:
        teacher = torch.load(TEACHER_CACHE)
        assert teacher["indices"] == indices, "teacher cache was built for a different fine-tune subset"
    loader = DataLoader(WithPosition(indices), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    model = load_model(args.model)
    epochs = 1 if args.probe else args.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=0)
    T, alpha = args.temperature, args.alpha
    label = args.name or "probe"
    print(f"[{label}] {args.model.name}: {len(indices)} images, {epochs} epoch(s), lr {args.lr}, "
          f"{'hard labels only' if args.no_kd else f'KD alpha {alpha} T {T}'}")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, running = time.perf_counter(), 0.0
        for x, y, pos in loader:
            student_logits = model(x)
            loss = F.cross_entropy(student_logits, y)
            if teacher is not None:
                distill = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                                   F.softmax(teacher["logits"][pos] / T, dim=1), reduction="batchmean") * T * T
                loss = (1 - alpha) * loss + alpha * distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
        seconds = round(time.perf_counter() - start)
        if args.probe:
            print(f"[probe] one epoch of {len(loader)} batches took {seconds}s ({len(indices) / seconds:.1f} images/s)")
            return
        quick = evaluate(model)
        history.append({"epoch": epoch, "mean_loss": round(running / len(loader), 4),
                        "quick_eval_accuracy": quick["accuracy"], "seconds": seconds})
        print(f"[{label}] epoch {epoch}: loss {running / len(loader):.3f}, quick-eval {quick['accuracy']:.2f}% "
              f"(worst {quick['worst_class']} {quick['worst_class_accuracy']:.1f}), {seconds}s")

    model.eval()
    unseen = {"accuracy": None, "images": 0, "per_class_accuracy": None} if args.no_unseen else evaluate(model, unseen_file())
    quick = evaluate(model)
    params, macs = count(model)
    torch.save(model, out)
    source_config = args.model.with_suffix(".json")
    if not source_config.exists():
        source_config = CONFIGS_DIR / f"{args.model.stem}.json"
    config = {
        "name": args.name, "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model": str(args.model.resolve().relative_to(ROOT)),
        "source_config": str(source_config.relative_to(ROOT)) if source_config.exists() else None,
        "fine_tune": {"data": str(FINETUNE_SUBSET.relative_to(ROOT)), "images": len(indices), "augmentation": "none",
                      "optimizer": f"SGD momentum 0.9, lr {args.lr}, cosine to 0, weight decay {args.weight_decay}",
                      "batch_size": args.batch_size, "epochs": epochs, "seed": 0,
                      "loss": "cross-entropy" if args.no_kd else f"(1-{alpha})*CE + {alpha}*T^2*KL, T={T}",
                      "teacher_cache": None if args.no_kd else str(TEACHER_CACHE.relative_to(ROOT))},
        "history": history,
        "quick_eval_accuracy": quick["accuracy"], "quick_eval_worst": [quick["worst_class"], quick["worst_class_accuracy"]],
        "unseen_80_accuracy": unseen["accuracy"], "unseen_80_images": unseen["images"],
        "unseen_80_per_class": unseen["per_class_accuracy"],
        "params_million": params, "macs_giga": macs,
        "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    CONFIGS_DIR.mkdir(exist_ok=True)
    (CONFIGS_DIR / f"{args.name}.json").write_text(json.dumps(config, indent=2))
    print(f"[{label}] saved {out.relative_to(ROOT)} | quick-eval {quick['accuracy']:.2f}% | unseen-80% {unseen['accuracy']} "
          f"| params {params:.3f} M | MACs {macs:.3f} G")


if __name__ == "__main__":
    main()
