"""Structured pruning of the MobileNetV2 student with torch-pruning (PLAN_PRUNING.md).

    python prune.py --config configs/<name>.json            # regenerate a candidate exactly
    python prune.py --name a3_global_r30 --ratio 0.3 --global --criterion magnitude
    python prune.py --name a2_blocks3 --remove-blocks 10 9 12

Two kinds of pruning, applied in this order:
  depth  - whole inverted-residual blocks with a residual connection are replaced by
           nn.Identity (block input and output shapes are equal, so nothing else changes);
  width  - channels removed with torch-pruning's dependency graph (coupled layers pruned
           together), classifier never pruned, round_to=8, global or uniform ratio,
           importance = group magnitude (p=2), BN scale, or Taylor.
Then adaptive BN: BatchNorm running statistics are recomputed on 100 batches of the
fine-tune subset. Everything is seeded with 0 and recorded in configs/<name>.json so
`--config` rebuilds the same model from the original checkpoint. The model is saved whole
(torch.save(model)) to models/<name>.pt.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from common import ROOT, get_loaders, load_dataset, load_model

MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
STUDENT = ROOT / "student_mobilenetv2_distilled.pt"
FINETUNE_SUBSET = ROOT / "subsets" / "finetune_subset_20.json"
QUICKEVAL_SUBSET = ROOT / "subsets" / "quickeval_subset_10.json"
SEED = 0
ROUND_TO = 8
ADAPTIVE_BN_BATCHES = 100
ADAPTIVE_BN_BATCH_SIZE = 32
TAYLOR_BATCHES = 10
CRITERIA = ("magnitude", "bnscale", "taylor")
EXAMPLE_INPUT = (1, 3, 224, 224)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_loader(subset_file, batch_size, shuffle=False, seed=SEED):
    indices = json.loads(Path(subset_file).read_text())["indices"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(load_dataset(), indices), batch_size=batch_size, shuffle=shuffle, generator=generator)


def residual_block_ids(model):
    """Indices in model.features of inverted-residual blocks with a residual connection."""
    return [i for i, block in enumerate(model.features) if getattr(block, "use_res_connect", False)]


def remove_blocks(model, block_ids):
    removable = residual_block_ids(model)
    for i in block_ids:
        if i not in removable:
            raise ValueError(f"features[{i}] has no residual connection and cannot be removed")
        model.features[i] = nn.Identity()


def importance_for(criterion, model):
    if criterion == "magnitude":
        return tp.importance.GroupMagnitudeImportance(p=2)
    if criterion == "bnscale":
        return tp.importance.BNScaleImportance()
    if criterion == "taylor":
        # Gradients of the cross-entropy loss over a few fine-tune batches, accumulated before pruning
        model.train()
        model.zero_grad()
        loader = subset_loader(FINETUNE_SUBSET, ADAPTIVE_BN_BATCH_SIZE, shuffle=True)
        for step, (x, y) in enumerate(loader):
            if step == TAYLOR_BATCHES:
                break
            F.cross_entropy(model(x), y).backward()
        model.eval()
        return tp.importance.GroupTaylorImportance()
    raise ValueError(criterion)


def prune_channels(model, ratio, global_pruning, criterion):
    """Width pruning with torch-pruning. Autograd must be on while the dependency graph is built."""
    importance = importance_for(criterion, model)
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=torch.randn(*EXAMPLE_INPUT),
        importance=importance,
        global_pruning=global_pruning,
        pruning_ratio=ratio,
        isomorphic=global_pruning,
        round_to=ROUND_TO,
        ignored_layers=[model.classifier[1]],
    )
    pruner.step()
    model.zero_grad(set_to_none=True)


def adaptive_bn(model):
    """Recompute BatchNorm running statistics on the fine-tune subset (EagleEye-style)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over the batches seen
    model.train()
    loader = subset_loader(FINETUNE_SUBSET, ADAPTIVE_BN_BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for step, (x, _) in enumerate(loader):
            if step == ADAPTIVE_BN_BATCHES:
                break
            model(x)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1
    model.eval()


def count(model):
    import copy

    from thop import profile

    # thop attaches counting buffers to the modules it profiles, so give it a copy
    macs, params = profile(copy.deepcopy(model), inputs=(torch.randn(*EXAMPLE_INPUT),), verbose=False)
    return round(params / 1e6, 4), round(macs / 1e9, 4)


def evaluate(model, subset_file=QUICKEVAL_SUBSET, batch_size=64):
    """Accuracy, per-class accuracy and worst class on one subset (provisional, not official)."""
    classes = load_dataset().classes
    loader = subset_loader(subset_file, batch_size)
    n = len(classes)
    confusion = torch.zeros(n, n, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            confusion += torch.bincount(y * n + model(x).argmax(1), minlength=n * n).view(n, n)
    per_class = {c: round(v.item() * 100, 2) for c, v in zip(classes, confusion.diag().float() / confusion.sum(1).clamp(min=1))}
    worst = min(per_class, key=per_class.get)
    return {"accuracy": round(confusion.diag().sum().item() / confusion.sum().item() * 100, 2),
            "per_class_accuracy": per_class, "worst_class": worst, "worst_class_accuracy": per_class[worst],
            "images": int(confusion.sum())}


def quick_accuracy(model, images):
    """Kept for finetune.py: top-1 on an evenly spaced subset of the whole dataset."""
    _, loader, _ = get_loaders(batch_size=64, limit=images)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total * 100


def build(config):
    """Rebuild a pruned model from the original checkpoint and a config dict."""
    seed_everything(config.get("seed", SEED))
    model = load_model(STUDENT)
    if config.get("remove_blocks"):
        remove_blocks(model, config["remove_blocks"])
    width = config.get("width")
    if width and width["ratio"] > 0:
        prune_channels(model, width["ratio"], width["global"], width["criterion"])
    if config.get("adaptive_bn", True):
        adaptive_bn(model)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(*EXAMPLE_INPUT)).shape == (1, 10)
    return model


def make_config(name, remove_blocks_ids, ratio, global_pruning, criterion, adaptive=True):
    return {
        "name": name,
        "checkpoint": str(STUDENT.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(STUDENT.read_bytes()).hexdigest(),
        "seed": SEED,
        "remove_blocks": sorted(remove_blocks_ids or []),
        "width": {"ratio": ratio, "global": global_pruning, "criterion": criterion, "round_to": ROUND_TO,
                  "isomorphic": global_pruning, "taylor_batches": TAYLOR_BATCHES if criterion == "taylor" else None,
                  "ignored_layers": ["classifier.1"]} if ratio else None,
        "adaptive_bn": {"batches": ADAPTIVE_BN_BATCHES, "batch_size": ADAPTIVE_BN_BATCH_SIZE,
                        "subset": str(FINETUNE_SUBSET.relative_to(ROOT))} if adaptive else False,
        "torch_pruning": "1.6.1",
    }


def save(model, config):
    CONFIGS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"{config['name']}.pt"
    torch.save(model, out)
    config = {**config, "created_at": datetime.now().isoformat(timespec="seconds"),
              "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (CONFIGS_DIR / f"{config['name']}.json").write_text(json.dumps(config, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="rebuild from configs/<name>.json")
    parser.add_argument("--name")
    parser.add_argument("--ratio", type=float, default=0.0, help="channel pruning ratio")
    parser.add_argument("--global", dest="global_pruning", action="store_true")
    parser.add_argument("--criterion", choices=CRITERIA, default="magnitude")
    parser.add_argument("--remove-blocks", type=int, nargs="*", default=[], help="features[i] indices to replace with Identity")
    parser.add_argument("--no-adaptive-bn", action="store_true")
    parser.add_argument("--eval", action="store_true", help="quick-eval on the 10%% subset after pruning")
    args = parser.parse_args()

    if args.config:
        config = json.loads(args.config.read_text())
    elif args.name:
        config = make_config(args.name, args.remove_blocks, args.ratio, args.global_pruning, args.criterion,
                             not args.no_adaptive_bn)
    else:
        parser.error("give --config or --name")
    out = MODELS_DIR / f"{config['name']}.pt"
    if out.exists() and not args.config:
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; pick another --name")

    model = build(config)
    params, macs = count(model)
    print(f"[{config['name']}] params {params:.3f} M | MACs {macs:.3f} G | blocks removed {config['remove_blocks']} | "
          f"width {config['width']}")
    if args.eval:
        result = evaluate(model)
        print(f"[{config['name']}] quick-eval {result['accuracy']:.2f}% (worst {result['worst_class']} "
              f"{result['worst_class_accuracy']:.2f}%, {result['images']} images, provisional)")
    save(model, config)


if __name__ == "__main__":
    main()
