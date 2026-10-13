"""Structured channel pruning of the MobileNetV2 student (plan.md, Section 6).

    python prune.py --ratio 0.3 --name pruned_r30
    python prune.py --ratio 0.3 --name pruned_r30 --eval-images 2000     # quick accuracy check

Prunes the hidden (expanded) channels inside every inverted-residual block: the expand
1x1 conv's output channels, the matching depthwise 3x3 channels, and the project 1x1
conv's input channels are removed together. Block inputs and outputs keep their shapes,
so the residual connections are untouched and the model stays exportable. Channels are
ranked by |gamma| of the depthwise BatchNorm; the lowest `ratio` are dropped, rounded so
each block keeps a multiple of 8 channels. Whole channels are removed, so parameter
count, MACs and file size all fall (unstructured pruning would not change them).

Saves models/<name>.pt (the whole module, since layer shapes changed) and a sidecar
JSON. Export with export_onnx.py, then quantize.py, as for the unpruned model.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torchvision.models.mobilenetv2 import InvertedResidual

from common import ROOT, get_loaders, load_model

MODELS_DIR = ROOT / "models"
CHANNEL_MULTIPLE = 8
MIN_CHANNELS = 8


def prunable_blocks(model):
    """Inverted-residual blocks that have an expand conv (features[1] has none)."""
    return [(name, block) for name, block in model.features.named_children()
            if isinstance(block, InvertedResidual) and len(block.conv) == 4]


CRITERIA = ("gamma_beta", "gamma")


def channels_to_keep(block, ratio, criterion="gamma_beta"):
    """Rank hidden channels by the depthwise BatchNorm: |gamma| + |beta| by default.

    |gamma| alone was the worst criterion in the no-fine-tune sweep (14% vs 71% at 10%
    pruning): a channel with small gamma but large beta still feeds a constant offset
    into the next layer, and removing it shifts everything downstream.
    """
    bn = block.conv[1][1]
    score = bn.weight.detach().abs()
    if criterion == "gamma_beta":
        score = score + bn.bias.detach().abs()
    hidden = score.numel()
    keep = max(MIN_CHANNELS, int(round(hidden * (1 - ratio) / CHANNEL_MULTIPLE)) * CHANNEL_MULTIPLE)
    keep = min(keep, hidden)
    return score.topk(keep).indices.sort().values


def slim_conv(conv, out_idx=None, in_idx=None):
    """A copy of a 1x1 (groups=1) conv with only the given output and/or input channels."""
    weight = conv.weight.detach()
    if out_idx is not None:
        weight = weight[out_idx]
    if in_idx is not None:
        weight = weight[:, in_idx]
    new = nn.Conv2d(weight.shape[1], weight.shape[0], conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, bias=conv.bias is not None)
    new.weight = nn.Parameter(weight.clone())
    if conv.bias is not None:
        new.bias = nn.Parameter(conv.bias.detach()[out_idx].clone() if out_idx is not None else conv.bias.detach().clone())
    return new


def slim_bn(bn, idx):
    new = nn.BatchNorm2d(len(idx), eps=bn.eps, momentum=bn.momentum)
    new.weight = nn.Parameter(bn.weight.detach()[idx].clone())
    new.bias = nn.Parameter(bn.bias.detach()[idx].clone())
    new.running_mean = bn.running_mean[idx].clone()
    new.running_var = bn.running_var[idx].clone()
    return new


def prune_block(block, keep):
    expand, depthwise, project, project_bn = block.conv[0], block.conv[1], block.conv[2], block.conv[3]
    expand[0] = slim_conv(expand[0], out_idx=keep)
    expand[1] = slim_bn(expand[1], keep)
    dw = depthwise[0]
    new_dw = nn.Conv2d(len(keep), len(keep), dw.kernel_size, dw.stride, dw.padding, dw.dilation, groups=len(keep), bias=False)
    new_dw.weight = nn.Parameter(dw.weight.detach()[keep].clone())
    depthwise[0] = new_dw
    depthwise[1] = slim_bn(depthwise[1], keep)
    block.conv[2] = slim_conv(project, in_idx=keep)
    block.conv[3] = project_bn


def count(model):
    from thop import profile

    macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    return params / 1e6, macs / 1e9


def quick_accuracy(model, images):
    _, loader, _ = get_loaders(batch_size=64, limit=images)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratio", type=float, required=True, help="fraction of hidden channels to drop per block")
    parser.add_argument("--name", required=True, help="output: models/<name>.pt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "student_mobilenetv2_distilled.pt")
    parser.add_argument("--criterion", choices=CRITERIA, default="gamma_beta")
    parser.add_argument("--eval-images", type=int, help="quick accuracy on this many images (not official)")
    args = parser.parse_args()
    out = MODELS_DIR / f"{args.name}.pt"
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; pick another --name")

    model = load_model(args.checkpoint)
    before = count(model)
    plan = {}
    for name, block in prunable_blocks(model):
        keep = channels_to_keep(block, args.ratio, args.criterion)
        plan[f"features.{name}"] = {"hidden": block.conv[1][1].num_features, "kept": len(keep)}
        prune_block(block, keep)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 10)
    after = count(model)
    torch.save(model, out)

    print(f"[{args.name}] ratio {args.ratio:.2f}: params {before[0]:.3f} -> {after[0]:.3f} M, "
          f"MACs {before[1]:.3f} -> {after[1]:.3f} G, {out.stat().st_size / 2**20:.2f} MB")
    sidecar = {
        "name": args.name, "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "method": "structured pruning of inverted-residual hidden channels, ranked by the depthwise BatchNorm "
                  f"({'|gamma| + |beta|' if args.criterion == 'gamma_beta' else '|gamma|'}), "
                  f"uniform ratio per block, kept channels rounded to a multiple of {CHANNEL_MULTIPLE}",
        "criterion": args.criterion,
        "ratio": args.ratio, "fine_tuned": False,
        "params_million_before": round(before[0], 4), "params_million_after": round(after[0], 4),
        "macs_giga_before": round(before[1], 4), "macs_giga_after": round(after[1], 4),
        "blocks": plan,
    }
    if args.eval_images:
        start = time.perf_counter()
        sidecar["quick_accuracy"] = {"images": args.eval_images, "accuracy": round(quick_accuracy(model, args.eval_images), 2),
                                     "note": "subset check, not the official 27,000-image figure"}
        print(f"[{args.name}] quick accuracy on {args.eval_images} images: {sidecar['quick_accuracy']['accuracy']:.2f}% "
              f"({time.perf_counter() - start:.0f}s)")
    (MODELS_DIR / f"{args.name}.json").write_text(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
