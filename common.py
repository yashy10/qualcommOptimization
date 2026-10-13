"""Dataset and model loading shared by the challenge scripts."""
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 10

# Preprocessing from Section 4 of the challenge packet
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dataset():
    return datasets.EuroSAT(root=ROOT / "data", download=False, transform=transform)


def get_loaders(batch_size=32, limit=None):
    """Training and evaluation loaders and the class names.

    Per judge instruction the dataset is not split: both loaders cover all of EuroSAT.
    limit keeps an evenly spaced (so still class-balanced) subset of the evaluation
    images for quick checks.
    """
    dataset = load_dataset()
    indices = list(range(len(dataset)))
    if limit and limit < len(indices):
        indices = indices[:: len(indices) // limit][:limit]
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
    return train_loader, eval_loader, dataset.classes


def build_model(arch="mobilenet_v2"):
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return model


def load_model(path, arch="mobilenet_v2"):
    """Load a state dict into the stock architecture, or a whole pickled nn.Module as-is."""
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # A model saved whole (e.g. after structured pruning changed its layer shapes)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        # thop leaves float64 counting buffers on profiled modules; drop them (they break MPS and bloat exports)
        for module in checkpoint.modules():
            for name in ("total_ops", "total_params"):
                if name in module._buffers:
                    del module._buffers[name]
        return checkpoint.eval()
    model = build_model(arch)
    model.load_state_dict(checkpoint)
    return model.eval()
