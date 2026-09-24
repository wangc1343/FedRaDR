from __future__ import annotations

from torch import nn
from torchvision import models


def _group_norm(channels: int) -> nn.GroupNorm:
    # The official FedDC ResNet-18 implementation uses two groups.
    return nn.GroupNorm(2 if channels % 2 == 0 else 1, channels)


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _group_norm(child.num_features))
        else:
            _replace_batch_norm(child)


def make_model(name: str, num_classes: int) -> nn.Module:
    """Create the architectures used by the manuscript with GN."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif normalized == "mobilenetv2":
        model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"unsupported model: {name}")
    _replace_batch_norm(model)
    return model
