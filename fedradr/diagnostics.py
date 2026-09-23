from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ForgettingMetrics:
    mean_forgetting_gap: float
    high_forgetting_ratio: float
    sample_count: int


@torch.no_grad()
def forgetting_metrics(
    before: nn.Module,
    after: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: torch.device,
) -> ForgettingMetrics:
    gaps = []
    before.eval()
    after.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).reshape(-1).long()
        before_loss = F.cross_entropy(before(inputs), targets, reduction="none")
        after_loss = F.cross_entropy(after(inputs), targets, reduction="none")
        gaps.append((after_loss - before_loss).clamp_min(0).cpu())
    values = torch.cat(gaps) if gaps else torch.empty(0)
    return ForgettingMetrics(
        mean_forgetting_gap=float(values.mean()) if values.numel() else 0.0,
        high_forgetting_ratio=float((values > threshold).float().mean()) if values.numel() else 0.0,
        sample_count=values.numel(),
    )


@torch.no_grad()
def absent_class_retention(
    before: nn.Module,
    after: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    absent_classes: list[int],
    device: torch.device,
) -> dict[str, float]:
    if not absent_classes:
        return {"probability_mass_retention": 1.0, "records": 0.0}
    before.eval()
    after.eval()
    before_mass = torch.zeros((), device=device)
    after_mass = torch.zeros((), device=device)
    records = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        before_probs = F.softmax(before(inputs), dim=1)[:, absent_classes]
        after_probs = F.softmax(after(inputs), dim=1)[:, absent_classes]
        before_mass += before_probs.sum()
        after_mass += after_probs.sum()
        records += inputs.shape[0] * len(absent_classes)
    ratio = after_mass / before_mass.clamp_min(torch.finfo(before_mass.dtype).tiny)
    return {"probability_mass_retention": float(ratio), "records": float(records)}


def serialized_state_dict_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()
