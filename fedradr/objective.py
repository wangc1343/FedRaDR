from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import nn

Normalization = Literal["exact", "minibatch", "ema"]


@dataclass(frozen=True)
class FedRaDRConfig:
    lambda_max: float = 0.5
    temperature: float = 4.0
    epsilon: float = 1e-8
    normalization: Normalization = "exact"
    ema_decay: float = 0.9

    def __post_init__(self) -> None:
        if self.lambda_max < 0:
            raise ValueError("lambda_max must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.normalization not in {"exact", "minibatch", "ema"}:
            raise ValueError(f"unsupported normalization: {self.normalization}")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


def stage_schedule(round_index: int, total_rounds: int) -> float:
    if total_rounds <= 0 or not 0 <= round_index < total_rounds:
        raise ValueError("round_index must be within total_rounds")
    return 0.5 + 0.5 * math.sin(math.pi * round_index / total_rounds)


def teacher_confidence(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = F.softmax(logits / temperature, dim=1)
    entropy = -(probs.clamp_min(torch.finfo(probs.dtype).tiny).log() * probs).sum(dim=1)
    return (1.0 - entropy / math.log(probs.shape[1])).clamp(0.0, 1.0)


def positive_forgetting_gap(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    local_loss = F.cross_entropy(student_logits, targets, reduction="none")
    global_loss = F.cross_entropy(teacher_logits, targets, reduction="none")
    return (local_loss - global_loss).clamp_min(0.0)


def normalized_gate(gaps: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (gaps / (reference + epsilon)).clamp(0.0, 1.0)


class FedRaDRLocalTrainer:
    """Local FedDC-style training with reliability-aware distillation.

    Reliability weights are explicitly detached before the weighted KL loss.
    The caller supplies the FedDC correction and proximal center as flat tensors.
    """

    def __init__(self, config: FedRaDRConfig) -> None:
        self.config = config
        self._ema_reference: torch.Tensor | None = None

    @staticmethod
    def _flat_parameters(model: nn.Module) -> torch.Tensor:
        return torch.cat([parameter.reshape(-1) for parameter in model.parameters()])

    @torch.no_grad()
    def _exact_reference(
        self,
        model: nn.Module,
        teacher: nn.Module,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        total = torch.zeros((), device=device)
        count = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).reshape(-1).long()
            gaps = positive_forgetting_gap(model(inputs), teacher(inputs), targets)
            total += gaps.sum()
            count += targets.numel()
        model.train(was_training)
        return total / max(count, 1)

    def train(
        self,
        model: nn.Module,
        global_model: nn.Module,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        round_index: int,
        total_rounds: int,
        local_epochs: int,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.0,
        proximal_center: torch.Tensor | None = None,
        gradient_correction: torch.Tensor | None = None,
        max_grad_norm: float | None = None,
    ) -> tuple[nn.Module, list[dict[str, float]]]:
        device = next(model.parameters()).device
        teacher = copy.deepcopy(global_model).to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        if proximal_center is None:
            proximal_center = self._flat_parameters(global_model).detach().to(device)
        if gradient_correction is None:
            gradient_correction = torch.zeros_like(proximal_center)

        metrics: list[dict[str, float]] = []
        schedule = stage_schedule(round_index, total_rounds)

        for local_epoch in range(local_epochs):
            if self.config.normalization == "exact":
                reference = self._exact_reference(model, teacher, loader, device)
            else:
                reference = None

            model.train()
            for batch_index, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device).reshape(-1).long()
                with torch.no_grad():
                    # Batch-position caching is invalid when the training loader shuffles.
                    # Use an indexed dataset for persistent per-sample caches in full runs.
                    teacher_logits = teacher(inputs).detach()

                student_logits = model(inputs)
                gaps = positive_forgetting_gap(student_logits, teacher_logits, targets)
                batch_reference = gaps.detach().mean()
                if self.config.normalization == "minibatch":
                    reference = batch_reference
                elif self.config.normalization == "ema":
                    if self._ema_reference is None:
                        self._ema_reference = batch_reference
                    else:
                        decay = self.config.ema_decay
                        self._ema_reference = decay * self._ema_reference + (1 - decay) * batch_reference
                    reference = self._ema_reference

                confidence = teacher_confidence(teacher_logits, self.config.temperature)
                gate = normalized_gate(gaps, reference, self.config.epsilon)
                weights = (self.config.lambda_max * schedule * confidence * gate).detach()
                teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=1)
                student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
                distillation = (weights * kl).mean() * self.config.temperature**2

                flat = self._flat_parameters(model)
                supervised = F.cross_entropy(student_logits, targets)
                proximal = 0.5 * alpha * (flat - proximal_center).square().sum()
                correction = torch.dot(flat, gradient_correction)
                loss = supervised + proximal + correction + distillation

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                metrics.append(
                    {
                        "local_epoch": float(local_epoch),
                        "batch": float(batch_index),
                        "loss": float(loss.detach()),
                        "supervised_loss": float(supervised.detach()),
                        "distillation_loss": float(distillation.detach()),
                        "mfg": float(gaps.detach().mean()),
                        "hfr": float((gaps.detach() > reference.detach()).float().mean()),
                        "mean_confidence": float(confidence.mean()),
                        "mean_gate": float(gate.detach().mean()),
                    }
                )
        return model, metrics
