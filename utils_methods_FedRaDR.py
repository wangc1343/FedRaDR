"""FedRaDR training entry point built on the official FedDC state update."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedradr.diagnostics import serialized_state_dict_bytes
from fedradr.logging import JsonlLogger
from fedradr.objective import FedRaDRConfig, FedRaDRLocalTrainer
from fedradr.reproducibility import seed_everything
from fedradr.resource import measure_resources


def _flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


@torch.no_grad()
def _load_flat_parameters(model: torch.nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.copy_(flat[offset : offset + count].view_as(parameter))
        offset += count
    if offset != flat.numel():
        raise ValueError("flat parameter vector does not match model")


class _ArrayDataset(Dataset):
    def __init__(self, data_x, data_y, train: bool, dataset_name: str) -> None:
        self.inputs = torch.as_tensor(np.asarray(data_x))
        self.targets = torch.as_tensor(np.asarray(data_y)).reshape(-1).long()
        self.train = train
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        inputs = self.inputs[index]
        if self.dataset_name in {"CIFAR10", "CIFAR100"} and self.train:
            if torch.rand(()) > 0.5:
                inputs = inputs.flip(-1)
            padded = torch.nn.functional.pad(inputs, (4, 4, 4, 4))
            top, left = torch.randint(0, 9, (2,)).tolist()
            inputs = padded[:, top : top + 32, left : left + 32]
        if self.dataset_name == "shakespeare":
            inputs = inputs.long()
        else:
            inputs = inputs.float()
        return inputs, self.targets[index]


def _default_dataset_factory(data_x, data_y, train: bool, dataset_name: str):
    return _ArrayDataset(data_x, data_y, train, dataset_name)


def _default_evaluator(data_x, data_y, model, dataset_name: str):
    device = next(model.parameters()).device
    loader = DataLoader(_ArrayDataset(data_x, data_y, False, dataset_name), batch_size=2000)
    total_loss = 0.0
    total_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
            total_correct += int((logits.argmax(1) == targets).sum())
            total += targets.numel()
    return total_loss / total, total_correct / total


def _schedule_hash(client_schedule: list[list[int]]) -> str:
    payload = json.dumps(client_schedule, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_weighted_mean(rows: list[dict[str, float]], key: str) -> float:
    count = sum(row["batch_size"] for row in rows)
    return sum(row[key] * row["batch_size"] for row in rows) / count


def _sample_weighted_mean_optional(rows: list[dict[str, float]], key: str) -> float | None:
    matching = [row for row in rows if key in row]
    return _sample_weighted_mean(matching, key) if matching else None


class _MaterializedIndexedDataset(Dataset):
    """Freeze one round's augmentation and expose stable cache indices."""

    def __init__(self, source: Dataset) -> None:
        self.records = [source[index] for index in range(len(source))]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        inputs, targets = self.records[index][:2]
        return inputs, targets, index


def train_FedRaDR(
    data_obj,
    model_func: Callable[[], torch.nn.Module],
    init_model: torch.nn.Module,
    *,
    client_schedule: list[list[int]],
    output_dir: str | Path,
    config: FedRaDRConfig = FedRaDRConfig(),
    learning_rate: float = 0.1,
    learning_rate_decay: float = 1.0,
    batch_size: int = 50,
    local_epochs: int = 5,
    alpha: float = 0.1,
    weight_decay: float = 1e-3,
    seed: int = 20,
    max_grad_norm: float | None = 10.0,
    dataset_factory: Callable | None = None,
    evaluator: Callable | None = None,
    device: str | torch.device | None = None,
    checkpoint_every: int = 0,
) -> torch.nn.Module:
    """Run FedRaDR with the cloud/state equations from official FedDC.

    Custom dataset/evaluator adapters make the integration test independent of
    the legacy FedDC dependency stack. Production runs use the original adapters.
    """
    if not client_schedule:
        raise ValueError("client_schedule must contain at least one round")
    if learning_rate <= 0 or batch_size <= 0 or local_epochs <= 0 or alpha <= 0:
        raise ValueError("learning rate, batch size, local epochs, and alpha must be positive")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")

    seed_everything(seed)
    dataset_factory = dataset_factory or _default_dataset_factory
    evaluator = evaluator or _default_evaluator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to append to existing run log: {metrics_path}")
    logger = JsonlLogger(metrics_path)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    n_clients = int(data_obj.n_client)
    client_sizes = np.asarray([len(y) for y in data_obj.clnt_y], dtype=np.float64)
    fed_dc_weights = client_sizes / client_sizes.sum() * n_clients
    for round_clients in client_schedule:
        if not round_clients or len(set(round_clients)) != len(round_clients):
            raise ValueError("each round must contain distinct selected clients")
        if min(round_clients) < 0 or max(round_clients) >= n_clients:
            raise ValueError("client schedule contains an invalid client id")

    global_model = copy.deepcopy(init_model).to(run_device)
    cloud_params = _flat_parameters(global_model)
    n_params = cloud_params.numel()
    parameter_drifts = torch.zeros((n_clients, n_params), device="cpu")
    gradient_states = torch.zeros((n_clients + 1, n_params), device="cpu")
    mean_parameter_drift = torch.zeros(n_params, device=run_device)
    cumulative_serialized_bytes = 0

    run_config = {
        "seed": seed,
        "learning_rate": learning_rate,
        "learning_rate_decay": learning_rate_decay,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "alpha": alpha,
        "weight_decay": weight_decay,
        "device": str(run_device),
        "checkpoint_every": checkpoint_every,
        "client_schedule_sha256": _schedule_hash(client_schedule),
        "fedradr": config.__dict__,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for round_index, selected_clients in enumerate(client_schedule):
        current_lr = learning_rate * learning_rate_decay**round_index
        delta_gradient_sum = torch.zeros_like(cloud_params)
        selected_parameters = []
        round_metrics: list[dict[str, float]] = []

        with measure_resources(run_device) as resource_result:
            for client_id in selected_clients:
                client_model = model_func().to(run_device)
                client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
                train_dataset = _MaterializedIndexedDataset(
                    dataset_factory(
                        data_obj.clnt_x[client_id], data_obj.clnt_y[client_id], True, data_obj.dataset
                    )
                )
                generator = torch.Generator().manual_seed(seed * 100000 + round_index * 1000 + client_id)
                loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
                reference_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

                client_weight = float(fed_dc_weights[client_id])
                local_update_last = gradient_states[client_id].to(run_device)
                global_update_last = gradient_states[-1].to(run_device) / client_weight
                client_alpha = alpha / client_weight
                proximal_center = cloud_params - parameter_drifts[client_id].to(run_device)
                correction = -local_update_last + global_update_last
                optimizer = torch.optim.SGD(
                    client_model.parameters(), lr=current_lr, weight_decay=weight_decay
                )
                trainer = FedRaDRLocalTrainer(config)
                client_model, metrics = trainer.train(
                    client_model,
                    global_model,
                    loader,
                    reference_loader=reference_loader,
                    round_index=round_index,
                    total_rounds=len(client_schedule),
                    local_epochs=local_epochs,
                    optimizer=optimizer,
                    alpha=client_alpha,
                    proximal_center=proximal_center,
                    gradient_correction=correction,
                    max_grad_norm=max_grad_norm,
                )

                current_params = _flat_parameters(client_model)
                delta_params = current_params - cloud_params
                parameter_drifts[client_id] += delta_params.detach().cpu()
                mean_parameter_drift += delta_params / n_clients
                n_minibatches = local_epochs * math.ceil(client_sizes[client_id] / batch_size)
                beta = 1.0 / (n_minibatches * current_lr)
                state_g = local_update_last - global_update_last - beta * delta_params
                delta_gradient_sum += (state_g - gradient_states[client_id].to(run_device)) * client_weight
                gradient_states[client_id] = state_g.detach().cpu()
                selected_parameters.append(current_params)
                round_metrics.extend(metrics)

            selected_stack = torch.stack(selected_parameters)
            selected_weights = torch.tensor(
                [client_sizes[client_id] for client_id in selected_clients],
                dtype=selected_stack.dtype,
                device=run_device,
            )
            selected_weights = selected_weights / selected_weights.sum()
            selected_average = (selected_stack * selected_weights[:, None]).sum(dim=0)
            gradient_states[-1] += (delta_gradient_sum / n_clients).detach().cpu()
            cloud_params = selected_average + mean_parameter_drift
            _load_flat_parameters(global_model, cloud_params)

        test_loss, test_accuracy = evaluator(
            data_obj.tst_x, data_obj.tst_y, global_model, data_obj.dataset
        )
        measurement = resource_result["measurement"]
        payload_per_client = serialized_state_dict_bytes(global_model)
        downlink_bytes = payload_per_client * len(selected_clients)
        uplink_bytes = payload_per_client * len(selected_clients)
        cumulative_serialized_bytes += downlink_bytes + uplink_bytes
        logger.log(
            {
                "round": round_index,
                "selected_clients": selected_clients,
                "learning_rate": current_lr,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "train_objective": _sample_weighted_mean(round_metrics, "loss"),
                "mfg": _sample_weighted_mean(round_metrics, "mfg"),
                "hfr_tau": _sample_weighted_mean_optional(round_metrics, "hfr_tau"),
                "hfr_reference": _sample_weighted_mean(round_metrics, "hfr_reference"),
                "single_device_elapsed_seconds": measurement.elapsed_seconds,
                "peak_cuda_bytes": measurement.peak_cuda_bytes,
                "energy_joules": measurement.energy_joules,
                "downlink_bytes_per_client": payload_per_client,
                "uplink_bytes_per_client": payload_per_client,
                "downlink_bytes": downlink_bytes,
                "uplink_bytes": uplink_bytes,
                "cumulative_serialized_bytes": cumulative_serialized_bytes,
            }
        )
        torch.save(
            {
                "round": round_index,
                "model": global_model.state_dict(),
                "config": run_config,
            },
            output_dir / "latest.pt",
        )
        if checkpoint_every > 0 and (round_index + 1) % checkpoint_every == 0:
            torch.save(
                {
                    "round": round_index,
                    "model": global_model.state_dict(),
                    "cloud_params": cloud_params.cpu(),
                    "parameter_drifts": parameter_drifts,
                    "gradient_states": gradient_states,
                    "mean_parameter_drift": mean_parameter_drift.cpu(),
                    "config": run_config,
                },
                output_dir / f"state_round_{round_index + 1}.pt",
            )

    return global_model
