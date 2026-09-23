"""FedRaDR training entry point compatible with the original FedDC data objects."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedradr.diagnostics import serialized_state_dict_bytes
from fedradr.logging import JsonlLogger
from fedradr.objective import FedRaDRConfig, FedRaDRLocalTrainer
from fedradr.reproducibility import seed_everything
from fedradr.resource import measure_resources
from utils_dataset import Dataset
from utils_general import get_acc_loss


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


def train_FedRaDR(
    data_obj,
    model_func: Callable[[], torch.nn.Module],
    init_model: torch.nn.Module,
    *,
    client_schedule: list[list[int]],
    output_dir: str | Path,
    config: FedRaDRConfig = FedRaDRConfig(),
    learning_rate: float = 0.1,
    batch_size: int = 50,
    local_epochs: int = 5,
    alpha: float = 0.1,
    weight_decay: float = 1e-3,
    seed: int = 20,
    aggregation: str = "data_size",
    max_grad_norm: float | None = None,
) -> torch.nn.Module:
    """Train FedRaDR using fixed client selections and auditable raw logs.

    `aggregation="data_size"` matches the global objective in the manuscript.
    Use `uniform` only for a separately named ablation.
    """
    if aggregation not in {"data_size", "uniform"}:
        raise ValueError("aggregation must be data_size or uniform")
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "local_epochs": local_epochs,
                "alpha": alpha,
                "weight_decay": weight_decay,
                "aggregation": aggregation,
                "fedradr": config.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = copy.deepcopy(init_model).to(device)
    history = [torch.zeros_like(_flat_parameters(global_model)) for _ in range(data_obj.n_client)]
    trainer = FedRaDRLocalTrainer(config)

    for round_index, selected_clients in enumerate(client_schedule):
        if not selected_clients:
            raise ValueError(f"round {round_index} has no selected clients")
        global_flat = _flat_parameters(global_model)
        local_parameters = []
        local_sizes = []
        round_metrics = []

        with measure_resources(device) as resource_result:
            for client_id in selected_clients:
                client_model = model_func().to(device)
                client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
                client_x = data_obj.clnt_x[client_id]
                client_y = data_obj.clnt_y[client_id]
                loader = DataLoader(
                    Dataset(client_x, client_y, train=True, dataset_name=data_obj.dataset),
                    batch_size=batch_size,
                    shuffle=True,
                )
                optimizer = torch.optim.SGD(
                    client_model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                client_model, metrics = trainer.train(
                    client_model,
                    global_model,
                    loader,
                    round_index=round_index,
                    total_rounds=len(client_schedule),
                    local_epochs=local_epochs,
                    optimizer=optimizer,
                    alpha=alpha,
                    proximal_center=global_flat,
                    gradient_correction=-history[client_id],
                    max_grad_norm=max_grad_norm,
                )
                local_flat = _flat_parameters(client_model)
                history[client_id] += alpha * (local_flat - global_flat)
                local_parameters.append(local_flat)
                local_sizes.append(len(client_y))
                round_metrics.extend(metrics)

            stacked = torch.stack(local_parameters)
            if aggregation == "data_size":
                weights = torch.tensor(local_sizes, dtype=stacked.dtype, device=device)
                weights /= weights.sum()
            else:
                weights = torch.full(
                    (len(selected_clients),), 1.0 / len(selected_clients), device=device
                )
            _load_flat_parameters(global_model, (stacked * weights[:, None]).sum(dim=0))

        test_loss, test_accuracy = get_acc_loss(
            data_obj.tst_x, data_obj.tst_y, global_model, data_obj.dataset, 0
        )
        measurement = resource_result["measurement"]
        logger.log(
            {
                "round": round_index,
                "selected_clients": selected_clients,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "mfg": float(np.mean([row["mfg"] for row in round_metrics])),
                "hfr": float(np.mean([row["hfr"] for row in round_metrics])),
                "elapsed_seconds": measurement.elapsed_seconds,
                "peak_cuda_bytes": measurement.peak_cuda_bytes,
                "energy_joules": measurement.energy_joules,
                "downlink_bytes_per_client": serialized_state_dict_bytes(global_model),
                "uplink_bytes_per_client": serialized_state_dict_bytes(global_model),
            }
        )
        torch.save(global_model.state_dict(), output_dir / "latest.pt")

    return global_model
