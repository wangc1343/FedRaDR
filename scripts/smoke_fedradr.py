"""Dependency-light end-to-end FedRaDR smoke run on deterministic toy data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.objective import FedRaDRConfig
from utils_methods_FedRaDR import train_FedRaDR


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    client_x, client_y = [], []
    for client_id in range(4):
        x = rng.normal(loc=(client_id - 1.5) * 0.15, size=(24, 4)).astype(np.float32)
        y = ((x[:, 0] + 0.7 * x[:, 1] - 0.2 * x[:, 2]) > 0).astype(np.int64)
        client_x.append(x)
        client_y.append(y)
    return SimpleNamespace(
        n_client=4,
        dataset="synt",
        clnt_x=client_x,
        clnt_y=client_y,
        tst_x=np.concatenate(client_x),
        tst_y=np.concatenate(client_y),
    )


def dataset_factory(x, y, train, dataset_name):
    del train, dataset_name
    return TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(y).long())


def evaluator(x, y, model, dataset_name):
    del dataset_name
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, device=next(model.parameters()).device).float())
        targets = torch.as_tensor(y, device=logits.device).long()
        loss = nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return float(loss), float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("smoke_output"))
    args = parser.parse_args()
    torch.manual_seed(11)
    initial_model = TinyClassifier()
    data_obj = make_data()
    schedule = [[0, 1], [2, 3], [0, 2]]
    model = train_FedRaDR(
        data_obj,
        TinyClassifier,
        initial_model,
        client_schedule=schedule,
        output_dir=args.output,
        config=FedRaDRConfig(normalization="exact"),
        learning_rate=0.05,
        batch_size=8,
        local_epochs=1,
        alpha=0.01,
        weight_decay=0.0,
        seed=20,
        dataset_factory=dataset_factory,
        evaluator=evaluator,
        device="cpu",
    )
    loss, accuracy = evaluator(data_obj.tst_x, data_obj.tst_y, model, "synt")
    summary = {"status": "ok", "rounds": len(schedule), "loss": loss, "accuracy": accuracy}
    (args.output / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
