"""Run FedRaDR from a serialized FedDC-style data object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.models import make_model
from fedradr.objective import FedRaDRConfig
from fedradr.reproducibility import make_client_schedule
from utils_methods_FedRaDR import train_FedRaDR


def load_data(path: Path):
    """Load clnt_x/clnt_y/tst_x/tst_y/dataset from a compressed NPZ file."""
    archive = np.load(path, allow_pickle=True)
    dataset = str(archive["dataset"].item())
    client_x = list(archive["clnt_x"])
    client_y = list(archive["clnt_y"])
    return SimpleNamespace(
        dataset=dataset,
        n_client=len(client_x),
        clnt_x=client_x,
        clnt_y=client_y,
        tst_x=archive["tst_x"],
        tst_y=archive["tst_y"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenetv2"], default="resnet18")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate-decay", type=float, default=0.998)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--normalization", choices=["exact", "minibatch", "ema"], default="exact")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()

    data_obj = load_data(args.data)
    num_classes = 100 if data_obj.dataset == "CIFAR100" else 10
    model_func = lambda: make_model(args.model, num_classes)
    torch.manual_seed(37)
    initial_model = model_func()
    schedule = make_client_schedule(
        data_obj.n_client, args.clients_per_round, args.rounds, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "client_schedule.json").write_text(
        json.dumps(schedule, indent=2) + "\n", encoding="utf-8"
    )
    train_FedRaDR(
        data_obj,
        model_func,
        initial_model,
        client_schedule=schedule,
        output_dir=args.output,
        config=FedRaDRConfig(
            lambda_max=args.lambda_max,
            temperature=args.temperature,
            normalization=args.normalization,
        ),
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
