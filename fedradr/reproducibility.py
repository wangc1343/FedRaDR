from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_client_schedule(num_clients: int, clients_per_round: int, rounds: int, seed: int) -> list[list[int]]:
    if not 0 < clients_per_round <= num_clients:
        raise ValueError("clients_per_round must be within num_clients")
    generator = np.random.default_rng(seed)
    return [sorted(generator.choice(num_clients, clients_per_round, replace=False).tolist()) for _ in range(rounds)]


def save_json_with_hash(value: object, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
