from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
import subprocess
from collections.abc import Sequence

import numpy as np
try:
    import torch
except ImportError:  # Schedule and metadata tools do not require PyTorch.
    torch = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_client_schedule(num_clients: int, clients_per_round: int, rounds: int, seed: int) -> list[list[int]]:
    if not 0 < clients_per_round <= num_clients:
        raise ValueError("clients_per_round must be within num_clients")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    generator = np.random.default_rng(seed)
    return [sorted(generator.choice(num_clients, clients_per_round, replace=False).tolist()) for _ in range(rounds)]


def make_nested_client_schedules(
    num_clients: int,
    clients_per_round: Sequence[int],
    rounds: int,
    seed: int,
) -> dict[int, list[list[int]]]:
    """Build participation schedules from one shared permutation per round.

    For every round, all participation settings use prefixes of the same
    shuffled client order. This isolates the participation-count effect while
    keeping the seed, client pool, and round-level sampling randomness fixed.
    """
    counts = sorted(set(int(count) for count in clients_per_round))
    if not counts or any(not 0 < count <= num_clients for count in counts):
        raise ValueError("clients_per_round values must be within num_clients")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    generator = np.random.default_rng(seed)
    schedules = {count: [] for count in counts}
    for _ in range(rounds):
        permutation = generator.permutation(num_clients)
        for count in counts:
            schedules[count].append(sorted(permutation[:count].tolist()))
    return schedules


def make_nested_client_schedule_family(
    num_clients: int,
    clients_per_round: Sequence[int],
    rounds: int,
    seed: int,
) -> tuple[dict[int, list[list[int]]], list[str]]:
    """Return nested schedules and a hash for each shared round permutation.

    The ordinary helper preserves its historical return type. This companion
    API makes the controlled-participation claim auditable without storing a
    second copy of every full permutation in each schedule file.
    """
    counts = sorted(set(int(count) for count in clients_per_round))
    if not counts or any(not 0 < count <= num_clients for count in counts):
        raise ValueError("clients_per_round values must be within num_clients")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    generator = np.random.default_rng(seed)
    schedules = {count: [] for count in counts}
    permutation_hashes: list[str] = []
    for _ in range(rounds):
        permutation = generator.permutation(num_clients).tolist()
        encoded = json.dumps(permutation, separators=(",", ":")).encode("utf-8")
        permutation_hashes.append(hashlib.sha256(encoded).hexdigest())
        for count in counts:
            # Keep the historical file representation canonical and stable.
            schedules[count].append(sorted(int(value) for value in permutation[:count]))
    return schedules, permutation_hashes


def save_json_with_hash(value: object, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_metadata(root: str | Path) -> dict[str, object]:
    root = Path(root)

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "tags_at_commit": run("tag", "--points-at", "HEAD").splitlines(),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tags_at_commit": [], "dirty": None}
