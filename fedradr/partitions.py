from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


ALGORITHM_VERSION = "fedradr_partition_v1_classwise"


def labels_digest(labels: Sequence[int] | np.ndarray) -> str:
    """Hash the exact label array used as the partition index basis."""
    array = np.asarray(labels, dtype=np.int64)
    descriptor = f"dtype={array.dtype};shape={array.shape};".encode("ascii")
    return hashlib.sha256(descriptor + array.tobytes(order="C")).hexdigest()


def dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Create a deterministic class-wise Dirichlet partition."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    generator = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == class_id)
        generator.shuffle(class_indices)
        proportions = generator.dirichlet(np.full(num_clients, alpha))
        boundaries = np.cumsum(proportions)[:-1]
        splits = np.split(class_indices, (boundaries * len(class_indices)).astype(int))
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(int(index) for index in split)

    for values in client_indices:
        generator.shuffle(values)
    return client_indices


def pathological_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> list[list[int]]:
    """Create a deterministic label-limited partition with balanced class coverage."""
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    classes = np.unique(labels_array)
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0 < classes_per_client <= len(classes):
        raise ValueError("classes_per_client must be within the number of classes")

    generator = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    slots = np.resize(np.tile(classes, classes_per_client), num_clients * classes_per_client)
    generator.shuffle(slots)
    for client_id in range(num_clients):
        start = client_id * classes_per_client
        chosen = list(dict.fromkeys(int(value) for value in slots[start : start + classes_per_client]))
        while len(chosen) < classes_per_client:
            candidate = int(generator.choice(classes))
            if candidate not in chosen:
                chosen.append(candidate)
        assignments[client_id] = chosen

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in classes:
        class_indices = np.flatnonzero(labels_array == class_id)
        generator.shuffle(class_indices)
        recipients = [client_id for client_id, values in enumerate(assignments) if int(class_id) in values]
        if not recipients:
            recipients = [int(generator.integers(num_clients))]
        for client_id, split in zip(recipients, np.array_split(class_indices, len(recipients))):
            client_indices[client_id].extend(int(index) for index in split)

    for values in client_indices:
        generator.shuffle(values)
    return client_indices


def build_partition_payload(
    labels: Sequence[int] | np.ndarray,
    dataset: str,
    partition_type: str,
    num_clients: int,
    seed: int,
    *,
    alpha: float | None = None,
    classes_per_client: int | None = None,
) -> dict[str, object]:
    partition = partition_type.upper()
    if partition in {"D1", "D2"}:
        if alpha is None:
            raise ValueError("alpha is required for a Dirichlet partition")
        indices = dirichlet_partition(labels, num_clients, alpha, seed)
        hyper: dict[str, float | int] = {"alpha": alpha}
    elif partition in {"P1", "P2"}:
        if classes_per_client is None:
            raise ValueError("classes_per_client is required for a pathological partition")
        indices = pathological_partition(labels, num_clients, classes_per_client, seed)
        hyper = {"n_class_per_client": classes_per_client}
    else:
        raise ValueError(f"unsupported partition type: {partition_type}")

    return {
        "client_indices": indices,
        "n_client": num_clients,
        "dataset": dataset.lower(),
        "partition_type": partition,
        "hyper": hyper,
        "seed": seed,
        "index_basis": "input_labels_array_order",
        "sample_count": int(len(labels)),
        "labels_sha256": labels_digest(labels),
        "algorithm_version": ALGORITHM_VERSION,
        "historical": False,
    }


def write_partition(payload: dict[str, object], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    output.write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest()
