from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle


SEEDS = (20, 21, 22)
SPECS = {
    "cifar10": {
        "clients": (100, 500),
        "partitions": {"D1": {"alpha": 0.3}, "D2": {"alpha": 0.6}, "P1": {"n_class_per_client": 3}, "P2": {"n_class_per_client": 5}},
    },
    "cifar100": {
        "clients": (100, 500),
        "partitions": {"D1": {"alpha": 0.3}, "D2": {"alpha": 0.6}, "P1": {"n_class_per_client": 10}, "P2": {"n_class_per_client": 20}},
    },
    "tinyimagenet": {
        "clients": (100,),
        "partitions": {"D1": {"alpha": 0.1}},
    },
}
REQUIRED_KEYS = {
    "client_indices",
    "n_client",
    "dataset",
    "partition_type",
    "hyper",
    "seed",
    "index_basis",
}


def expected_files() -> list[tuple[str, int, str, int, dict[str, float | int]]]:
    rows = []
    for dataset, spec in SPECS.items():
        for clients in spec["clients"]:
            for partition, hyper in spec["partitions"].items():
                for seed in SEEDS:
                    rows.append((dataset, clients, partition, seed, hyper))
    return rows


def validate_payload(
    payload: object,
    clients: int,
    seed: int,
    expected_hyper: dict[str, float | int],
    *,
    expected_dataset: str | None = None,
    expected_partition: str | None = None,
    expected_sample_count: int | None = None,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["root object is not a dictionary"]
    errors = [f"missing key: {key}" for key in sorted(REQUIRED_KEYS - payload.keys())]
    if errors:
        return errors
    if payload["n_client"] != clients:
        errors.append(f"n_client={payload['n_client']} expected {clients}")
    if payload["seed"] != seed:
        errors.append(f"seed={payload['seed']} expected {seed}")
    if expected_dataset is not None and str(payload.get("dataset", "")).lower() != expected_dataset.lower():
        errors.append(f"dataset={payload.get('dataset')!r} expected {expected_dataset!r}")
    if expected_partition is not None and str(payload.get("partition_type", "")).upper() != expected_partition.upper():
        errors.append(f"partition_type={payload.get('partition_type')!r} expected {expected_partition!r}")
    if payload.get("index_basis") != "input_labels_array_order":
        errors.append("index_basis must be input_labels_array_order")
    indices = payload["client_indices"]
    if not isinstance(indices, list) or len(indices) != clients:
        errors.append("client_indices must be a list with one entry per client")
    else:
        flattened = []
        for client_id, values in enumerate(indices):
            if not isinstance(values, list) or any(not isinstance(value, int) or value < 0 for value in values):
                errors.append(f"client {client_id} indices must be nonnegative integers")
                continue
            flattened.extend(values)
        if len(flattened) != len(set(flattened)):
            errors.append("sample indices overlap across clients")
        sample_count = payload.get("sample_count", expected_sample_count)
        if sample_count is not None:
            if not isinstance(sample_count, int) or sample_count < 0:
                errors.append("sample_count must be a nonnegative integer")
            else:
                if expected_sample_count is not None and sample_count != expected_sample_count:
                    errors.append(f"sample_count={sample_count} expected {expected_sample_count}")
                if sorted(flattened) != list(range(sample_count)):
                    errors.append("client indices must cover every sample exactly once")
    hyper = payload["hyper"]
    if not isinstance(hyper, dict):
        errors.append("hyper must be a dictionary")
    else:
        for key, expected in expected_hyper.items():
            if hyper.get(key) != expected:
                errors.append(f"hyper.{key}={hyper.get(key)!r} expected {expected!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the 51 serialized paper partition files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    failed = False
    for dataset, clients, partition, seed, hyper in expected_files():
        filename = f"{dataset}_N{clients}_{partition}_seed{seed}.pkl"
        path = args.root / filename
        record = {"file": filename, "exists": path.is_file(), "valid": None}
        if not path.is_file():
            record["errors"] = ["missing file"]
            failed = True
        else:
            payload_bytes = path.read_bytes()
            try:
                payload = pickle.loads(payload_bytes)
                errors = validate_payload(
                    payload,
                    clients,
                    seed,
                    hyper,
                    expected_dataset=dataset,
                    expected_partition=partition,
                )
            except Exception as error:
                errors = [f"pickle load failed: {error}"]
            record.update({
                "valid": not errors,
                "errors": errors,
                "bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            })
            failed = failed or bool(errors)
        records.append(record)

    summary = {"expected": len(records), "valid": sum(row.get("valid") is True for row in records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"validated {summary['valid']}/{summary['expected']} partition files")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
