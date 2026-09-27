"""Create the concrete file inventory and commands for a FedRaDR rerun.

This utility prepares files; it does not fabricate measurements.  It emits a
run matrix, explicit tuning grids for the rerun campaign, and command files
for partition generation and training.  Commands are executable once the
dataset labels/NPZ files and the torch environment are available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
from typing import Any


SEEDS = (20, 21, 22)
METHODS = (
    "FedAvg", "SCAFFOLD", "FedDyn", "FedVRA", "FedDisco", "FedSpeed",
    "FedKT", "FedSKC", "FedRaDR", "FedRaDR-SAM",
)

DATASETS = {
    "cifar10": {"clients": (100, 500), "partitions": {"D1": "--alpha 0.3", "D2": "--alpha 0.6", "P1": "--n-class-per-client 3", "P2": "--n-class-per-client 5"}},
    "cifar100": {"clients": (100, 500), "partitions": {"D1": "--alpha 0.3", "D2": "--alpha 0.6", "P1": "--n-class-per-client 10", "P2": "--n-class-per-client 20"}},
    "tinyimagenet": {"clients": (100,), "partitions": {"D1": "--alpha 0.1"}},
}


def tuning_protocol() -> dict[str, Any]:
    """Explicit rerun grids, separated from the author's final values."""
    shared = {
        "learning_rate": [0.03, 0.1, 0.3],
        "weight_decay": [0.0001, 0.001, 0.01],
        "local_epochs": [3, 5, 10],
    }
    return {
        "schema_version": "fedradr.rerun-tuning.v1",
        "selection_setting": "CIFAR10-D1-C100-P10",
        "selection_rule": "Select on this setting and freeze the selected values for every test setting and seed.",
        "status": "rerun_protocol",
        "historical": False,
        "shared": shared,
        "methods": {
            "FedAvg": {"method_specific": {}},
            "SCAFFOLD": {"method_specific": {}},
            "FedDyn": {"method_specific": {"alpha": [0.01, 0.1, 1.0]}},
            "FedVRA": {"method_specific": {"control_weight": [0.1, 0.5, 0.9]}},
            "FedDisco": {"method_specific": {"lambda_disco": [0.1, 0.22, 0.5]}},
            "FedSpeed": {"optimizer": "SAM", "method_specific": {"rho": [0.01, 0.05, 0.1], "sam_alpha": [0.05, 0.1, 0.2]}},
            "FedKT": {"method_specific": {"kd_temperature": [2.0, 4.0, 8.0], "kd_lambda": [0.25, 0.5, 0.75]}},
            "FedSKC": {"method_specific": {"struct_coeff": [0.25, 0.65, 1.0], "kd_temperature": [2.0, 4.0, 8.0]}},
            "FedRaDR": {"method_specific": {"lambda_max": [0.1, 0.3, 0.5, 0.7, 1.0], "temperature": [1, 2, 3, 4, 5], "alpha": [0.05, 0.1, 0.2]}},
            "FedRaDR-SAM": {"optimizer": "SAM", "method_specific": {"lambda_max": [0.1, 0.3, 0.5, 0.7, 1.0], "temperature": [1, 2, 3, 4, 5], "alpha": [0.05, 0.1, 0.2], "rho": [0.01, 0.05, 0.1], "sam_alpha": [0.05, 0.1, 0.2]}},
        },
        "author_recorded_selections": {"FedRaDR": {"lambda_max": 0.5, "temperature": 4.0, "alpha": 0.1}, "FedRaDR-SAM": {"lambda_max": 0.5, "temperature": 4.0, "alpha": 0.1, "rho": 0.05, "sam_alpha": 0.1}},
    }


def matrix_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for clients in spec["clients"]:
            for partition, partition_args in spec["partitions"].items():
                participant_counts = (10, 5, 2) if clients == 100 else (10,)
                for participants in participant_counts:
                    rounds = 500 if dataset == "tinyimagenet" else 1000
                    setting = f"C{clients}-P{participants}"
                    for method in METHODS:
                        for seed in SEEDS:
                            stem = f"{dataset}_N{clients}P{participants}_{partition}_{method}_seed{seed}"
                            schedule = root / "schedules" / f"c{clients}-p{participants}-r{rounds}-seed{seed}.json"
                            partition_file = root / "partitions" / f"{dataset}_N{clients}_{partition}_seed{seed}.pkl"
                            output = Path("results") / stem
                            record = {
                                "run_id": stem,
                                "dataset": dataset,
                                "clients": clients,
                                "participants": participants,
                                "setting": setting,
                                "partition": partition,
                                "method": method,
                                "seed": seed,
                                "rounds": rounds,
                                "partition_file": partition_file.as_posix(),
                                "schedule_file": schedule.as_posix(),
                                "output_dir": output.as_posix(),
                                "metrics_jsonl": (output / "metrics.jsonl").as_posix(),
                                "csv_output": (root / "raw_logs" / f"log_{stem}.csv").as_posix(),
                                "partition_args": partition_args,
                            }
                            if method == "FedRaDR":
                                record["command"] = f"python scripts/run_fedradr.py --data data/{dataset}_N{clients}.npz --partition-file {partition_file.as_posix()} --schedule-file {schedule.as_posix()} --output {output.as_posix()} --rounds {rounds} --clients-per-round {participants} --seed {seed}"
                            else:
                                record["command"] = None
                            records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reproducibility"))
    parser.add_argument("--emit-commands", action="store_true")
    args = parser.parse_args()
    root = args.root
    records = matrix_records(root)
    write_json(root / "configs" / "rerun_tuning_protocol.json", tuning_protocol())
    write_json(root / "run_matrix.json", {"schema_version": "fedradr.run-matrix.v1", "historical": False, "records": records})
    csv_path = root / "run_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["run_id", "dataset", "clients", "participants", "partition", "method", "seed", "rounds", "partition_file", "schedule_file", "output_dir", "metrics_jsonl", "csv_output", "command"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields} for record in records)
    if args.emit_commands:
        command_dir = root / "commands"
        command_dir.mkdir(parents=True, exist_ok=True)
        partition_lines = ["$ErrorActionPreference = 'Stop'", "# Requires labels/<dataset>_train_labels.npy", "python scripts/build_reproducibility_bundle.py generate-partitions --root reproducibility --labels-root labels"]
        train_lines = ["$ErrorActionPreference = 'Stop'", "# FedRaDR commands only; baseline adapters remain method-specific."]
        for record in records:
            if record["command"]:
                train_lines.append(record["command"])
                train_lines.append(f"python scripts/export_run_csv.py --input {record['metrics_jsonl']} --output {record['csv_output']} --config {record['output_dir']}/config.json")
        (command_dir / "generate_partitions.ps1").write_text("\n".join(partition_lines) + "\n", encoding="utf-8")
        (command_dir / "run_fedradr.ps1").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(records), "fedradr_commands": sum(record["command"] is not None for record in records), "root": root.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
