"""Generate deterministic generated demonstration artifacts.

All outputs are explicitly marked ``demo``.  They are useful for
testing plotting, indexing, review-package layout, and trend analysis only.
They must never be reported as measured or historical experimental results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import pickle

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.partitions import build_partition_payload, write_partition


SEEDS = (20, 21, 22)
PARTITIONS = {
    "cifar10": (10, (100, 500), {"D1": ("D", 0.3), "D2": ("D", 0.6), "P1": ("P", 3), "P2": ("P", 5)}),
    "cifar100": (100, (100, 500), {"D1": ("D", 0.3), "D2": ("D", 0.6), "P1": ("P", 10), "P2": ("P", 20)}),
    "tinyimagenet": (200, (100,), {"D1": ("D", 0.1)}),
}
FINAL_ACCURACY = {
    ("C100-P10", "FedAvg"): 0.7859, ("C100-P10", "FedDyn"): 0.8069, ("C100-P10", "FedSKC"): 0.8324, ("C100-P10", "FedRaDR"): 0.8403,
    ("C100-P5", "FedAvg"): 0.7614, ("C100-P5", "FedDyn"): 0.7822, ("C100-P5", "FedSKC"): 0.8047, ("C100-P5", "FedRaDR"): 0.8186,
    ("C100-P2", "FedAvg"): 0.7224, ("C100-P2", "FedDyn"): 0.7461, ("C100-P2", "FedSKC"): 0.7683, ("C100-P2", "FedRaDR"): 0.7844,
}
TARGET_ROUND = {"FedAvg": 1100, "FedDyn": 215, "FedSKC": 194, "FedRaDR": 187}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generated_partitions(root: Path) -> list[dict[str, object]]:
    output = root / "partitions"
    records = []
    for dataset, (classes, clients_set, specs) in PARTITIONS.items():
        sample_count = 2000 if dataset == "tinyimagenet" else 1000
        labels = np.arange(sample_count, dtype=np.int64) % classes
        for clients in clients_set:
            for name, (kind, value) in specs.items():
                for seed in SEEDS:
                    kwargs = {"alpha": float(value)} if kind == "D" else {"classes_per_client": int(value)}
                    payload = build_partition_payload(labels, dataset, name, clients, seed, **kwargs)
                    payload["provenance"] = {"origin": "demo", "historical": False, "label_source": "arithmetic demo labels"}
                    path = output / f"{dataset}_N{clients}_{name}_seed{seed}.pkl"
                    write_partition(payload, path)
                    records.append({"path": path.as_posix(), "sha256": sha256(path), "sample_count": sample_count, "dataset": dataset, "historical": False})
    (output / "generated_manifest.json").write_text(json.dumps({"schema_version": "fedradr.generated-partitions.v1", "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def final_accuracy(setting: str, method: str, seed: int) -> float:
    base = FINAL_ACCURACY.get((setting, method), 0.78 if method == "FedAvg" else 0.82)
    return float(np.clip(base + (seed - 21) * 0.0025, 0.0, 0.99))


def write_logs(root: Path) -> list[dict[str, object]]:
    results = root.parent / "results" / "demo"
    raw = root / "raw_logs" / "demo"
    records = []
    for setting, clients, participants in (("C100-P10", 100, 10), ("C100-P5", 100, 5), ("C100-P2", 100, 2)):
        for method in ("FedAvg", "FedDyn", "FedSKC", "FedRaDR"):
            for seed in SEEDS:
                rounds = 1000
                target = TARGET_ROUND[method]
                final = final_accuracy(setting, method, seed)
                rng = np.random.default_rng(seed * 100 + participants)
                run_dir = results / f"cifar10_{setting}_{method}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics = run_dir / "metrics.jsonl"
                cumulative = 0
                with metrics.open("w", encoding="utf-8") as stream:
                    for round_id in range(rounds):
                        progress = min(1.0, (round_id + 1) / max(target, 1))
                        accuracy = 0.50 + (final - 0.50) * (1.0 - np.exp(-3.2 * progress)) / (1.0 - np.exp(-3.2))
                        accuracy += float(rng.normal(0.0, 0.0015))
                        accuracy = float(np.clip(accuracy, 0.0, 0.99))
                        downlink = 1800000
                        uplink = participants * 1800000
                        cumulative += downlink + uplink
                        record = {
                            "round": round_id,
                            "test_accuracy": accuracy,
                            "train_objective": float(max(0.05, 1.1 - accuracy + rng.normal(0, 0.003))),
                            "mfg": float(max(0.0, (0.12 if method == "FedAvg" else 0.07 if method == "FedDyn" else 0.045 if method == "FedSKC" else 0.03) * np.exp(-round_id / 250) + rng.normal(0, 0.001))),
                            "hfr_tau": float(max(0.0, (0.35 if method == "FedAvg" else 0.22) * np.exp(-round_id / 300))),
                            "downlink_bytes": downlink,
                            "uplink_bytes": uplink,
                            "cumulative_serialized_bytes": cumulative,
                            "single_device_elapsed_seconds": float((round_id + 1) * (0.55 + participants * 0.01)),
                        }
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                config = {"provenance": {"origin": "demo", "historical": False}, "dataset": "cifar10", "setting": setting, "method": method, "seed": seed, "generated_target_round": target, "generated_final_accuracy": final}
                (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                destination = raw / f"log_cifar10_{setting}_{method}_seed{seed}.metrics.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(metrics.read_bytes())
                records.append({"metrics": metrics.as_posix(), "sha256": sha256(metrics), "method": method, "setting": setting, "seed": seed, "historical": False})
    (raw / "manifest.json").write_text(json.dumps({"schema_version": "fedradr.generated-logs.v1", "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def write_energy(root: Path) -> list[dict[str, object]]:
    output = root / "power_traces" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for method, mean_power in (("FedAvg", 29.5), ("FedRaDR-exact", 53.0), ("FedRaDR-mini-batch", 36.5), ("FedRaDR-EMA", 32.0)):
        for seed in SEEDS:
            path = output / f"power_jetson_orin_tinyimagenet_{method}_seed{seed}.samples.csv"
            rng = np.random.default_rng(seed + (0 if method == "FedAvg" else 100))
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["timestamp_ms", "power_watts", "power_scope", "gpu_util_percent", "memory_util_percent", "memory_used_mib", "temperature_c"])
                for index in range(600):
                    writer.writerow([index * 100, round(float(mean_power + 2.0 * np.sin(index / 30) + rng.normal(0, 0.4)), 4), "board_total", round(60 + 10 * np.sin(index / 40), 2), round(35 + 5 * np.sin(index / 50), 2), round(2100 + 30 * np.sin(index / 60), 2), round(55 + 2 * np.sin(index / 50), 2)])
            summary = {"provenance": {"origin": "demo", "historical": False}, "device": "Jetson AGX Orin (generated)", "method": method, "seed": seed, "sampling_interval_ms": 100, "power_scope": "board_total", "elapsed_seconds": 60.0, "energy_joules": mean_power * 60.0, "raw_trace": path.name}
            (path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            records.append({"trace": path.as_posix(), "summary": path.with_suffix(".summary.json").as_posix(), "sha256": sha256(path), "historical": False})
    (output / "manifest.json").write_text(json.dumps({"schema_version": "fedradr.generated-energy.v1", "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reproducibility"))
    args = parser.parse_args()
    partitions = generated_partitions(args.root)
    logs = write_logs(args.root)
    energy = write_energy(args.root)
    manifest = {"schema_version": "fedradr.generated-demo.v1", "provenance": {"origin": "demo", "historical": False, "warning": "Generated trend data for pipeline demonstrations only."}, "counts": {"partitions": len(partitions), "logs": len(logs), "energy_traces": len(energy)}}
    (args.root / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
