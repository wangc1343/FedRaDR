"""Build the machine-readable evidence bundle for a FedRaDR rerun.

The command deliberately separates deterministic rerun assets from historical
evidence.  Schedules and protocol snapshots can be generated locally.  Raw
training logs, partition realizations, and power traces are only indexed when
they already exist or are produced by the measurement wrappers.

Typical usage::

    python scripts/build_reproducibility_bundle.py plan
    python scripts/build_reproducibility_bundle.py generate-schedules
    python scripts/build_reproducibility_bundle.py snapshot-config
    python scripts/build_reproducibility_bundle.py index-logs --results-root results
    python scripts/build_reproducibility_bundle.py manifest

Use ``all`` to run the non-destructive planning/snapshot steps together.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.reproducibility import (
    make_client_schedule,
    make_nested_client_schedule_family,
)
from fedradr.partitions import build_partition_payload, write_partition


SCRIPT_VERSION = "1.0"
SEEDS = (20, 21, 22)
LOG_FIELDS = (
    "round",
    "test_accuracy",
    "train_objective",
    "mfg",
    "hfr_tau",
    "downlink_bytes",
    "uplink_bytes",
    "cumulative_serialized_bytes",
    "single_device_elapsed_seconds",
)

METHODS = (
    "FedAvg",
    "SCAFFOLD",
    "FedDyn",
    "FedVRA",
    "FedDisco",
    "FedSpeedOE",
    "FedKT",
    "FedSKC",
    "FedRaDR",
    "FedRaDR-SAMOE",
)
METHOD_DISPLAY_NAMES = {
    "FedAvg": "FedAvg",
    "SCAFFOLD": "Scaffold",
    "FedDyn": "FedDyn",
    "FedVRA": "FedVRA",
    "FedDisco": "FedDisco",
    "FedSpeedOE": "FedSpeed^OE",
    "FedKT": "FedKT",
    "FedSKC": "FedSKC",
    "FedRaDR": "FedRaDR",
    "FedRaDR-SAMOE": "FedRaDR-SAM^OE",
}

PARTITION_SPECS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "clients": (100, 500),
        "partitions": {
            "D1": {"alpha": 0.3},
            "D2": {"alpha": 0.6},
            "P1": {"n_class_per_client": 3},
            "P2": {"n_class_per_client": 5},
        },
    },
    "cifar100": {
        "clients": (100, 500),
        "partitions": {
            "D1": {"alpha": 0.3},
            "D2": {"alpha": 0.6},
            "P1": {"n_class_per_client": 10},
            "P2": {"n_class_per_client": 20},
        },
    },
    "tinyimagenet": {
        "clients": (100,),
        "partitions": {"D1": {"alpha": 0.1}},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any, *, overwrite: bool) -> dict[str, Any]:
    payload = json_bytes(value)
    existed = path.is_file()
    if existed and not overwrite:
        existing_hash = sha256_file(path)
        expected_hash = sha256_bytes(payload)
        return {
            "path": path.as_posix(),
            "sha256": existing_hash,
            "bytes": path.stat().st_size,
            "expected_sha256": expected_hash,
            "status": "existing_match" if existing_hash == expected_hash else "existing_different",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "status": "written",
    }


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repo_metadata(repo: Path) -> dict[str, Any]:
    remotes: dict[str, dict[str, str]] = {}
    remote_text = git_output(repo, "remote", "-v") or ""
    for line in remote_text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        name, url, direction = parts
        remotes.setdefault(name, {})[direction.strip("()")]=url
    tags = (git_output(repo, "tag", "--points-at", "HEAD") or "").splitlines()
    status = git_output(repo, "status", "--porcelain")
    return {
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "commit_short": git_output(repo, "rev-parse", "--short", "HEAD"),
        "tags_at_commit": tags,
        "dirty": bool(status) if status is not None else None,
        "remotes": remotes,
    }


def provenance(repo: Path, *, origin: str = "generated", historical: bool = False) -> dict[str, Any]:
    return {
        "origin": origin,
        "historical": historical,
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "git": repo_metadata(repo),
    }


def protocol_record(source_document: str | None) -> dict[str, Any]:
    source_info: dict[str, Any] | None = None
    if source_document:
        source_path = Path(source_document)
        source_info = {"path": source_document, "sha256": sha256_file(source_path)} if source_path.is_file() else {
            "path": source_document,
            "sha256": None,
            "status": "not_found",
        }
    return {
        "schema_version": "fedradr.reproduction.v1",
        "source_document": source_info,
        "datasets": {
            "cifar10": {"model": "ResNet-18-GN", "rounds": 1000},
            "cifar100": {"model": "ResNet-18-GN", "rounds": 1000},
            "tinyimagenet": {"model": "ResNet-18-GN", "rounds": 500},
            "mobile_transfer": {
                "datasets": ["cifar100-D1", "tinyimagenet-D1"],
                "model": "MobileNetV2-GN",
            },
        },
        "seeds": list(SEEDS),
        "partition_specs": PARTITION_SPECS,
        "reported_partition_hash_prefixes": {
            "cifar10_D1": "a19cfe9ffd3f",
            "cifar10_D2": "6234713dd488",
            "cifar10_P1": "7406585d0814",
            "cifar10_P2": "e6c84e4ebae2",
            "cifar100_D1": "99754bc40ccf",
            "tinyimagenet_D1": "dbbf1ccc215a",
            "status": "prefixes are manuscript references; full per-seed binaries are still required",
        },
        "settings": [
            {"id": "C100-P10", "clients": 100, "clients_per_round": 10, "rounds": 1000},
            {"id": "C500-P2", "clients": 500, "clients_per_round": 10, "rounds": 1000},
            {
                "id": "C100-P5",
                "clients": 100,
                "clients_per_round": 5,
                "rounds": 1000,
                "purpose": "controlled participation",
            },
            {
                "id": "C100-P2",
                "clients": 100,
                "clients_per_round": 2,
                "rounds": 1000,
                "purpose": "controlled participation",
            },
            {
                "id": "Tiny-C100-P10",
                "clients": 100,
                "clients_per_round": 10,
                "rounds": 500,
            },
        ],
        "methods": [
            {"canonical_id": method, "display_name": METHOD_DISPLAY_NAMES[method]}
            for method in METHODS
        ],
        "shared_hyperparameters": {
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "learning_rate_decay_per_round": 0.998,
            "weight_decay": 0.001,
            "batch_size": 50,
            "local_steps": 5,
            "local_work_unit": "paper says local steps; current FedDC-compatible runner parameter is local_epochs and traverses the loader per epoch; verify against the historical script",
            "aggregation": "data-size weighted",
        },
        "fedradr_hyperparameters": {
            "dynamic_regularization_alpha": 0.1,
            "lambda_max": 0.5,
            "temperature": 4.0,
            "epsilon": 1e-8,
            "normalization": "exact client mean",
        },
        "tuning_protocol": {
            "selection_setting": "CIFAR10-D1/C100-P10",
            "freeze_after_selection": True,
            "lambda_max_candidates": [0.1, 0.3, 0.5, 0.7, 1.0],
            "temperature_candidates": [1, 2, 3, 4, 5],
            "baseline_search_spaces": "required historical input; do not infer",
        },
        "reporting": {
            "seeds": list(SEEDS),
            "censored_target_notation": ">budget",
            "time_definition": "single-device aggregate compute time",
            "raw_log_fields": list(LOG_FIELDS),
            "additional_diagnostics": [
                "forward_flops",
                "backward_flops",
                "peak_memory_bytes",
                "gradient_variance",
                "client_drift_norm",
                "update_cosine_similarity",
                "missing_class_probability_mass_retention",
                "missing_class_accuracy_delta",
            ],
            "diagnostic_subset": "fixed client-local V_i, never used for model updates",
            "hfr_threshold": "pre-registered FedAvg 90th percentile at the matched early round",
            "implementation_status": "The current FedRaDR runner logs training-batch diagnostics; provide fixed held-out V_i and tau artifacts before claiming exact paper MFG/HFR reproduction.",
        },
        "artifact_policy": {
            "generated_schedules_are_historical": False,
            "aggregates_are_raw_logs": False,
            "generated_power_traces_allowed": False,
            "generated_training_logs_allowed": False,
        },
        "campaigns": {
            "primary": {
                "name": "main_tables",
                "partition_and_schedule_reuse": "within matched setting and seed",
            },
            "appendix_b_independent": {
                "name": "independent_per_seed_reproduction",
                "partition_and_schedule_reuse": "independently generated from primary campaign",
                "must_not_reuse_primary_assets": True,
            },
        },
    }


def measurement_protocol() -> dict[str, Any]:
    return {
        "schema_version": "fedradr.energy.v1",
        "integration": "trapezoidal over timestamped power samples",
        "required_raw_files": ["*.samples.csv", "*.summary.json"],
        "desktop_nvidia": {
            "device": "NVIDIA RTX 4090",
            "power_scope": "gpu_board",
            "sampling_interval_ms": 100,
            "warmup_runs_excluded": 1,
            "measured_seeds": list(SEEDS),
            "required_metadata": [
                "gpu_index",
                "gpu_identity_and_driver",
                "command",
                "return_code",
                "elapsed_seconds",
                "sample_count",
            ],
            "command_template": "python scripts/measure_nvidia_energy.py --gpu-index 0 --interval-ms 100 --output-prefix <prefix> -- <experiment command>",
        },
        "jetson_orin": {
            "device": "Jetson AGX Orin",
            "mode": "MAXN",
            "dataset": "Tiny-ImageNet",
            "model": "ResNet-18-GN",
            "batch_size": 32,
            "fixed_client_shard_samples": 1000,
            "local_epochs": 5,
            "warmup_rounds_excluded": 1,
            "sampling_interval_ms": 100,
            "power_scope": "explicit tegrastats rail (for example VDD_IN)",
            "measured_seeds": list(SEEDS),
            "required_metadata": ["rail", "raw_tegrastats_line", "timestamps", "sample_count"],
            "command_template": "python scripts/measure_jetson_energy.py --rail VDD_IN --interval-ms 100 --output-prefix <prefix> -- <experiment command>",
        },
        "variants": ["FedAvg", "FedRaDR-exact", "FedRaDR-mini-batch", "FedRaDR-EMA"],
        "missing_evidence_rule": "A protocol file is not a measurement; retain raw traces before reporting joules.",
    }


def baseline_tuning_template(repo: Path) -> dict[str, Any]:
    baseline_path = repo / "reproducibility" / "configs" / "baseline_hyperparameters.json"
    baseline: dict[str, Any] = {}
    if baseline_path.is_file():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline = {}
    methods = baseline.get("methods", {}) if isinstance(baseline, dict) else {}
    records: dict[str, Any] = {}
    for method in METHODS:
        source_name = method.replace("OE", "")
        source = methods.get(source_name, methods.get(method, {}))
        records[method] = {
            "search_space": None,
            "selected": source.get("method_specific_parameters", {}) if isinstance(source, dict) else {},
            "optimizer": source.get("optimizer", "SGD") if isinstance(source, dict) else "SGD",
            "evidence": "historical search grid required before claiming exact baseline tuning",
        }
    return {
        "schema_version": "fedradr.tuning.v1",
        "selection_setting": "CIFAR10-D1/C100-P10",
        "selection_rule": "Search under the shared protocol, select once, then freeze for test settings.",
        "status": "incomplete_until_original_baseline_grids_are_supplied",
        "methods": records,
        "fedradr": {
            "search_space": {
                "lambda_max": [0.1, 0.3, 0.5, 0.7, 1.0],
                "temperature": [1, 2, 3, 4, 5],
            },
            "selected": {"lambda_max": 0.5, "temperature": 4.0},
            "epsilon": 1e-8,
        },
        "provenance": provenance(repo),
    }


def generate_schedules(root: Path, *, overwrite: bool, repo: Path) -> list[dict[str, Any]]:
    schedule_root = root / "schedules"
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        nested, permutation_hashes = make_nested_client_schedule_family(100, [10, 5, 2], 1000, seed)
        family_id = f"c100-r1000-seed{seed}"
        family_path = root / "schedule_families" / f"{family_id}.permutation_hashes.json"
        family_result = write_json(family_path, permutation_hashes, overwrite=overwrite)
        family_hash = family_result["sha256"]
        for count, schedule in nested.items():
            name = f"c100-p{count}-r1000-seed{seed}.json"
            payload = {
                "num_clients": 100,
                "clients_per_round": count,
                "rounds": 1000,
                "seed": seed,
                "schedule_family": "nested_permutation_prefix",
                "family_id": family_id,
                "base_permutation_hashes_sha256": family_hash,
                "base_permutation_count": len(permutation_hashes),
                "historical": False,
                "schedule": schedule,
                "provenance": provenance(repo),
            }
            result = write_json(schedule_root / name, payload, overwrite=overwrite)
            result.update({"kind": "client_schedule", "seed": seed, "setting": name[:-5]})
            records.append(result)
        regular_specs = ((500, 10, 1000), (100, 10, 500))
        for clients, count, rounds in regular_specs:
            schedule = make_client_schedule(clients, count, rounds, seed)
            name = f"c{clients}-p{int(100 * count / clients)}-r{rounds}-seed{seed}.json"
            payload = {
                "num_clients": clients,
                "clients_per_round": count,
                "rounds": rounds,
                "seed": seed,
                "schedule_family": "independent_round_sampling",
                "historical": False,
                "schedule": schedule,
                "provenance": provenance(repo),
            }
            result = write_json(schedule_root / name, payload, overwrite=overwrite)
            result.update({"kind": "client_schedule", "seed": seed, "setting": name[:-5]})
            records.append(result)
    return records


def generate_partitions(
    root: Path,
    labels_root: Path,
    *,
    overwrite: bool,
    require_all: bool,
) -> dict[str, Any]:
    """Generate every declared partition from caller-supplied label arrays."""
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    output_root = root / "partitions"
    for dataset, spec in PARTITION_SPECS.items():
        label_path = labels_root / f"{dataset}_train_labels.npy"
        if not label_path.is_file():
            missing.append(label_path.as_posix())
            continue
        labels = np.load(label_path, allow_pickle=False)
        if labels.ndim != 1:
            raise ValueError(f"{label_path} must contain a one-dimensional label array")
        for clients in spec["clients"]:
            for partition, hyper in spec["partitions"].items():
                for seed in SEEDS:
                    filename = f"{dataset}_N{clients}_{partition}_seed{seed}.pkl"
                    path = output_root / filename
                    if path.exists() and not overwrite:
                        records.append({
                            "path": path.as_posix(),
                            "sha256": sha256_file(path),
                            "status": "existing",
                            "historical": None,
                        })
                        continue
                    payload = build_partition_payload(
                        labels,
                        dataset,
                        partition,
                        clients,
                        seed,
                        alpha=hyper.get("alpha"),
                        classes_per_client=hyper.get("n_class_per_client"),
                    )
                    digest = write_partition(payload, path)
                    records.append({
                        "path": path.as_posix(),
                        "sha256": digest,
                        "status": "written",
                        "historical": False,
                        "labels_sha256": payload["labels_sha256"],
                        "sample_count": payload["sample_count"],
                        "algorithm_version": payload["algorithm_version"],
                    })
    result = {
        "schema_version": "fedradr.partition-index.v1",
        "generated_at_utc": utc_now(),
        "labels_root": labels_root.as_posix(),
        "records": records,
        "missing_label_arrays": missing,
        "complete": not missing and len(records) == 51,
        "policy": "Generated partitions are rerun assets until their full hashes match recovered historical files.",
    }
    write_json(output_root / "generated_index.json", result, overwrite=overwrite)
    if require_all and not result["complete"]:
        raise FileNotFoundError(
            "missing label arrays or partition records; see "
            f"{(output_root / 'generated_index.json').as_posix()}"
        )
    return result


def snapshot_configs(root: Path, repo: Path, *, overwrite: bool, source_document: str | None) -> list[dict[str, Any]]:
    config_root = root / "configs"
    protocol = protocol_record(source_document)
    generated = {
        "protocol": protocol,
        "provenance": provenance(repo),
    }
    outputs = [
        (config_root / "experiment_protocol.generated.json", generated),
        (config_root / "baseline_tuning_template.json", baseline_tuning_template(repo)),
    ]
    return [write_json(path, value, overwrite=overwrite) for path, value in outputs]


def capture_version(root: Path, repo: Path, *, overwrite: bool) -> dict[str, Any]:
    record = {
        "schema_version": "fedradr.code-version.v1",
        "captured_at_utc": utc_now(),
        **repo_metadata(repo),
        "provenance": provenance(repo),
    }
    return write_json(root / "code_version.json", record, overwrite=overwrite)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(root: Path, repo: Path, *, overwrite: bool) -> dict[str, Any]:
    record = {
        "schema_version": "fedradr.environment.v1",
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
        },
        "git": repo_metadata(repo),
        "provenance": provenance(repo),
    }
    return write_json(root / "environment" / "bundle_capture.json", record, overwrite=overwrite)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [str(error)]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: record is not an object")
            continue
        missing = [field for field in LOG_FIELDS if field not in record]
        if missing:
            errors.append(f"line {line_number}: missing {', '.join(missing)}")
            continue
        if not isinstance(record.get("round"), int) or isinstance(record.get("round"), bool) or record["round"] < 0:
            errors.append(f"line {line_number}: round must be a nonnegative integer")
        for field in LOG_FIELDS:
            if field == "hfr_tau" and record[field] is None:
                continue
            if not _finite_number(record[field]):
                errors.append(f"line {line_number}: {field} is not finite numeric")
        if _finite_number(record.get("test_accuracy")) and not 0 <= float(record["test_accuracy"]) <= 1:
            errors.append(f"line {line_number}: test_accuracy must be in [0, 1]")
        if _finite_number(record.get("hfr_tau")) and not 0 <= float(record["hfr_tau"]) <= 1:
            errors.append(f"line {line_number}: hfr_tau must be in [0, 1]")
        for field in ("downlink_bytes", "uplink_bytes", "cumulative_serialized_bytes"):
            if _finite_number(record.get(field)) and float(record[field]) < 0:
                errors.append(f"line {line_number}: {field} must be nonnegative")
        rows.append(record)
    rounds = [row.get("round") for row in rows]
    if len(rounds) != len(set(rounds)):
        errors.append("round values are not unique")
    if rounds:
        numeric_rounds = sorted(rounds)
        expected_zero = list(range(len(rounds)))
        expected_one = list(range(1, len(rounds) + 1))
        if numeric_rounds not in (expected_zero, expected_one):
            errors.append("round values are not contiguous from zero or one")
    previous_cumulative = None
    for line_number, row in enumerate(rows, 1):
        cumulative = row.get("cumulative_serialized_bytes")
        if _finite_number(cumulative):
            if previous_cumulative is not None and float(cumulative) < previous_cumulative:
                errors.append(f"record {line_number}: cumulative_serialized_bytes decreases")
            if _finite_number(row.get("downlink_bytes")) and _finite_number(row.get("uplink_bytes")):
                expected_cumulative = (previous_cumulative or 0.0) + float(row["downlink_bytes"]) + float(row["uplink_bytes"])
                if not math.isclose(float(cumulative), expected_cumulative, rel_tol=0.0, abs_tol=0.5):
                    errors.append(
                        f"record {line_number}: cumulative_serialized_bytes does not equal accumulated payload"
                    )
            previous_cumulative = float(cumulative)
    return rows, errors


def export_rows(rows: Iterable[dict[str, Any]], output: Path, *, overwrite: bool) -> dict[str, Any]:
    if output.exists() and not overwrite:
        return {"path": output.as_posix(), "sha256": sha256_file(output), "status": "existing"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in LOG_FIELDS})
    return {"path": output.as_posix(), "sha256": sha256_file(output), "status": "written"}


def index_logs(root: Path, results_root: Path, *, overwrite: bool, copy_raw: bool) -> dict[str, Any]:
    log_root = root / "raw_logs"
    entries: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("metrics.jsonl")) if results_root.is_dir() else []:
        rows, errors = validate_log(path)
        relative = path.relative_to(results_root)
        stem = "_".join(relative.parent.parts) or path.parent.name
        csv_path = log_root / f"{stem}.csv"
        config_path = path.with_name("config.json")
        csv_record = export_rows(rows, csv_path, overwrite=overwrite) if not errors else None
        sidecar_path = None
        if csv_record is not None:
            sidecar_path = csv_path.with_suffix(".source.json")
            write_json(
                sidecar_path,
                {
                    "source_jsonl": path.as_posix(),
                    "source_sha256": sha256_file(path),
                    "config_json": config_path.as_posix() if config_path.is_file() else None,
                    "derived_fields": list(LOG_FIELDS),
                },
                overwrite=overwrite,
            )
        raw_copy_path = None
        if copy_raw:
            raw_copy_path = log_root / "source" / f"{stem}.metrics.jsonl"
            if overwrite or not raw_copy_path.exists():
                raw_copy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, raw_copy_path)
        run_provenance: dict[str, Any] = {}
        if config_path.is_file():
            try:
                config_record = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(config_record, dict):
                    candidate = config_record.get("provenance")
                    if isinstance(candidate, dict):
                        run_provenance = candidate
            except (OSError, json.JSONDecodeError):
                run_provenance = {}
        historical = run_provenance.get("historical")
        origin = "recovered" if historical is True else "indexed_existing"
        entry = {
            "source_jsonl": path.as_posix(),
            "source_sha256": sha256_file(path),
            "source_bytes": path.stat().st_size,
            "rounds": len(rows),
            "valid": not errors,
            "errors": errors,
            "config": config_path.as_posix() if config_path.is_file() else None,
            "derived_csv": csv_record,
            "derived_sidecar": sidecar_path.as_posix() if sidecar_path else None,
            "retained_raw_jsonl": raw_copy_path.as_posix() if raw_copy_path else None,
            "provenance": {"origin": origin, "historical": historical, "run": run_provenance},
        }
        entries.append(entry)
    index = {
        "schema_version": "fedradr.raw-log-index.v1",
        "generated_at_utc": utc_now(),
        "source_root": results_root.as_posix(),
        "entries": entries,
        "valid_count": sum(bool(entry["valid"]) for entry in entries),
        "invalid_count": sum(not bool(entry["valid"]) for entry in entries),
        "policy": "Only metrics.jsonl is treated as a raw training log; aggregate CSVs are never promoted.",
    }
    # The index is a derived inventory and must refresh when new runs appear.
    write_result = write_json(log_root / "index.json", index, overwrite=True)
    return {"index": write_result, "entries": entries}


def inventory(root: Path, repo: Path) -> dict[str, Any]:
    def files(pattern: str, directory: Path) -> list[dict[str, Any]]:
        values = []
        for path in sorted(directory.glob(pattern)) if directory.is_dir() else []:
            if not path.is_file():
                continue
            values.append({
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        return values

    schedules = files("*.json", root / "schedules")
    partitions = files("*.pkl", root / "partitions")
    logs = [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((root / "raw_logs").rglob("*.jsonl"))
        if path.is_file()
    ] + files("*.csv", root / "raw_logs")
    powers = files("*.samples.csv", root / "power_traces")
    version = root / "code_version.json"
    version_record: dict[str, Any] = {}
    if version.is_file():
        try:
            loaded_version = json.loads(version.read_text(encoding="utf-8"))
            if isinstance(loaded_version, dict):
                version_record = loaded_version
        except (OSError, json.JSONDecodeError):
            version_record = {}
    manifest = {
        "schema_version": "fedradr.artifact-manifest.v1",
        "generated_at_utc": utc_now(),
        "provenance": provenance(repo),
        "counts": {
            "schedules": len(schedules),
            "partitions": len(partitions),
            "raw_log_files": len(logs),
            "power_sample_files": len(powers),
        },
        "expected": {
            "schedules": 15,
            "partitions": 51,
            "raw_per_round_logs": "expanded from protocol; no generated target",
            "power_traces": 12,
        },
        "status": {
            "schedules": len(schedules) >= 15,
            "partitions": len(partitions) >= 51,
            "raw_logs": any(path.endswith(".jsonl") for path in [item["path"] for item in logs]),
            "power_traces": len(powers) >= 12,
            "baseline_search_grids": False,
            "code_version": bool(
                version_record.get("commit")
                and version_record.get("tags_at_commit")
                and version_record.get("dirty") is False
            ),
        },
        "artifacts": {
            "schedules": schedules,
            "partitions": partitions,
            "raw_logs": logs,
            "power_traces": powers,
        },
        "notes": [
            "Generated schedules are deterministic rerun assets, not recovered historical schedules.",
            "A valid aggregate table is not a substitute for metrics.jsonl/stdout.",
            "Energy values require raw timestamped traces and an explicit power scope.",
        ],
    }
    result = write_json(root / "bundle_manifest.json", manifest, overwrite=True)
    return {"manifest": result, "record": manifest}


def audit_bundle(root: Path, repo: Path) -> dict[str, Any]:
    """Run evidence checks without promoting any aggregate or generated file."""
    checks: dict[str, Any] = {}
    schedule_paths = sorted((root / "schedules").glob("*.json")) if (root / "schedules").is_dir() else []
    checks["schedules"] = {
        "found": len(schedule_paths),
        "required": 15,
        "complete": len(schedule_paths) >= 15,
        "errors": [],
    }
    for path in schedule_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root is not an object")
            schedule = payload.get("schedule")
            clients = int(payload["num_clients"])
            count = int(payload["clients_per_round"])

            def valid_round(round_clients: object) -> bool:
                if not isinstance(round_clients, list) or len(round_clients) != count:
                    return False
                if any(
                    not isinstance(client, int)
                    or isinstance(client, bool)
                    or not 0 <= client < clients
                    for client in round_clients
                ):
                    return False
                return len(set(round_clients)) == len(round_clients)

            if not isinstance(schedule, list) or any(not valid_round(round_clients) for round_clients in schedule):
                checks["schedules"]["errors"].append(f"{path.name}: invalid schedule shape")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            checks["schedules"]["errors"].append(f"{path.name}: {error}")
    checks["schedules"]["complete"] = checks["schedules"]["complete"] and not checks["schedules"]["errors"]

    partition_paths = sorted((root / "partitions").glob("*.pkl")) if (root / "partitions").is_dir() else []
    checks["partitions"] = {
        "found": len(partition_paths),
        "required": 51,
        "complete": len(partition_paths) >= 51,
        "errors": [] if len(partition_paths) >= 51 else ["historical/generated partition files are missing"],
    }

    log_index = root / "raw_logs" / "index.json"
    try:
        log_record = json.loads(log_index.read_text(encoding="utf-8"))
        invalid_logs = int(log_record.get("invalid_count", 0))
        log_count = len(log_record.get("entries", []))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        invalid_logs = 0
        log_count = 0
    checks["raw_logs"] = {
        "found": log_count,
        "invalid": invalid_logs,
        "complete": log_count > 0 and invalid_logs == 0,
        "errors": [] if log_count > 0 and invalid_logs == 0 else ["valid per-round metrics.jsonl evidence is missing"],
    }

    power_paths = sorted((root / "power_traces").glob("*.samples.csv")) if (root / "power_traces").is_dir() else []
    power_errors: list[str] = []
    for path in power_paths:
        summary = path.with_name(path.name.replace(".samples.csv", ".summary.json"))
        if not summary.is_file():
            power_errors.append(f"{path.name}: missing summary sidecar")
        try:
            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) < 2:
                power_errors.append(f"{path.name}: fewer than two samples")
            timestamps = [float(row["timestamp_ms"]) for row in rows]
            if timestamps != sorted(timestamps):
                power_errors.append(f"{path.name}: timestamps are not monotonic")
            scopes = {row.get("power_scope") for row in rows}
            if len(scopes) != 1:
                power_errors.append(f"{path.name}: power_scope changes within trace")
        except (OSError, ValueError, KeyError):
            power_errors.append(f"{path.name}: malformed sample file")
    checks["power_traces"] = {
        "found": len(power_paths),
        "required": 12,
        "complete": len(power_paths) >= 12 and not power_errors,
        "errors": power_errors if power_errors else ([] if len(power_paths) >= 12 else ["raw power traces are missing"]),
    }

    tuning_path = root / "configs" / "tuning_protocol.json"
    try:
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        baseline_complete = bool(tuning.get("baselines", {}).get("exact_search_spaces_and_selected_values"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        baseline_complete = False
    checks["baseline_tuning"] = {
        "complete": baseline_complete,
        "errors": [] if baseline_complete else ["exact baseline grids and selections are missing"],
    }
    version_path = root / "code_version.json"
    try:
        version = json.loads(version_path.read_text(encoding="utf-8"))
        version_complete = bool(version.get("commit") and version.get("tags_at_commit") and version.get("dirty") is False)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        version_complete = False
    checks["code_version"] = {
        "complete": version_complete,
        "errors": [] if version_complete else ["a clean tagged code version is required"],
    }
    audit = {
        "schema_version": "fedradr.bundle-audit.v1",
        "generated_at_utc": utc_now(),
        "complete": all(bool(value["complete"]) for value in checks.values()),
        "checks": checks,
        "policy": "Incomplete audits are expected until historical partitions, raw logs, traces, baseline grids, and a tagged clean commit are supplied.",
        "provenance": provenance(repo),
    }
    write_json(root / "bundle_audit.json", audit, overwrite=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "generate-schedules", "generate-partitions", "snapshot-config", "capture", "index-logs", "manifest", "audit", "all"))
    parser.add_argument("--root", type=Path, default=Path("reproducibility"), help="Artifact root (default: reproducibility)")
    parser.add_argument("--repo", type=Path, default=None, help="Repository root (default: inferred from this script)")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--labels-root", type=Path, default=Path("labels"), help="Directory containing <dataset>_train_labels.npy files")
    parser.add_argument("--source-document", default=None, help="Optional paper path recorded as provenance")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files; existing evidence is never deleted")
    parser.add_argument("--require-valid-logs", action="store_true", help="Fail if an indexed JSONL log is malformed")
    parser.add_argument("--copy-raw-logs", action="store_true", help="Copy genuine metrics.jsonl files under raw_logs/source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = (args.repo or Path(__file__).resolve().parents[1]).resolve()
    root = args.root if args.root.is_absolute() else repo / args.root
    source_document = args.source_document
    if source_document:
        candidate = Path(source_document)
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = repo / candidate
        source_document = candidate.as_posix()
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Any] = []

    if args.command in {"plan", "all"}:
        outputs.append(write_json(root / "experiment_plan.json", protocol_record(source_document), overwrite=args.overwrite))
    if args.command in {"generate-schedules", "all"}:
        outputs.extend(generate_schedules(root, overwrite=args.overwrite, repo=repo))
    if args.command == "generate-partitions":
        labels_root = args.labels_root if args.labels_root.is_absolute() else repo / args.labels_root
        try:
            result = generate_partitions(root, labels_root, overwrite=args.overwrite, require_all=True)
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        outputs.append({"partition_count": len(result["records"]), "complete": result["complete"]})
    if args.command in {"snapshot-config", "all"}:
        outputs.extend(snapshot_configs(root, repo, overwrite=args.overwrite, source_document=source_document))
        outputs.append(write_json(root / "energy" / "measurement_protocol.json", measurement_protocol(), overwrite=args.overwrite))
    if args.command in {"capture", "all"}:
        outputs.append(capture_version(root, repo, overwrite=args.overwrite))
        outputs.append(capture_environment(root, repo, overwrite=args.overwrite))
    if args.command in {"index-logs", "all"}:
        indexed = index_logs(
            root,
            args.results_root if args.results_root.is_absolute() else repo / args.results_root,
            overwrite=args.overwrite,
            copy_raw=args.copy_raw_logs,
        )
        outputs.append(indexed["index"])
        if args.require_valid_logs and any(not entry["valid"] for entry in indexed["entries"]):
            raise SystemExit("one or more metrics.jsonl files failed validation")
    if args.command in {"manifest", "all"}:
        outputs.append(inventory(root, repo)["manifest"])
    if args.command == "audit":
        audit = audit_bundle(root, repo)
        outputs.append({"audit": (root / "bundle_audit.json").as_posix(), "complete": audit["complete"]})
        if not audit["complete"]:
            print(json.dumps(audit, indent=2))
            raise SystemExit(1)

    print(json.dumps({"command": args.command, "root": root.as_posix(), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
