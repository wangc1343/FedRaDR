from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil

from scripts.validate_partitions import expected_files, validate_payload


LOG_COLUMNS = {
    "round",
    "test_accuracy",
    "train_objective",
    "mfg",
    "hfr_tau",
    "downlink_bytes",
    "uplink_bytes",
    "cumulative_serialized_bytes",
    "single_device_elapsed_seconds",
}
POWER_COLUMNS = {"timestamp_ms", "power_watts", "power_scope"}
EXPECTED_COUNTS = {
    "partitions": 51,
    "run_logs": 240,
    "power_traces": 12,
    "schedules": 15,
    "configs": 4,
    "reported_results": 5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_columns(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def audit_partitions(root: Path) -> dict[str, object]:
    errors: list[str] = []
    valid_paths: list[Path] = []
    for dataset, clients, partition, seed, hyper in expected_files():
        filename = f"{dataset}_N{clients}_{partition}_seed{seed}.pkl"
        path = root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        try:
            payload = pickle.loads(path.read_bytes())
            payload_errors = validate_payload(payload, clients, seed, hyper)
        except Exception as error:
            payload_errors = [f"pickle load failed: {error}"]
        if payload_errors:
            errors.extend(f"{filename}: {message}" for message in payload_errors)
        else:
            valid_paths.append(path)
    return {
        "required": EXPECTED_COUNTS["partitions"],
        "valid": len(valid_paths),
        "complete": not errors and len(valid_paths) == EXPECTED_COUNTS["partitions"],
        "errors": errors,
        "files": valid_paths,
    }


def audit_csv_group(root: Path, pattern: str, required: int, columns: set[str]) -> dict[str, object]:
    paths = sorted(root.glob(pattern)) if root.is_dir() else []
    errors: list[str] = []
    valid_paths: list[Path] = []
    for path in paths:
        missing = columns - csv_columns(path)
        if missing:
            errors.append(f"{path.name}: missing columns {sorted(missing)}")
        else:
            valid_paths.append(path)
    if len(valid_paths) < required:
        errors.append(f"found {len(valid_paths)} valid files; require at least {required}")
    return {
        "required": required,
        "valid": len(valid_paths),
        "complete": not errors,
        "errors": errors,
        "files": valid_paths,
    }


def audit_simple_group(root: Path, pattern: str, required: int) -> dict[str, object]:
    paths = sorted(root.glob(pattern)) if root.is_dir() else []
    return {
        "required": required,
        "valid": len(paths),
        "complete": len(paths) >= required,
        "errors": [] if len(paths) >= required else [f"found {len(paths)} files; require at least {required}"],
        "files": paths,
    }


def audit_reported_trace_references(root: Path) -> dict[str, object]:
    errors: list[str] = []
    checked = 0
    reported_root = root / "reported_results"
    trace_root = root / "power_traces"
    for path in sorted(reported_root.glob("*.csv")) if reported_root.is_dir() else []:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                for row_number, row in enumerate(csv.DictReader(stream), start=2):
                    trace_name = row.get("trace_file")
                    if not trace_name:
                        continue
                    checked += 1
                    candidates = [trace_root / trace_name]
                    if not trace_name.endswith(".samples.csv"):
                        candidates.append(trace_root / trace_name.replace(".csv", ".samples.csv"))
                    if not any(candidate.is_file() for candidate in candidates):
                        errors.append(f"{path.name}:{row_number}: missing trace {trace_name}")
        except (OSError, csv.Error) as error:
            errors.append(f"{path.name}: cannot read CSV ({error})")
    return {"checked": checked, "complete": not errors, "errors": errors}


def audit_tuning_record(path: Path) -> dict[str, object]:
    errors: list[str] = []
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        details = record.get("baselines", {}).get("exact_search_spaces_and_selected_values")
        if not details:
            errors.append("baseline exact search spaces and selected values are missing")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        errors.append(f"cannot read tuning record: {error}")
    return {
        "required": 1,
        "valid": 0 if errors else 1,
        "complete": not errors,
        "errors": errors,
        "files": [path] if not errors else [],
    }


def audit_code_version(path: Path) -> dict[str, object]:
    errors: list[str] = []
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("commit"):
            errors.append("release commit is missing")
        if not record.get("tags_at_commit"):
            errors.append("release tag is missing")
        if record.get("dirty") is not False:
            errors.append("captured working tree is not clean")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        errors.append(f"cannot read code-version record: {error}")
    return {
        "required": 1,
        "valid": 0 if errors else 1,
        "complete": not errors,
        "errors": errors,
        "files": [path] if not errors else [],
    }


def public_result(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "files"}


def copy_group(paths: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in paths:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and package genuine supplementary experiment artifacts."
    )
    parser.add_argument("--source-root", type=Path, default=Path("reproducibility"))
    parser.add_argument("--output-root", type=Path, default=Path("supplementary_material"))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reproducibility/supplementary_export_audit.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source = args.source_root
    checks = {
        "partitions": audit_partitions(source / "partitions"),
        "run_logs": audit_csv_group(
            source / "raw_logs", "*.csv", EXPECTED_COUNTS["run_logs"], LOG_COLUMNS
        ),
        "power_traces": audit_csv_group(
            source / "power_traces", "*.samples.csv", EXPECTED_COUNTS["power_traces"], POWER_COLUMNS
        ),
        "schedules": audit_simple_group(source / "schedules", "*.json", EXPECTED_COUNTS["schedules"]),
        "configs": audit_simple_group(source / "configs", "*.*", EXPECTED_COUNTS["configs"]),
        "reported_results": audit_simple_group(
            source / "reported_results", "*.csv", EXPECTED_COUNTS["reported_results"]
        ),
        "reported_trace_references": audit_reported_trace_references(source),
        "baseline_tuning_record": audit_tuning_record(source / "configs" / "tuning_protocol.json"),
        "tagged_code_version": audit_code_version(source / "code_version.json"),
    }
    complete = all(bool(result["complete"]) for result in checks.values())
    audit = {"complete": complete, "checks": {key: public_result(value) for key, value in checks.items()}}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    for name, result in checks.items():
        print(f"{name}: {result['valid']}/{result['required']} valid")
    if args.audit_only:
        raise SystemExit(0 if complete else 1)
    if not complete:
        print(f"refusing to export incomplete evidence; see {args.audit_output}")
        raise SystemExit(1)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_root}")

    copied: list[Path] = []
    for name, result in checks.items():
        copied.extend(copy_group(result["files"], args.output_root / name))
    for filename in ("artifact_manifest.json", "code_version.json"):
        path = source / filename
        if path.is_file():
            copied.extend(copy_group([path], args.output_root / "metadata"))
    for path in sorted((source / "power_traces").glob("*.summary.json")):
        copied.extend(copy_group([path], args.output_root / "power_traces"))
    for source_path, target_name in (
        (Path("README_Supplementary.md"), "README.md"),
        (Path(__file__), Path(__file__).name),
        (Path("calc_hardware_summary.py"), "calc_hardware_summary.py"),
    ):
        if source_path.is_file():
            target = args.output_root / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            copied.append(target)

    manifest = {
        "files": [
            {
                "path": path.relative_to(args.output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(copied)
        ]
    }
    manifest_path = args.output_root / "SHA256_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(copied)} genuine files to {args.output_root}")


if __name__ == "__main__":
    main()
