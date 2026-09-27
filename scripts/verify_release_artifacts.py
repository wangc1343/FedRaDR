from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.is_dir() else 0


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def audit_reported_trace_references(root: Path) -> dict[str, object]:
    """Find dangling trace_file references in reported resource tables."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reviewer-facing reproducibility artifacts.")
    parser.add_argument("--root", type=Path, default=Path("reproducibility"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tuning = load_json(args.root / "configs" / "tuning_protocol.json")
    baseline_tuning = tuning.get("baselines", {}) if isinstance(tuning, dict) else {}
    version = load_json(args.root / "code_version.json")
    checks: dict[str, dict[str, object]] = {
        "partition_files": {"found": count(args.root / "partitions", "*.pkl"), "required": 51},
        "client_schedules": {"found": count(args.root / "schedules", "*.json"), "required": 15},
        "configuration_records": {"found": count(args.root / "configs", "*.*"), "required": 4},
        "baseline_tuning_details": {
            "complete": bool(isinstance(baseline_tuning, dict) and baseline_tuning.get("exact_search_spaces_and_selected_values"))
        },
        "code_version": {
            "commit": version.get("commit"),
            "tags_at_commit": version.get("tags_at_commit", []),
            "dirty": version.get("dirty"),
            "complete": bool(version.get("commit") and version.get("dirty") is False),
        },
        "per_round_logs": {
            "found": count(args.root / "raw_logs", "*.csv"),
            "required": 240,
            "scope": "Appendix B: 8 settings x 10 methods x 3 seeds",
        },
        "power_traces": {
            "found": count(args.root / "power_traces", "*.csv"),
            "required": 12,
            "scope": "Appendix D: 4 variants x 3 seeds",
        },
        "reported_result_tables": {"found": count(args.root / "reported_results", "*.csv"), "required": 5},
        "reported_trace_references": audit_reported_trace_references(args.root),
    }
    for value in checks.values():
        if "complete" not in value:
            value["complete"] = int(value["found"]) >= int(value["required"])
    report = {"complete": all(value["complete"] for value in checks.values()), "checks": checks}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
