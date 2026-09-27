from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDS = (
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a FedRaDR metrics JSONL file as reviewer-ready CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional run config retained in the CSV sidecar.")
    parser.add_argument("--no-sidecar", action="store_true", help="Do not write the source-hash sidecar.")
    args = parser.parse_args()

    rows = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(f"line {line_number} is missing fields: {', '.join(missing)}")
        if not isinstance(record["round"], int) or isinstance(record["round"], bool) or record["round"] < 0:
            raise ValueError(f"line {line_number} round must be a nonnegative integer")
        for field in FIELDS:
            if field == "hfr_tau" and record[field] is None:
                continue
            value = record[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"line {line_number} field {field} must be finite numeric")
        if not 0 <= float(record["test_accuracy"]) <= 1:
            raise ValueError(f"line {line_number} test_accuracy must be in [0, 1]")
        if record["hfr_tau"] is not None and not 0 <= float(record["hfr_tau"]) <= 1:
            raise ValueError(f"line {line_number} hfr_tau must be in [0, 1]")
        if any(float(record[field]) < 0 for field in ("downlink_bytes", "uplink_bytes", "cumulative_serialized_bytes")):
            raise ValueError(f"line {line_number} payload bytes must be nonnegative")
        rows.append({field: record[field] for field in FIELDS})

    rounds = [int(row["round"]) for row in rows]
    if rounds and sorted(rounds) not in (list(range(len(rounds))), list(range(1, len(rounds) + 1))):
        raise ValueError("round values must be contiguous from zero or one")
    previous = None
    for line_number, row in enumerate(rows, 1):
        expected = (previous or 0) + row["downlink_bytes"] + row["uplink_bytes"]
        if not math.isclose(row["cumulative_serialized_bytes"], expected, abs_tol=0.5):
            raise ValueError(f"line {line_number} cumulative_serialized_bytes is inconsistent")
        previous = row["cumulative_serialized_bytes"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_sidecar:
        digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
        sidecar = args.output.with_suffix(".source.json")
        sidecar.write_text(
            json.dumps(
                {
                    "source_jsonl": str(args.input),
                    "source_sha256": digest,
                    "config_json": str(args.config) if args.config else None,
                    "fields": list(FIELDS),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"exported {len(rows)} rounds to {args.output}")


if __name__ == "__main__":
    main()
