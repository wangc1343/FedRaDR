"""Summarize genuine per-round JSONL logs without fabricating missing curves.

The output keeps right-censored targets explicit (``>budget``) and retains a
source SHA-256 for every summary row. It never reads Appendix aggregate CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_reproducibility_bundle import validate_log


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(path: Path, root: Path, target: float, budget: int | None) -> dict[str, Any]:
    rows, errors = validate_log(path)
    if errors:
        return {
            "source": path.as_posix(),
            "source_sha256": sha256_file(path),
            "valid": False,
            "errors": errors,
        }
    effective_budget = budget if budget is not None else (len(rows) if rows else 0)
    hits = [row for row in rows if float(row["test_accuracy"]) >= target]
    hit = hits[0] if hits else None
    at_budget = next((row for row in rows if int(row["round"]) == effective_budget), None)
    if at_budget is None and rows:
        at_budget = rows[-1]
    return {
        "source": path.as_posix(),
        "source_sha256": sha256_file(path),
        "valid": True,
        "target_accuracy": target,
        "budget_rounds": effective_budget,
        "first_hit_round": int(hit["round"]) if hit else f">{effective_budget}",
        "censored": hit is None,
        "accuracy_at_budget": float(at_budget["test_accuracy"]) if at_budget else None,
        "final_accuracy": float(rows[-1]["test_accuracy"]) if rows else None,
        "rounds": len(rows),
        "config": (path.with_name("config.json").as_posix() if path.with_name("config.json").is_file() else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    if not 0 <= args.target <= 1:
        parser.error("--target must be in [0, 1]")
    if args.budget is not None and args.budget < 0:
        parser.error("--budget must be nonnegative")
    entries = [
        summarize(path, args.input_root, args.target, args.budget)
        for path in sorted(args.input_root.rglob("metrics.jsonl"))
    ] if args.input_root.is_dir() else []
    record = {
        "schema_version": "fedradr.log-summary.v1",
        "target_accuracy": args.target,
        "budget_rounds": args.budget,
        "entries": entries,
        "policy": "Censored targets remain >budget; no speedup is inferred from missing references.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(entries), "output": args.output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
