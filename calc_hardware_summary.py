from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fedradr.energy import integrate_power_samples


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    return None


def summarize_trace(path: Path) -> dict[str, object]:
    samples: list[tuple[float, float]] = []
    memory_values: list[float] = []
    scopes: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            timestamp = _first_present(row, ("timestamp_ms",))
            power = _first_present(row, ("power_watts", "gpu_power_w"))
            if timestamp is None or power is None:
                raise ValueError(f"{path}:{line_number}: timestamp or power is missing")
            samples.append((float(timestamp) / 1000.0, float(power)))
            memory = _first_present(row, ("memory_used_mib", "ram_mb"))
            if memory is not None:
                memory_values.append(float(memory))
            scopes.add(row.get("power_scope") or "unspecified_legacy")
    if len(samples) < 2:
        raise ValueError(f"{path}: at least two power samples are required")
    if len(scopes) != 1:
        raise ValueError(f"{path}: multiple power scopes found: {sorted(scopes)}")
    elapsed = samples[-1][0] - samples[0][0]
    if elapsed <= 0:
        raise ValueError(f"{path}: timestamps must span a positive duration")
    energy = integrate_power_samples(samples)
    return {
        "file": path.name,
        "sample_count": len(samples),
        "power_scope": next(iter(scopes)),
        "elapsed_seconds": elapsed,
        "mean_power_watts": energy / elapsed,
        "peak_memory_mib": max(memory_values) if memory_values else None,
        "energy_joules": energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize genuine timestamped hardware traces.")
    parser.add_argument("--root", type=Path, default=Path("reproducibility/power_traces"))
    parser.add_argument("--pattern", default="*.samples.csv")
    parser.add_argument("--output", type=Path, default=Path("reproducibility/power_traces/hardware_summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.root.glob(args.pattern)) if args.root.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no traces matched {args.root / args.pattern}")
    rows = [summarize_trace(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summarized {len(rows)} traces to {args.output}")


if __name__ == "__main__":
    main()
