from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.energy import integrate_power_samples


def nvidia_query(fields: str, gpu_index: int) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--id={gpu_index}",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def sample_gpu(gpu_index: int, elapsed_seconds: float) -> dict[str, float | str]:
    fields = "power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu"
    values = [value.strip() for value in nvidia_query(fields, gpu_index).splitlines()[0].split(",")]
    if len(values) != 5:
        raise ValueError(f"unexpected nvidia-smi sample: {values!r}")
    power, gpu_util, memory_util, memory_used, temperature = map(float, values)
    return {
        "timestamp_ms": elapsed_seconds * 1000.0,
        "power_watts": power,
        "power_scope": "gpu_board",
        "gpu_util_percent": gpu_util,
        "memory_util_percent": memory_util,
        "memory_used_mib": memory_used,
        "temperature_c": temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command while sampling NVIDIA GPU power.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--warmup-excluded", type=int, default=1)
    parser.add_argument("--metadata-json", type=Path, help="Optional workload metadata JSON merged into the summary.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.warmup_excluded < 0:
        parser.error("--warmup-excluded must be nonnegative")

    identity = nvidia_query("name,driver_version", args.gpu_index)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    child = subprocess.Popen(command)
    raw_samples: list[dict[str, float | str]] = []
    query_errors = 0
    while child.poll() is None:
        timestamp = time.perf_counter() - started
        try:
            raw_samples.append(sample_gpu(args.gpu_index, timestamp))
        except (OSError, subprocess.CalledProcessError, ValueError):
            query_errors += 1
        time.sleep(args.interval_ms / 1000.0)
    return_code = child.wait()
    elapsed = time.perf_counter() - started
    try:
        raw_samples.append(sample_gpu(args.gpu_index, elapsed))
    except (OSError, subprocess.CalledProcessError, ValueError):
        query_errors += 1

    csv_path = args.output_prefix.with_suffix(".samples.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "timestamp_ms",
            "power_watts",
            "power_scope",
            "gpu_util_percent",
            "memory_util_percent",
            "memory_used_mib",
            "temperature_c",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_samples)

    integration_samples = [
        (float(row["timestamp_ms"]) / 1000.0, float(row["power_watts"]))
        for row in raw_samples
    ]

    summary = {
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "return_code": return_code,
        "gpu_index": args.gpu_index,
        "gpu_identity_and_driver": identity,
        "sampling_interval_ms": args.interval_ms,
        "warmup_runs_excluded": args.warmup_excluded,
        "elapsed_seconds": elapsed,
        "sample_count": len(raw_samples),
        "query_error_count": query_errors,
        "integrated_energy_joules": integrate_power_samples(integration_samples),
        "raw_samples": str(csv_path),
    }
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must contain an object")
        summary["workload_metadata"] = metadata
    json_path = args.output_prefix.with_suffix(".summary.json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
