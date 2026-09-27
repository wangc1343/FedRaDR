from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from queue import Empty, Queue
from threading import Thread

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.energy import integrate_power_samples


POWER_PATTERN = re.compile(r"(?P<rail>[A-Za-z0-9_]+)\s+(?P<mw>\d+)mW")


def parse_power_mw(line: str, rail: str) -> float:
    readings = {match.group("rail"): float(match.group("mw")) for match in POWER_PATTERN.finditer(line)}
    if rail not in readings:
        raise ValueError(f"rail {rail!r} was not present in tegrastats output")
    return readings[rail]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command while retaining Jetson tegrastats power samples.")
    parser.add_argument("--rail", default="VDD_IN", help="Explicit board or GPU rail reported by tegrastats.")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--device", default="Jetson AGX Orin")
    parser.add_argument("--mode", default="MAXN")
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

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    tegrastats = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child = subprocess.Popen(command)
    rows: list[dict[str, float | str]] = []
    raw_lines: list[str] = []
    line_queue: Queue[str | None] = Queue()
    assert tegrastats.stdout is not None

    def read_lines() -> None:
        assert tegrastats.stdout is not None
        for line in tegrastats.stdout:
            line_queue.put(line)
        line_queue.put(None)

    reader = Thread(target=read_lines, daemon=True)
    reader.start()
    reader_finished = False
    try:
        while child.poll() is None:
            try:
                line = line_queue.get(timeout=0.05)
            except Empty:
                continue
            if line is None:
                reader_finished = True
                continue
            raw_lines.append(line.rstrip("\r\n"))
            timestamp = (time.perf_counter() - started) * 1000.0
            try:
                power_mw = parse_power_mw(line, args.rail)
            except ValueError:
                continue
            rows.append({"timestamp_ms": timestamp, "power_watts": power_mw / 1000.0, "power_scope": args.rail, "raw": line.strip()})
    finally:
        tegrastats.terminate()
        try:
            tegrastats.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegrastats.kill()
            tegrastats.wait(timeout=5)
        reader.join(timeout=2)
        drain_deadline = time.perf_counter() + 0.5
        while time.perf_counter() < drain_deadline:
            try:
                line = line_queue.get(timeout=0.05)
            except Empty:
                continue
            if line is None:
                reader_finished = True
                break
            raw_lines.append(line.rstrip("\r\n"))
        return_code = child.wait()

    finished_utc = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter() - started

    samples_path = args.output_prefix.with_suffix(".samples.csv")
    with samples_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_ms", "power_watts", "power_scope", "raw"])
        writer.writeheader()
        writer.writerows(rows)
    raw_path = args.output_prefix.with_suffix(".tegrastats.log")
    raw_path.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")
    samples = [(row["timestamp_ms"] / 1000.0, row["power_watts"]) for row in rows]
    summary = {
        "started_at_utc": started_utc,
        "finished_at_utc": finished_utc,
        "command": command,
        "return_code": return_code,
        "device": args.device,
        "mode": args.mode,
        "warmup_rounds_excluded": args.warmup_excluded,
        "sampling_interval_ms": args.interval_ms,
        "power_scope": args.rail,
        "sample_count": len(rows),
        "elapsed_seconds": elapsed,
        "measurement_status": "ok" if len(rows) >= 2 else "insufficient_samples",
        "integrated_energy_joules": integrate_power_samples(samples),
        "raw_samples": str(samples_path),
        "raw_tegrastats": str(raw_path),
    }
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must contain an object")
        summary["workload_metadata"] = metadata
    args.output_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
