import csv

import pytest

from calc_hardware_summary import summarize_trace


def test_summary_uses_timestamps_and_trapezoidal_integration(tmp_path) -> None:
    path = tmp_path / "power.samples.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["timestamp_ms", "power_watts", "power_scope", "memory_used_mib"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"timestamp_ms": 0, "power_watts": 10, "power_scope": "gpu_board", "memory_used_mib": 100},
                {"timestamp_ms": 500, "power_watts": 20, "power_scope": "gpu_board", "memory_used_mib": 125},
                {"timestamp_ms": 2000, "power_watts": 30, "power_scope": "gpu_board", "memory_used_mib": 120},
            ]
        )
    summary = summarize_trace(path)
    assert summary["elapsed_seconds"] == pytest.approx(2.0)
    assert summary["energy_joules"] == pytest.approx(45.0)
    assert summary["mean_power_watts"] == pytest.approx(22.5)
    assert summary["peak_memory_mib"] == pytest.approx(125.0)
