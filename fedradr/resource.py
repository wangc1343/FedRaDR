from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class ResourceMeasurement:
    elapsed_seconds: float
    peak_cuda_bytes: int | None
    energy_joules: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@contextmanager
def measure_resources(device: torch.device) -> Iterator[dict[str, ResourceMeasurement]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result: dict[str, ResourceMeasurement] = {}
    try:
        yield result
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = None
        result["measurement"] = ResourceMeasurement(
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_bytes=peak,
            # Energy requires a registered hardware counter (e.g. NVML/tegrastats).
            energy_joules=None,
        )
