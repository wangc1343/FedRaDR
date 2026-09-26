from __future__ import annotations

from collections.abc import Sequence


def integrate_power_samples(samples: Sequence[tuple[float, float]]) -> float:
    """Integrate timestamped power samples in watts and return joules."""
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (time_a, power_a), (time_b, power_b) in zip(samples, samples[1:]):
        if time_b < time_a:
            raise ValueError("power sample timestamps must be nondecreasing")
        energy += 0.5 * (power_a + power_b) * (time_b - time_a)
    return energy

