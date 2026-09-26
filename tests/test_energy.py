import pytest

from fedradr.energy import integrate_power_samples


def test_integrates_constant_power() -> None:
    assert integrate_power_samples([(0.0, 100.0), (2.0, 100.0)]) == pytest.approx(200.0)


def test_rejects_decreasing_timestamps() -> None:
    with pytest.raises(ValueError):
        integrate_power_samples([(1.0, 100.0), (0.0, 100.0)])

