import pytest

from scripts.measure_jetson_energy import parse_power_mw


def test_parses_named_tegrastats_rail() -> None:
    line = "RAM 100/1000MB VDD_IN 12450mW VDD_GPU 4310mW"
    assert parse_power_mw(line, "VDD_IN") == pytest.approx(12450.0)
    assert parse_power_mw(line, "VDD_GPU") == pytest.approx(4310.0)


def test_rejects_missing_rail() -> None:
    with pytest.raises(ValueError):
        parse_power_mw("VDD_IN 1000mW", "VDD_GPU")
