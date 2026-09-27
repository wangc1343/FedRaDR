from fedradr.reproducibility import make_nested_client_schedules


def test_nested_schedules_share_round_level_client_order() -> None:
    schedules = make_nested_client_schedules(100, [10, 5, 2], 3, 20)
    for round_index in range(3):
        p10 = set(schedules[10][round_index])
        p5 = set(schedules[5][round_index])
        p2 = set(schedules[2][round_index])
        assert p5 <= p10
        assert p2 <= p5
