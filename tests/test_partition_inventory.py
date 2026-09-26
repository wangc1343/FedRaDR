from scripts.validate_partitions import expected_files, validate_payload


def test_expected_partition_inventory_has_51_files() -> None:
    rows = expected_files()
    assert len(rows) == 51
    assert len(set((dataset, clients, partition, seed) for dataset, clients, partition, seed, _ in rows)) == 51


def test_partition_payload_validation() -> None:
    payload = {
        "client_indices": [[0, 1], [2]],
        "n_client": 2,
        "dataset": "cifar10",
        "partition_type": "D1",
        "hyper": {"alpha": 0.3, "n_class_per_client": None},
        "seed": 20,
    }
    assert validate_payload(payload, clients=2, seed=20, expected_hyper={"alpha": 0.3}) == []

