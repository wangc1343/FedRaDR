import numpy as np

from fedradr.partitions import build_partition_payload, dirichlet_partition, pathological_partition


def assert_complete_partition(partition: list[list[int]], size: int) -> None:
    flattened = [index for client in partition for index in client]
    assert sorted(flattened) == list(range(size))
    assert len(flattened) == len(set(flattened))


def test_dirichlet_partition_is_deterministic_and_complete() -> None:
    labels = np.repeat(np.arange(4), 20)
    first = dirichlet_partition(labels, num_clients=8, alpha=0.3, seed=20)
    second = dirichlet_partition(labels, num_clients=8, alpha=0.3, seed=20)
    assert first == second
    assert_complete_partition(first, len(labels))


def test_pathological_partition_limits_local_classes() -> None:
    labels = np.repeat(np.arange(5), 30)
    partition = pathological_partition(labels, num_clients=10, classes_per_client=2, seed=21)
    assert_complete_partition(partition, len(labels))
    for indices in partition:
        assert len(set(labels[indices])) <= 2


def test_payload_records_index_basis() -> None:
    labels = np.repeat(np.arange(2), 10)
    payload = build_partition_payload(labels, "cifar10", "D1", 4, 22, alpha=0.6)
    assert payload["index_basis"] == "input_labels_array_order"
    assert payload["hyper"] == {"alpha": 0.6}
