from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fedradr.partitions import build_partition_payload, write_partition
from fedradr.reproducibility import make_nested_client_schedule_family
from scripts.build_reproducibility_bundle import (
    LOG_FIELDS,
    protocol_record,
    validate_log,
)


def test_nested_schedule_family_has_shared_round_hashes() -> None:
    schedules, hashes = make_nested_client_schedule_family(20, [2, 5, 10], 4, 20)
    assert len(hashes) == 4
    for round_index in range(4):
        assert set(schedules[2][round_index]) <= set(schedules[5][round_index])
        assert set(schedules[5][round_index]) <= set(schedules[10][round_index])


def test_partition_payload_hash_and_full_coverage(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(4), 10)
    payload = build_partition_payload(labels, "cifar10", "D1", 5, 20, alpha=0.3)
    path = tmp_path / "partition.pkl"
    first = write_partition(payload, path)
    second = write_partition(payload, tmp_path / "partition2.pkl")
    assert first == second
    assert payload["sample_count"] == len(labels)
    assert len(payload["labels_sha256"]) == 64
    flattened = sorted(index for client in payload["client_indices"] for index in client)
    assert flattened == list(range(len(labels)))


def test_log_validation_rejects_noncontiguous_rounds(tmp_path: Path) -> None:
    record = {
        field: (0 if field != "cumulative_serialized_bytes" else 0)
        for field in LOG_FIELDS
    }
    record.update({"round": 0, "test_accuracy": 0.5, "hfr_tau": None})
    second = dict(record)
    second.update({"round": 2, "test_accuracy": 0.6})
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    rows, errors = validate_log(path)
    assert len(rows) == 2
    assert any("contiguous" in error for error in errors)


def test_protocol_records_pdf_hash_when_present(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf marker")
    record = protocol_record(source.as_posix())
    assert record["source_document"]["sha256"]
    assert record["artifact_policy"]["generated_power_traces_allowed"] is False
