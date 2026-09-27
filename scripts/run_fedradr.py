"""Run FedRaDR from a serialized FedDC-style data object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.models import make_model
from fedradr.objective import FedRaDRConfig
from fedradr.partitions import labels_digest
from fedradr.reproducibility import current_git_metadata, make_client_schedule, seed_everything, sha256_file
from utils_methods_FedRaDR import train_FedRaDR


DATASET_CLASS_COUNTS = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "TINYIMAGENET": 200,
    "TINY-IMAGENET": 200,
    "TINY_IMAGE_NET": 200,
}


def load_data(path: Path):
    """Load clnt_x/clnt_y/tst_x/tst_y/dataset from a compressed NPZ file."""
    archive = np.load(path, allow_pickle=True)
    dataset = str(archive["dataset"].item())
    client_x = list(archive["clnt_x"])
    client_y = list(archive["clnt_y"])
    data_obj = SimpleNamespace(
        dataset=dataset,
        n_client=len(client_x),
        clnt_x=client_x,
        clnt_y=client_y,
        tst_x=archive["tst_x"],
        tst_y=archive["tst_y"],
        partition_applied=False,
        partition_application="prepartitioned_npz",
    )
    for x_key, y_key in (("train_x", "train_y"), ("trn_x", "trn_y")):
        if x_key in archive and y_key in archive:
            data_obj.train_x = archive[x_key]
            data_obj.train_y = archive[y_key]
            break
    return data_obj


def apply_partition(data_obj: SimpleNamespace, payload: dict[str, object], *, require_application: bool) -> bool:
    """Materialize client arrays when the NPZ also contains global train arrays.

    A legacy FedDC NPZ already contains ``clnt_x``/``clnt_y`` and no global
    arrays. In that case the partition file can be provenance-checked, but its
    indices cannot be applied without changing the dataset export. The caller
    records this distinction in ``config.json`` instead of silently claiming
    that the file was used.
    """
    indices = payload.get("client_indices")
    if not isinstance(indices, list):
        raise ValueError("partition payload client_indices must be a list")
    if hasattr(data_obj, "train_x") and hasattr(data_obj, "train_y"):
        train_x = np.asarray(data_obj.train_x)
        train_y = np.asarray(data_obj.train_y)
        sample_count = int(payload.get("sample_count", len(train_y)))
        if len(train_x) != len(train_y) or len(train_y) != sample_count:
            raise ValueError("partition sample_count does not match global train arrays")
        expected_labels_hash = payload.get("labels_sha256")
        if expected_labels_hash and labels_digest(train_y) != expected_labels_hash:
            raise ValueError("partition labels_sha256 does not match the global train_y array")
        if len(indices) != data_obj.n_client:
            raise ValueError("partition client count does not match the NPZ")
        flattened = [int(value) for values in indices for value in values]
        if sorted(flattened) != list(range(sample_count)):
            raise ValueError("partition client_indices do not cover the global train array exactly once")
        data_obj.clnt_x = [train_x[np.asarray(values, dtype=np.int64)] for values in indices]
        data_obj.clnt_y = [train_y[np.asarray(values, dtype=np.int64)] for values in indices]
        data_obj.partition_applied = True
        data_obj.partition_application = "global_train_arrays_materialized"
        return True
    observed = sum(len(values) for values in data_obj.clnt_y)
    expected = payload.get("sample_count")
    if expected is not None and int(expected) != observed:
        raise ValueError(
            f"partition sample_count={expected} does not match prepartitioned NPZ sample count={observed}"
        )
    data_obj.partition_application = "prepartitioned_npz_provenance_only"
    if require_application:
        raise ValueError(
            "--require-partition-application needs train_x/train_y (or trn_x/trn_y) in the NPZ"
        )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--schedule-file", type=Path)
    parser.add_argument(
        "--require-partition-application",
        action="store_true",
        help="Fail unless the partition indices can be materialized from global train arrays.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenetv2"], default="resnet18")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--initial-model-seed",
        type=int,
        default=None,
        help="Optional separate initialization seed; defaults to --seed and is recorded explicitly.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate-decay", type=float, default=0.998)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--normalization", choices=["exact", "minibatch", "ema"], default="exact")
    parser.add_argument("--hfr-threshold", type=float, default=None)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--target-budget", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()

    data_obj = load_data(args.data)
    dataset_key = data_obj.dataset.upper().replace(" ", "")
    if dataset_key not in DATASET_CLASS_COUNTS:
        parser.error(f"unsupported dataset {data_obj.dataset!r}; pass a recognized CIFAR/Tiny-ImageNet name")
    num_classes = DATASET_CLASS_COUNTS[dataset_key]
    model_func = lambda: make_model(args.model, num_classes)
    schedule_payload_record = None
    if args.schedule_file:
        if not args.schedule_file.is_file():
            parser.error(f"schedule file does not exist: {args.schedule_file}")
        schedule_payload = json.loads(args.schedule_file.read_text(encoding="utf-8"))
        if isinstance(schedule_payload, dict):
            schedule_payload_record = schedule_payload
            schedule = schedule_payload.get("schedule")
            if schedule_payload.get("num_clients") not in {None, data_obj.n_client}:
                parser.error("--schedule-file client count does not match the loaded NPZ")
            if schedule_payload.get("seed") not in {None, args.seed}:
                parser.error("--schedule-file seed does not match --seed")
        else:
            schedule = schedule_payload
        if not isinstance(schedule, list):
            parser.error("--schedule-file must contain a schedule list or an object with a schedule field")
        if len(schedule) != args.rounds:
            parser.error("--schedule-file round count does not match --rounds")
        def valid_round(round_clients: object) -> bool:
            if not isinstance(round_clients, list) or len(round_clients) != args.clients_per_round:
                return False
            if any(not isinstance(client, int) or isinstance(client, bool) or not 0 <= client < data_obj.n_client for client in round_clients):
                return False
            return len(set(round_clients)) == len(round_clients)

        if any(not valid_round(round_clients) for round_clients in schedule):
            parser.error("--schedule-file contains invalid client IDs or participant counts")
    else:
        schedule = make_client_schedule(
            data_obj.n_client, args.clients_per_round, args.rounds, args.seed
        )

    partition_metadata = None
    if args.partition_file:
        if not args.partition_file.is_file():
            parser.error(f"partition file does not exist: {args.partition_file}")
        payload = pickle.loads(args.partition_file.read_bytes())
        if not isinstance(payload, dict):
            parser.error("partition file root must be a dictionary")
        if int(payload.get("n_client", -1)) != data_obj.n_client:
            parser.error("partition n_client does not match the loaded NPZ")
        if int(payload.get("seed", -1)) != args.seed:
            parser.error("partition seed does not match --seed")
        payload_dataset = str(payload.get("dataset", "")).replace("-", "").replace("_", "").lower()
        data_dataset = data_obj.dataset.replace("-", "").replace("_", "").replace(" ", "").lower()
        if payload_dataset and payload_dataset != data_dataset:
            parser.error("partition dataset does not match the loaded NPZ")
        if payload.get("index_basis") != "input_labels_array_order":
            parser.error("partition index_basis is not supported by this runner")
        try:
            partition_applied = apply_partition(
                data_obj, payload, require_application=args.require_partition_application
            )
        except (KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        partition_metadata = {
            "path": str(args.partition_file),
            "sha256": sha256_file(args.partition_file),
            "dataset": payload.get("dataset"),
            "partition_type": payload.get("partition_type"),
            "index_basis": payload.get("index_basis"),
            "sample_count": payload.get("sample_count"),
            "labels_sha256": payload.get("labels_sha256"),
            "applied": partition_applied,
            "application": data_obj.partition_application,
        }
    else:
        partition_applied = False
    init_seed = args.initial_model_seed if args.initial_model_seed is not None else args.seed
    seed_everything(init_seed)
    initial_model = model_func()
    args.output.mkdir(parents=True, exist_ok=True)
    schedule_file_sha256 = sha256_file(args.schedule_file) if args.schedule_file else None
    schedule_record = {
        "schedule": schedule,
        "source": str(args.schedule_file) if args.schedule_file else "generated",
        "source_sha256": schedule_file_sha256,
        "historical": (
            schedule_payload_record.get("historical", False)
            if schedule_payload_record is not None
            else False
        ),
        "schedule_family": schedule_payload_record.get("schedule_family") if schedule_payload_record else "independent_round_sampling",
        "family_id": schedule_payload_record.get("family_id") if schedule_payload_record else None,
        "seed": args.seed,
        "num_clients": data_obj.n_client,
        "clients_per_round": args.clients_per_round,
        "rounds": args.rounds,
    }
    schedule_record["canonical_sha256"] = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (args.output / "client_schedule.json").write_text(
        json.dumps(schedule_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train_FedRaDR(
        data_obj,
        model_func,
        initial_model,
        client_schedule=schedule,
        output_dir=args.output,
        config=FedRaDRConfig(
            lambda_max=args.lambda_max,
            temperature=args.temperature,
            normalization=args.normalization,
            hfr_threshold=args.hfr_threshold,
        ),
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        run_metadata={
            "origin": "generated",
            "historical": False,
            "code": current_git_metadata(Path(__file__).resolve().parents[1]),
            "data_npz": {"path": str(args.data), "sha256": sha256_file(args.data)},
            "partition": partition_metadata,
            "schedule": schedule_record,
            "dataset_class_count": num_classes,
            "model": args.model,
            "initial_model_seed": init_seed,
            "experiment_seed": args.seed,
            "precision": str(torch.get_default_dtype()),
            "target": {
                "accuracy": args.target_accuracy,
                "budget_rounds": args.target_budget,
            },
        },
    )


if __name__ == "__main__":
    main()
