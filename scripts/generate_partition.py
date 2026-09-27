from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.partitions import build_partition_payload, write_partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one deterministic serialized partition.")
    parser.add_argument("--labels", type=Path, required=True, help="One-dimensional .npy label array.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--partition", choices=["D1", "D2", "P1", "P2"], required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=[20, 21, 22], required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--classes-per-client", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 1:
        parser.error("--labels must contain a one-dimensional array")
    payload = build_partition_payload(
        labels,
        args.dataset,
        args.partition,
        args.clients,
        args.seed,
        alpha=args.alpha,
        classes_per_client=args.classes_per_client,
    )
    digest = write_partition(payload, args.output)
    print(f"wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()

