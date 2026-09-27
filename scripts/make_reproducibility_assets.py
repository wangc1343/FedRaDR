from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.reproducibility import make_client_schedule, save_json_with_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed client schedule for all compared methods.")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--clients-per-round", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("--rounds must be non-negative")
    schedule = make_client_schedule(args.clients, args.clients_per_round, args.rounds, args.seed)
    digest = save_json_with_hash(
        {
            "num_clients": args.clients,
            "clients_per_round": args.clients_per_round,
            "rounds": args.rounds,
            "seed": args.seed,
            "schedule": schedule,
        },
        args.output,
    )
    print(f"wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
