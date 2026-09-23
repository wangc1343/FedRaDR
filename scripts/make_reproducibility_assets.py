from __future__ import annotations

import argparse
from pathlib import Path

from fedradr.reproducibility import make_client_schedule, save_json_with_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed client schedule for all compared methods.")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--clients-per-round", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
