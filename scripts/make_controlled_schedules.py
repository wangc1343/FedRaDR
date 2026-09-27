from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedradr.reproducibility import make_nested_client_schedule_family, save_json_with_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nested schedules for a fixed client pool and multiple participation rates."
    )
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--participants", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", required=True, help="For example c100")
    args = parser.parse_args()

    schedules, permutation_hashes = make_nested_client_schedule_family(
        args.clients, args.participants, args.rounds, args.seed
    )
    family_id = f"{args.name_prefix}-r{args.rounds}-seed{args.seed}"
    sidecar = args.output_dir.parent / "schedule_families" / f"{family_id}.permutation_hashes.json"
    base_hash = save_json_with_hash(permutation_hashes, sidecar)
    for count, schedule in schedules.items():
        percentage = 100 * count / args.clients
        if not percentage.is_integer():
            raise SystemExit("participant count must produce an integer percentage in the file name")
        filename = f"{args.name_prefix}-p{int(percentage)}-r{args.rounds}-seed{args.seed}.json"
        payload = {
            "num_clients": args.clients,
            "clients_per_round": count,
            "rounds": args.rounds,
            "seed": args.seed,
            "schedule_family": "nested_permutation_prefix",
            "family_id": family_id,
            "base_permutation_hashes_sha256": base_hash,
            "base_permutation_count": len(permutation_hashes),
            "schedule": schedule,
        }
        path = args.output_dir / filename
        digest = save_json_with_hash(payload, path)
        print(f"wrote {path} sha256={digest}")


if __name__ == "__main__":
    main()
