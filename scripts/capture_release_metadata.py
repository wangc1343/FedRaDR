from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the exact repository version used for a release or run.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()

    remotes = {}
    for line in git(repo, "remote", "-v").splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        name, url, direction = parts
        remotes.setdefault(name, {})[direction.strip("()")]=url
    tags = git(repo, "tag", "--points-at", "HEAD").splitlines()
    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "commit": git(repo, "rev-parse", "HEAD"),
        "commit_short": git(repo, "rev-parse", "--short", "HEAD"),
        "tags_at_commit": tags,
        "dirty": bool(git(repo, "status", "--porcelain")),
        "remotes": remotes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
