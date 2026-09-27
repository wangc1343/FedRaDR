from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

try:
    import torch
except ImportError:  # Metadata capture should still work before ML dependencies are installed.
    torch = None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the experiment software and hardware environment.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()

    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except (ImportError, RuntimeError):
        torchvision_version = None

    gpu_name = torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None
    git_status = command_output(["git", "-C", str(repo), "status", "--porcelain"])
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "git_commit": command_output(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty": None if git_status is None else bool(git_status),
        "operating_system": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__ if torch is not None else None,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda if torch is not None else None,
        "cudnn": torch.backends.cudnn.version() if torch is not None else None,
        "gpu_name": gpu_name,
        "driver_version": command_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"
        ]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
