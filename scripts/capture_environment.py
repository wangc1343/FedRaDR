from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

import torch


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the experiment software and hardware environment.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except (ImportError, RuntimeError):
        torchvision_version = None

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "operating_system": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
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

