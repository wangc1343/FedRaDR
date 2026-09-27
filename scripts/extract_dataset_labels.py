"""Extract train labels in dataset-loader order for partition generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "tinyimagenet"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="torchvision root or Tiny-ImageNet root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.dataset == "tinyimagenet":
        train = args.root / "train"
        class_names = sorted(path.name for path in train.iterdir() if path.is_dir())
        class_to_id = {name: index for index, name in enumerate(class_names)}
        labels = []
        for class_name in class_names:
            image_dir = train / class_name / "images"
            labels.extend(class_to_id[class_name] for path in sorted(image_dir.iterdir()) if path.is_file())
    else:
        try:
            from torchvision import datasets
        except ImportError as error:
            raise SystemExit("torchvision is required; install requirements-fedradr.txt first") from error
        cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        dataset = cls(root=args.root, train=True, download=args.download)
        labels = list(dataset.targets)
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise SystemExit("no one-dimensional training labels were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, values)
    print(f"wrote {values.size} labels to {args.output} (sha256 is recorded by partition generation)")


if __name__ == "__main__":
    main()
