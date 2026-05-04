"""Download MNIST, CIFAR-10, and CIFAR-100 into ``data/<dataset>/``.

Layout produced (relative to ``--data-dir``, default ``./data``):

    data/
      mnist/      <- MNIST/raw, MNIST/processed (torchvision layout)
      cifar10/    <- cifar-10-batches-py/
      cifar100/   <- cifar-100-python/

Re-running is safe; torchvision will skip files it already has.

Usage:
    python download_datasets.py                    # default ./data
    python download_datasets.py --data-dir /tmp/d  # custom root
    python download_datasets.py --only cifar100    # one dataset only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchvision import datasets

DATASETS = {
    "mnist":    ("mnist",    datasets.MNIST),
    "cifar10":  ("cifar10",  datasets.CIFAR10),
    "cifar100": ("cifar100", datasets.CIFAR100),
}


def download_one(name: str, root: Path) -> None:
    subdir, cls = DATASETS[name]
    target = root / subdir
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] -> {target}")
    cls(root=str(target), train=True,  download=True)
    cls(root=str(target), train=False, download=True)
    print(f"[{name}] done.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data",
        help="root folder for all datasets (default: ./data)",
    )
    parser.add_argument(
        "--only", choices=sorted(DATASETS.keys()), action="append",
        help="download only this dataset (repeatable). default: all",
    )
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}")

    selected = args.only or list(DATASETS.keys())
    for name in selected:
        download_one(name, root)

    print("All requested datasets are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
