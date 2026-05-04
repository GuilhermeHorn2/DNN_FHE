"""Export MNIST / CIFAR-10 / CIFAR-100 test images as PNGs.

Output layout (under ``--output-root``, default ``/repo/test_data`` when
the container's repo-root mount exists, else ``../test_data`` relative to
``new_models/``):

    test_data/
      mnist/
        50/0/00000.png 0/00010.png ... 9/...
        100/...
        500/...
      cifar10/
        50/0/...png ... 9/...
        100/...
        500/...
      cifar100/
        50/<fine_label>/...png   (only ~50 of 100 classes present)
        100/...
        500/...

This is the per-class directory layout that ``accuracy.cpp`` expects:

    test_root/
      0/some_zero.png
      0/another_zero.png
      1/...

so a typical run is:

    cd cifar10/build
    ./accuracy ../../test_data/cifar10/100

By default, MNIST PNGs are saved as 28x28 grayscale ('L' mode) and CIFAR
PNGs as 32x32 RGB, exactly what ``stbi_load(..., channels=1|3)`` returns.
The C++ side then thresholds each byte at 127 to {-1, +1}.

Re-running is destructive per ``<dataset>/<count>/`` directory: existing
groups are wiped and rewritten, so counts stay deterministic.

Usage:
    python export_test_images.py
    python export_test_images.py --counts 50,100,500,1000
    python export_test_images.py --dataset cifar100 --counts 200
    python export_test_images.py --output-root /tmp/imgs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from torchvision import datasets

DEFAULT_COUNTS = (50, 100, 500)
DATASETS = ("mnist", "cifar10", "cifar100")


def load_test_split(name: str, data_dir: Path):
    """Return a torchvision test dataset (PIL images, no transform)."""
    sub = data_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        return datasets.MNIST(root=str(sub), train=False, download=True)
    if name == "cifar10":
        return datasets.CIFAR10(root=str(sub), train=False, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root=str(sub), train=False, download=True)
    raise ValueError(f"unknown dataset {name!r}")


def export_dataset(name: str, data_dir: Path, out_root: Path, counts) -> None:
    ds = load_test_split(name, data_dir)
    n_total = len(ds)
    max_count = max(counts)
    if max_count > n_total:
        print(f"[{name}] WARN: requested {max_count} > test set size {n_total}; clamping.")
        max_count = n_total

    # Snapshot first max_count test items so every count uses the same prefix.
    cached: list[tuple] = []
    for i in range(max_count):
        img, label = ds[i]
        cached.append((img, int(label)))

    for n in counts:
        n = min(n, max_count)
        target = out_root / name / str(n)
        if target.exists():
            shutil.rmtree(target)
        seen_labels: set[int] = set()
        for i, (img, label) in enumerate(cached[:n]):
            cls_dir = target / str(label)
            cls_dir.mkdir(parents=True, exist_ok=True)
            img.save(cls_dir / f"{i:05d}.png")
            seen_labels.add(label)
        print(f"[{name}] {target}  ({n} images, {len(seen_labels)} classes)")


def resolve_default_output_root() -> Path:
    """Prefer the container's repo-root mount when present; fall back to ../test_data."""
    in_container = Path("/repo/test_data")
    if in_container.parent.is_dir():  # /repo exists -> we're in the configured container
        return in_container
    return Path("../test_data").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default=None,
        help="root for emitted PNG trees "
             "(default: /repo/test_data inside the container, else ../test_data)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="where to keep torchvision-downloaded raw datasets (default: ./data)",
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), action="append",
        help="dataset to export (repeatable). default: all",
    )
    parser.add_argument(
        "--counts", default=",".join(str(c) for c in DEFAULT_COUNTS),
        help="comma-separated image counts per dataset (default: 50,100,500)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve() if args.output_root else resolve_default_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    counts = tuple(sorted({int(c) for c in args.counts.split(",") if c.strip()}))
    if not counts:
        parser.error("--counts must contain at least one positive integer")

    selected = args.dataset or list(DATASETS)
    data_dir = Path(args.data_dir).resolve()
    print(f"Output root: {out_root}")
    print(f"Counts:      {counts}")
    print(f"Datasets:    {selected}")

    for name in selected:
        export_dataset(name, data_dir, out_root, counts)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
