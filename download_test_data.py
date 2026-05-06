#!/usr/bin/env python3
"""Download MNIST + CIFAR-10 test data and lay it out for the batch harness.

Each MVB sub-project's ``main`` is dual-mode: pass an image path for one-shot
inference, or pass a directory shaped like::

    <root>/0/*.png
    <root>/1/*.png
    ...
    <root>/9/*.png

to dispatch into the shared accuracy harness (see ``src/io/accuracy.cpp`` and
the root ``README.md`` §4). This script fetches the official test splits,
samples a configurable number of images per dataset (default: 50, balanced
across the 10 classes so every label directory is non-empty), and writes
them as PNGs in the expected layout::

    data/mnist/<label>/*.png        # 28x28 grayscale  (784 = 28*28*1)
    data/cifar10/<label>/*.png      # 32x32 RGB        (3072 = 32*32*3)

After running this you can drive the harness with, e.g.::

    cd MNIST_signal
    cmake -B build -S .
    cmake --build build -j4
    ./build/main ../data/mnist            # batch mode
    ./build/main ../img_1.jpg             # single-image mode (still works)

Dependencies: ``numpy``, ``Pillow`` (see ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import struct
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "images": "t10k-images-idx3-ubyte.gz",
    "labels": "t10k-labels-idx1-ubyte.gz",
}
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

NUM_CLASSES = 10


def _download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` (skip if it already exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    return dest


def _load_mnist_test(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images[N,28,28] uint8, labels[N] uint8)`` for the test split."""
    img_gz = _download(
        f"{MNIST_BASE}/{MNIST_FILES['images']}",
        cache_dir / MNIST_FILES["images"],
    )
    lbl_gz = _download(
        f"{MNIST_BASE}/{MNIST_FILES['labels']}",
        cache_dir / MNIST_FILES["labels"],
    )

    with gzip.open(img_gz, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic: {magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)

    with gzip.open(lbl_gz, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def _load_cifar10_test(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images[N,32,32,3] uint8, labels[N] uint8)`` for the test split."""
    archive = _download(CIFAR10_URL, cache_dir / "cifar-10-binary.tar.gz")

    with tarfile.open(archive, "r:gz") as tar:
        try:
            member = next(
                m for m in tar.getmembers() if m.name.endswith("test_batch.bin")
            )
        except StopIteration as e:
            raise RuntimeError("cifar-10 archive missing test_batch.bin") from e
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError("could not open test_batch.bin from archive")
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    record = 1 + 32 * 32 * 3
    raw = raw.reshape(-1, record)
    labels = raw[:, 0].copy()
    # Stored as planar [R-plane | G-plane | B-plane], row-major within each
    # plane. Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3) for PIL.
    pixels = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return pixels, labels


def _sample_indices(
    labels: np.ndarray,
    total: int,
    rng: random.Random,
    *,
    balanced: bool,
) -> list[int]:
    """Pick ``total`` indices into ``labels``.

    ``balanced=True`` distributes picks evenly across the 10 classes (with
    the remainder spread over the lowest-numbered classes); ``balanced=False``
    samples uniformly at random over the whole pool.
    """
    if total <= 0:
        return []

    if not balanced:
        if total > len(labels):
            raise RuntimeError(
                f"requested {total} samples but only {len(labels)} available"
            )
        return rng.sample(range(len(labels)), total)

    by_class: dict[int, list[int]] = {c: [] for c in range(NUM_CLASSES)}
    for idx, lbl in enumerate(labels.tolist()):
        if lbl in by_class:
            by_class[lbl].append(idx)

    base, extra = divmod(total, NUM_CLASSES)
    picked: list[int] = []
    for c in range(NUM_CLASSES):
        wanted = base + (1 if c < extra else 0)
        pool = by_class[c]
        if wanted > len(pool):
            raise RuntimeError(
                f"class {c} has only {len(pool)} examples, asked for {wanted}"
            )
        picked.extend(rng.sample(pool, wanted))
    return picked


def _save_dataset(
    name: str,
    images: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    out_root: Path,
    *,
    grayscale: bool,
) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    counts = [0] * NUM_CLASSES
    mode = "L" if grayscale else "RGB"
    for idx in indices:
        label = int(labels[idx])
        cls_dir = out_root / str(label)
        cls_dir.mkdir(parents=True, exist_ok=True)
        path = cls_dir / f"{name}_{idx:05d}.png"
        Image.fromarray(images[idx], mode=mode).save(path)
        counts[label] += 1

    print(f"  wrote {sum(counts)} {name} images to {out_root}/")
    for c, n in enumerate(counts):
        print(f"    class {c}: {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--count",
        type=int,
        default=50,
        help="images per dataset (default: 50)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="output root directory (default: ./data)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("data/_downloads"),
        help="raw download cache (default: ./data/_downloads)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="RNG seed (default: 0)"
    )
    ap.add_argument(
        "--only",
        choices=["mnist", "cifar10"],
        default=None,
        help="restrict to a single dataset (default: both)",
    )
    ap.add_argument(
        "--unbalanced",
        action="store_true",
        help="sample uniformly at random instead of balancing across classes",
    )
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    balanced = not args.unbalanced

    if args.only in (None, "mnist"):
        print("MNIST:")
        images, labels = _load_mnist_test(args.cache)
        idx = _sample_indices(labels, args.count, rng, balanced=balanced)
        _save_dataset(
            "mnist", images, labels, idx, args.out / "mnist", grayscale=True
        )

    if args.only in (None, "cifar10"):
        print("CIFAR-10:")
        images, labels = _load_cifar10_test(args.cache)
        idx = _sample_indices(labels, args.count, rng, balanced=balanced)
        _save_dataset(
            "cifar10", images, labels, idx, args.out / "cifar10", grayscale=False
        )

    print("\nDone. Run the accuracy harness with, e.g.:")
    print(f"  ./build/accuracy {args.out}/mnist <weights_dir>")
    print(f"  ./build/accuracy {args.out}/cifar10 <weights_dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
