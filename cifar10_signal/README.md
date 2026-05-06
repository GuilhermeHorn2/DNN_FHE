# `cifar10` — CIFAR-10 driver on the MVB framework

MVB-based homomorphic inference for CIFAR-10 with the same DiNN30
topology used by `MNIST_30/`, retargeted to RGB images. For framework
internals, build-flag reference, and the batch accuracy harness, see the
[**root README**](../README.md).

---

## 1. Topology

| Layer       | Shape           | Notes                                                              |
| ----------- | --------------- | ------------------------------------------------------------------ |
| Input       | 3072 (32×32×3)  | bipolar `{-1, +1}` after per-byte threshold (loaded RGB, channels=3) |
| Linear 1    | 3072 → 30       | exact integer matvec + bias                                        |
| `Sign`      | 30 → 30         | `(x ≥ 0) ? +1 : -1` via MVB LUT                                    |
| Linear 2    | 30 → 10         | output scores (one per CIFAR-10 class)                             |

### Class index ↔ name (canonical CIFAR-10 order)

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| - | - | - | - | - | - | - | - | - | - |
| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |

The driver prints both index and name in its score dump and final summary.

### Crypto context overrides

CIFAR's flat input is 3072 entries, which doesn't fit in the 1024-slot
default. `main.cpp` overrides:

```cpp
FHEParams p;
p.numSlots = 4096;             // next power of two ≥ 3072
p.ringDim  = 1u << 13;         // 8192 (≥ 2 * numSlots)
```

Everything else uses the framework defaults from
[`FHEParams`](../include/network/context.h).

---

## 2. Files in this folder

```
cifar10/
├── CMakeLists.txt              build script (mirrors MNIST_30/MNIST_100 layout)
├── README.md                   you are here
├── main.cpp                    driver: bigger ringDim/numSlots, RGB image loader, class-name labels
├── cifar10_weights_W1.csv      hidden-layer weights  (3072 × 30)
├── cifar10_weights_b1.csv      hidden-layer biases   (30)
├── cifar10_weights_W2.csv      output-layer weights  (30 × 10)
├── cifar10_weights_b2.csv      output-layer biases   (10)
└── cifar10_test_images.zip     ~100 sample CIFAR-10 PNGs, named "<idx>_<class>.png"
```

CSV orientation: `LoadCsv2D` accepts both `out × in` and `in × out` and
transposes automatically, so either layout works.

### Test images

Unzip `cifar10_test_images.zip` next to the binary (or anywhere) before
running. Filenames follow the convention `<idx>_<classname>.png`
(e.g. `06_automobile.png`, `52_airplane.png`, `93_frog.png`). The driver
parses the `<classname>` part to extract a ground-truth label and prints
a `(CORRECT)` / `(wrong)` verdict alongside the prediction. Files that
don't match this pattern (e.g. your own `cat.png`) still work — the
ground-truth line is just skipped.

```bash
unzip cifar10_test_images.zip -d test_images
```

---

## 3. Build & run

```bash
cd cifar10
cmake -B build -S .
cmake --build build -j4
cd build
./main ../test_images/06_automobile.png
# or with a custom image:
./main ../cat.png
```

Sample output:

```
--- Final Class Scores ---
  [0] airplane    : -3
  [1] automobile  : 17
  [2] bird        : 2
  …
================================
 PREDICTED CLASS  : 1 (automobile)
 REFERENCE (plain): 1 (automobile)  (matches)
 GROUND TRUTH     : 1 (automobile)  (CORRECT)
================================
```

To enable timers / RSS reports:

```bash
cmake -B build -S . \
  -DBENCH_TOTAL=ON -DBENCH_INFERENCE=ON \
  -DBENCH_BOOTSTRAP=ON -DBENCH_LAYERS=ON -DBENCH_MEMORY=ON
cmake --build build -j4
```

See [root README §3](../README.md#3-common-build--run-idiom) for the full
flag reference.

---

## 4. Things to watch out for

* **Ring dimension blowup.** Going from `ringDim = 2048` (MNIST) to
  `8192` (CIFAR) makes every CKKS operation roughly **4× slower** and
  **3-4× more memory-hungry**. A single inference will likely require
  well over 1 GB of RSS and several minutes of wall-time. Use
  `-DBENCH_MEMORY=ON` to track it.
* **Image channels.**
  [`LoadImageBipolar`](../include/io/image.h) is called with
  `channels=3` here. The default (1) is what MNIST uses, so don't drop
  the third argument or the byte stream will be 1024 long instead of
  3072.
* **Same depth budget as DiNN30.** Both layers are linear, both consume
  two rescales, so `Network::Compile` settles on `levelsComputation = 2`
  exactly like `MNIST_30/`. The ring dimension is the only crypto param
  that grew.
* **Plaintext reference.** `main.cpp` recomputes the network in `double`
  after the FHE inference and prints `(matches)` / `(MISMATCH)`. Use it
  as a quick correctness check on any new weight matrix.
* **Batch harness works out of the box.** Pass `main` a directory shaped
  as `<root>/<label>/*.{png,jpg,jpeg}` (labels `0/` … `9/`) and it will
  use this project's CIFAR-aware `PixelLoader` (3-channel,
  `LoadImageBipolar`, `IN_DIM = 3072`) — no recompile or hand-editing
  needed.
