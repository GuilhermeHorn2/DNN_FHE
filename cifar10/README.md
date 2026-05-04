# `cifar10` — CIFAR-10 driver on the modular FHE NN framework

Same network topology as the MNIST DiNN30 sub-project (`Linear → Sign →
Linear`), retargeted to CIFAR-10:

* **Input.** 32 × 32 × 3 RGB → flat **3072** entries (vs. 784 for MNIST).
* **Hidden.** 30 bipolar neurons.
* **Output.** 10 classes.
* **Crypto context.** `numSlots = 4096` (next power of two ≥ 3072) and
  `ringDim = 8192` (= 2 × numSlots). All other crypto parameters use the
  framework defaults from [`FHEParams`](../include/network/context.h).

For full framework documentation (sequential builder API, depth scheduler,
benchmark flags, etc.) see [`MNIST_30/README.md`](../MNIST_30/README.md).

---

## 1. Files in this folder

```
cifar10/
├── CMakeLists.txt              build script (mirrors MNIST_30/MNIST_100 layout)
├── README.md                   this file
├── main.cpp                    driver: bigger ringDim/numSlots, RGB image loader
├── cifar10_weights_W1.csv      hidden-layer weights  (3072 × 30)   ← provide
├── cifar10_weights_b1.csv      hidden-layer biases   (30)          ← provide
├── cifar10_weights_W2.csv      output-layer weights  (30 × 10)     ← provide
├── cifar10_weights_b2.csv      output-layer biases   (10)          ← provide
└── *.png / *.jpg               sample CIFAR images                 ← provide
```

The four `cifar10_weights_*.csv` files and the input images are **not
checked in** — drop your own next to this README. The driver expects them at
`../cifar10_weights_*.csv` (i.e. it is run from `cifar10/build/`).

CSV orientation: `LoadCsv2D` accepts both `out × in` and `in × out` and
transposes automatically, so either layout works.

---

## 2. Build & run

```bash
cd cifar10
cmake -B build -S .
cmake --build build -j4
cd build
./main ../cat.png
```

To enable timers / RSS reports, flip on any of the bench options:

```bash
cmake -B build -S . \
  -DBENCH_TOTAL=ON -DBENCH_INFERENCE=ON \
  -DBENCH_BOOTSTRAP=ON -DBENCH_LAYERS=ON -DBENCH_MEMORY=ON
cmake --build build -j4
```

See [`MNIST_30/README.md`](../MNIST_30/README.md#6-benchmark-flag-reference)
for what each flag prints.

---

## 3. Things to watch out for

* **Ring dimension blowup.** Going from `ringDim = 2048` (MNIST) to `8192`
  (CIFAR) makes every CKKS operation roughly **4× slower** and **3-4× more
  memory-hungry**. A single inference will likely require well over 1 GB of
  RSS and several minutes of wall-time. Use `-DBENCH_MEMORY=ON` to track it.
* **Image channels.** [`LoadImageBipolar`](../include/io/image.h) is called
  with `channels=3` here. The default (1) is what MNIST uses, so don't drop
  the third argument or the byte stream will be 1024 long instead of 3072.
* **Same depth budget as DiNN30.** Both layers are linear, both consume two
  rescales, so `Network::Compile` settles on `levelsComputation = 2` exactly
  like the MNIST sub-project. The ring dimension is the only crypto param
  that grew.
* **Plaintext reference.** `main.cpp` recomputes the CIFAR network in
  `double` after the FHE inference and prints `(matches)` / `(MISMATCH)`.
  Use it as a quick correctness check on any new weight matrix.
