# DNN_FHE — Homomorphic Neural-Network Inference on OpenFHE

A collection of small, self-contained projects that run quantized
neural-network inference **fully homomorphically** on top of
[OpenFHE](https://github.com/openfheorg/openfhe-development)'s CKKS scheme.

The repo explores two different ways of evaluating non-linear activations
under FHE, on the same MNIST and CIFAR-10 topologies:

| Approach            | Activation              | Setup cost  | Per-inference   | Accuracy           | Folders                                                |
| ------------------- | ----------------------- | ----------- | --------------- | ------------------ | ------------------------------------------------------ |
| **MVB**             | Exact `Sign` via LUT    | High (FBT)  | High (refresh)  | Exact (bipolar)    | `MNIST_30/`, `MNIST_100/`, `cifar10/`                  |
| **Polynomial**      | Chebyshev approx `tanh` | Low         | Lower memory    | Approximate        | `poly_MNIST_30/`, `poly_MNIST_100/`, `poly_cifar10/`   |

* The **MVB** projects share a small reusable framework
  (`include/` + `src/` → `libfhednn.a`) that exposes a PyTorch-style
  sequential builder. They use multi-value bootstrap via the schemelet
  RLWE↔CKKS refresh path to evaluate exact look-up-table activations.
* The **polynomial** projects are standalone single-file implementations
  (`main.cpp` only) that talk directly to OpenFHE and replace the LUT with a
  degree-7 Chebyshev approximation of `tanh`. They are smaller, faster to
  set up, and lighter on RAM, but lose accuracy near the activation boundary.

---

## 1. Repository layout

```
DNN_FHE/
├── include/                           public headers for the MVB framework
│   ├── bench/bench.h                  scoped timers + RSS reporter (compile-time gated)
│   ├── io/{csv,image}.h               CSV / PNG-JPG loaders
│   ├── network/{activation,context,
│   │            layer,network}.h      builder, layers, FHEContext, activations
│   └── stb_image.h                    vendored image decoder
│
├── src/                               framework implementation (compiled into libfhednn.a)
│   ├── bench/bench.cpp
│   ├── io/{csv,image}.cpp
│   └── network/{activation,context,layer,network}.cpp
│
├── accuracy.cpp                       optional batch-accuracy harness (see §4)
│
├── MNIST_30/                          MVB · 784 → 30 → 10 (DiNN30)
├── MNIST_100/                         MVB · 784 → 100 → 10 (DiNN100)
├── cifar10/                           MVB · 3072 → 30 → 10 (CIFAR-10)
│
├── poly_MNIST_30/                     polynomial · 784 → 30 → 10
├── poly_MNIST_100/                    polynomial · 784 → 100 → 10
└── poly_cifar10/                      polynomial · 3072 → 30 → 10
```

Each sub-project owns its own `main.cpp`, `CMakeLists.txt`, weights, and
sample images. The MVB sub-projects link against the shared library compiled
from `../src`; the polynomial ones are independent translation units.

---

## 2. Prerequisites

* **OpenFHE** (development branch) installed system-wide so
  `find_package(OpenFHE REQUIRED)` resolves. The MVB projects additionally
  need the `schemelet` headers (RLWE↔CKKS refresh / `EvalFBT*`) bundled with
  recent OpenFHE.
* **CMake ≥ 3.10**, a **C++17** compiler, and the usual GMP / NTL stack
  pulled in by OpenFHE.
* Linux/WSL recommended (memory benchmarks read `/proc/self/status`).

---

## 3. Common build & run idiom

Every sub-project produces an executable called `main` that takes one image
path as its only argument:

```bash
cd <subproject>                    # e.g. cd MNIST_30
cmake -B build -S .
cmake --build build -j4
./build/main ../<subproject>/img_1.jpg     # or ./build/main ../cat.png  for cifar10
```

Each sub-project's README documents which weights and which sample images
it expects.

### MVB-only build options (CMake cache flags)

Available in `MNIST_30/`, `MNIST_100/`, `cifar10/` only — they're consumed
by `bench/bench.h` and `accuracy.cpp`:

| Flag                | Effect                                                              |
| ------------------- | ------------------------------------------------------------------- |
| `-DBENCH_TOTAL=ON`  | Wraps `main()` with a single end-to-end timer                       |
| `-DBENCH_INFERENCE` | Times `Network::Run` (per-query, excludes `Compile()` setup)        |
| `-DBENCH_BOOTSTRAP` | Times every `ActivationLayer::Apply` (the MVB refresh path)         |
| `-DBENCH_LAYERS`    | Times every individual layer (`Linear`, `Activation`, `DummyMult`)  |
| `-DBENCH_MEMORY`    | Adds `(peak RSS …)` to every timer line + `[BENCH][MEM]` checkpoints |
| `-DBUILD_ACCURACY`  | Builds the `accuracy` target alongside `main` (see §4)              |

When a flag is `OFF`, its macros expand to `((void)0)` — zero runtime cost.

```bash
cmake -B build -S . \
  -DBENCH_TOTAL=ON -DBENCH_INFERENCE=ON \
  -DBENCH_BOOTSTRAP=ON -DBENCH_LAYERS=ON -DBENCH_MEMORY=ON \
  -DBUILD_ACCURACY=ON
cmake --build build -j4
```

Sample output with all flags on:

```
[BENCH][MEM] startup                          RSS=6.38 MB    peak=6.38 MB
[BENCH][MEM] after-compile                    RSS=299.29 MB  peak=299.29 MB
[BENCH] InputEncoder                  67884.312 ms   (peak RSS 503.70 MB)
[BENCH] Linear                        48807.178 ms
[BENCH] Activation                    70286.521 ms
[BENCH] Bootstrap (refresh+MVB)       70286.172 ms
[BENCH] Linear                         5438.801 ms
[BENCH] OutputDecoder                  2065.538 ms
[BENCH] Inference (Run)              194483.505 ms
[BENCH] Total program                195729.122 ms   (peak RSS 515.35 MB)
[BENCH][MEM] end                              RSS=369.66 MB  peak=515.35 MB
```

---

## 4. `accuracy.cpp` — batch accuracy harness

`accuracy.cpp` (repo root) is a shared MVB-framework driver that builds the
network **once** and then iterates over a labeled directory tree, reporting
per-image predictions, an overall accuracy, and a 10×10 confusion matrix.

It is **not built by default**: each MVB sub-project's `CMakeLists.txt`
gates it behind `-DBUILD_ACCURACY=ON`, in which case the `accuracy` target
appears next to `main` in `build/`.

### Expected directory layout

One sub-directory per label, each named with the integer class:

```
test_root/
  0/some_zero.png
  0/another_zero.png
  1/...
  ...
  9/...
```

`.png`, `.jpg`, and `.jpeg` files are picked up; everything else is
skipped.

### Build & run

```bash
cd MNIST_30                                # any MVB sub-project works
cmake -B build -S . -DBUILD_ACCURACY=ON
cmake --build build -j4
cd build
./accuracy <test_root>                     # weights resolved from ../
./accuracy <test_root> <weights_dir>       # explicit weights directory
```

### Output

```
[   1] some_zero.png  -> pred=0  truth=0  OK
[   2] another_zero.png -> pred=8  truth=0  MISS
…
=== Accuracy ===
Correct: 947 / 1000  (94.70%)

Confusion matrix (row=truth, col=pred):
         0    1    2    3    4    5    6    7    8    9
  0 :   97    0    0    0    0    1    1    0    1    0
  1 :    0   99    0    1    0    0    0    0    0    0
  …
```

### Caveats

* **Wired for DiNN30.** As shipped, `accuracy.cpp` hard-codes
  `IN_DIM = 784`, `HID_DIM = 30`, and weight filenames `dinn30_*.csv`.
  Building it from `MNIST_100/` or `cifar10/` will compile, but will then
  read the wrong weights. To run it for DiNN100 or CIFAR-10, edit the
  three `static constexpr int` lines plus the weight paths near the top of
  `accuracy.cpp`. (The framework itself is generic — it's the harness's
  defaults that are MNIST-30-shaped.)
* **Cost is amortized.** `Network::Compile` is called exactly once. The
  per-image cost is therefore close to the `BENCH_INFERENCE` number you
  get from `main`, scaled by image count.

---

## 5. Where to go next

| If you want to…                                              | Read…                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------ |
| Run a single MNIST DiNN30 inference                          | [`MNIST_30/README.md`](MNIST_30/README.md)                         |
| Run a wider 100-neuron MNIST network                         | [`MNIST_100/README.md`](MNIST_100/README.md)                       |
| Run CIFAR-10 inference (bigger ring dim, slower)             | [`cifar10/README.md`](cifar10/README.md)                           |
| See the same networks evaluated with poly-approx `tanh`      | [`poly_MNIST_30/README.md`](poly_MNIST_30/README.md), [`poly_MNIST_100/`](poly_MNIST_100/README.md), [`poly_cifar10/`](poly_cifar10/README.md) |
| Understand the framework internals (`Compile`, depth budget) | §6 below                                                           |

---

## 6. Framework internals (MVB projects)

### Pipeline at a glance

```
raw input
   └─► InputEncoder  : add input shift, RLWE encrypt, RLWE→CKKS,
   │                   identity-MVB into slot space, undo input shift
   ├─► [Linear …]    : slot-space matvec + bias (consumes 2 levels each)
   ├─► [DummyMult]   : (auto-inserted) padding to align block depth
   ├─► Activation    : add LUT pre-shift, HomDecoding, CKKS↔RLWE refresh,
   │                   EvalMVBPrecompute + EvalMVBNoDecoding(LUT coeffs)
   ├─► [Linear …]    : (next slot-space block)
   ├─► …             : repeat for as many activations as you want
   └─► OutputDecoder : HomDecoding, CKKS→RLWE, DecryptCoeff
```

A *slot-space block* is the contiguous run of `LinearLayer`s between two
`ActivationLayer`s (or between the input encoder and the first activation,
or between the last activation and the output decoder). `Compile()`
measures the rescaling cost of every block, takes the global maximum as
`levelsComputation`, and pads shorter blocks with `DummyMultLayer`s so the
ciphertexts hit each activation at a consistent level.

### Building a network in code

```cpp
#include "network/activation.h"
#include "network/context.h"
#include "network/network.h"

using namespace fhednn;

FHEContext ctx;                    // defaults match the original DiNN30 setup
Network    net;
net.SetInputShift(1)               // pixels are {-1,+1}; shift to {0,2}
   .Linear(W1, b1)                 // 784 -> 30
   .Activate(activations::Sign(/*preShift=*/256, ctx.params().pInput))
   .Linear(W2, b2);                // 30 -> 10

net.Compile(ctx);                  // sizes depth, generates keys, FBT setup
auto scores = net.Run(pixels);     // returns std::vector<int64_t>
```

#### Available activations (`include/network/activation.h`)

| Factory                                              | LUT                                                |
| ---------------------------------------------------- | -------------------------------------------------- |
| `activations::Sign(preShift, pInput)`                | `(x ≥ preShift) ? +1 : -1`                         |
| `activations::Identity(pInput, pOutput)`             | `x mod pOutput` — used internally by InputEncoder |
| `activations::Step(threshold, lo, hi, pInput)`       | `(x ≥ threshold) ? hi : lo`                        |
| `activations::Custom(name, lambda, preShift, pInput)`| arbitrary `int64_t -> int64_t` LUT                 |

Add a new activation by pushing a new factory into `activations::`. Keep in
mind:

* `f` must be defined on `[0, pInput)` (post-`preShift` input).
* `pInput` must remain a power of two; the default `2^10` is the maximum
  with the current `Q` / `BIGQ` sizes.

#### Tweaking crypto parameters

Every constant from the legacy monolithic implementation now lives in
`FHEParams`:

```cpp
FHEParams p;
p.pInput   = lbcrypto::BigInteger(1) << 10;
p.pOutput  = lbcrypto::BigInteger(1) << 10;
p.Q        = lbcrypto::BigInteger(1) << 47;
p.BIGQ     = lbcrypto::BigInteger(1) << 47;
p.qBFVInit = lbcrypto::BigInteger(1) << 60;
p.numSlots = 1024;                      // CIFAR overrides this to 4096
p.ringDim  = 1u << 11;                  // CIFAR overrides this to 8192
p.scaleTHI = 32;
p.lvlb     = {3, 3};
// ...
FHEContext ctx{p};
```

Don't touch `Q`/`BIGQ`/`pInput` casually — they are at their maximum useful
values with the current depth budget.

### Generalizing to other networks

Adding a new fully-connected layer:

```cpp
net.Linear(W3, b3);
```

Adding a new activation (e.g. a binary step at threshold 100):

```cpp
net.Activate(activations::Step(/*threshold=*/100,
                               /*low=*/-1, /*high=*/+1,
                               ctx.params().pInput));
```

`Compile()` will:

1. Detect the new slot-space blocks created by the extra activation.
2. Recompute `levelsComputation = max(rescales-per-block)` across all blocks.
3. Insert `DummyMultLayer`s on shorter blocks so every block hits the next
   activation at the same depth.
4. Hand all distinct activations (including the InputEncoder's identity) to
   `FHEContext::Build`, which picks the deepest LUT (by
   `FHECKKSRNS::GetFBTDepth`) to size the bootstrap depth and seeds
   `EvalFBTSetup` with its coefficients.

Adding a new activation factory: edit
[`include/network/activation.h`](include/network/activation.h) and
[`src/network/activation.cpp`](src/network/activation.cpp) — each factory is
a one-shot lambda + metadata, ~10 lines.

Adding a non-image input modality: write a loader returning
`std::vector<int64_t>` and call `Network::Run` with it; nothing else
changes. `io::LoadImageBipolar` and `io::LoadImageGrayscale` are merely
convenience wrappers.

---

## 7. Limitations / things to know

* **MVB projects.** `pInput` and `pOutput` default to `2^10`. Increasing
  them past this point breaks the current `Q` / `BIGQ` sizing.
* **Depth budget.** `levelsComputation` is bounded above by what's left
  after FBT setup. Networks whose deepest block exceeds
  `fbtDepth + levelsAvailableAfterBootstrap` cannot be supported without
  retuning `BIGQ`. `Compile()` computes the needed depth and hands it to
  OpenFHE; if your block is too deep you'll see a `CCParams` failure at
  `GenCryptoContext`.
* **Pre-shift slot scope.** The activation pre-shift is currently applied
  to every CKKS slot (not just the slots holding live neuron values). The
  next `LinearLayer`'s mask zeroes out unused slots, so this is correct,
  but it differs cosmetically from the legacy implementation that shifted
  only the first `outDim` slots.
* **FBT seeding.** `EvalFBTSetup` is called once with the coefficients of
  the deepest registered activation. If the auto-detected "deepest"
  activation has uncharacteristic overhead, you can hard-code the
  activation passed to `EvalFBTSetup` by editing `FHEContext::Build` in
  [`src/network/context.cpp`](src/network/context.cpp).
* **Polynomial projects.** Each one is fully self-contained — they do
  *not* link `libfhednn.a` and do *not* honor the `BENCH_*` /
  `BUILD_ACCURACY` flags. They are useful as a baseline / low-RAM
  alternative, not as a drop-in for the MVB pipeline.
