# `mvb` — Modular FHE Neural-Network Framework (OpenFHE)

A small, generalizable framework for running quantized neural networks
homomorphically on top of OpenFHE's CKKS scheme, with multi-value bootstrap
(MVB) activations via the schemelet RLWE↔CKKS refresh path.

The original monolithic pipeline lived in a single `main.cpp` (~408 lines).
This refactor splits it into a reusable static library plus thin driver
binaries, exposing a PyTorch-style sequential builder so you can express new
networks in a handful of lines.

---

## 1. Directory layout

```
mvb/
├── CMakeLists.txt          single static-lib + executables, all options here
├── main.cpp                ~85-line single-image inference driver
├── accuracy.cpp            optional batch accuracy sweep (build with -DBUILD_ACCURACY=ON)
├── stb_image.h             vendored image loader (used by io::LoadImage*)
│
├── include/
│   ├── bench/
│   │   └── bench.h         compile-time-toggled scoped timers + RSS reporter
│   ├── io/
│   │   ├── csv.h           CSV loaders for weights / biases
│   │   └── image.h         PNG/JPG loaders (bipolar and grayscale)
│   └── network/
│       ├── activation.h    Activation struct + activations:: factory namespace
│       ├── context.h       FHEContext (Params + crypto context + keys + ep)
│       ├── layer.h         Layer base, LinearLayer, ActivationLayer, DummyMultLayer
│       └── network.h       Network: chainable sequential builder
│
├── src/
│   ├── bench/bench.cpp
│   ├── io/{csv,image}.cpp
│   └── network/{activation,context,layer,network}.cpp
│
├── dinn30_W1.csv           hidden-layer weights  (784 × 30)
├── dinn30_b1.csv           hidden-layer biases   (30)
├── dinn30_W2.csv           output-layer weights  (30  × 10)
├── dinn30_b2.csv           output-layer biases   (10)
└── img_{1..5}.jpg          sample MNIST-style inputs
```

CMake builds:

* `libfhednn.a` — the framework (everything under `src/`).
* `main`       — single-image inference driver.
* `accuracy`   — optional batch accuracy harness (`-DBUILD_ACCURACY=ON`).

---

## 2. The pipeline at a glance

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

A "slot-space block" is the contiguous run of `LinearLayer`s between two
`ActivationLayer`s (or between the input encoder and the first activation, or
between the last activation and the output decoder). `Compile()` measures the
rescaling cost of every block, takes the global maximum as
`levelsComputation`, and pads shorter blocks with `DummyMultLayer`s so the
ciphertexts hit the activation at a consistent level.

---

## 3. Building a network in code

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

### Available activations (`network/activation.h`)

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

### Tweaking crypto parameters

Every constant from the old monolith now lives in `FHEParams`:

```cpp
FHEParams p;
p.pInput   = lbcrypto::BigInteger(1) << 10;
p.pOutput  = lbcrypto::BigInteger(1) << 10;
p.Q        = lbcrypto::BigInteger(1) << 47;
p.BIGQ     = lbcrypto::BigInteger(1) << 47;
p.qBFVInit = lbcrypto::BigInteger(1) << 60;
p.numSlots = 1024;
p.ringDim  = 1u << 11;
p.scaleTHI = 32;
p.lvlb     = {3, 3};
// ...
FHEContext ctx{p};
```

Don't touch `Q`/`BIGQ`/`pInput` casually — they are at their maximum useful
values with the current depth budget.

---

## 4. Build commands

All build commands run from this folder.

### Default build (no benchmarks)

```bash
rm -rf build && mkdir build && cd build
cmake ..
make -j4
```

Produces `./main`.

### Build with benchmark instrumentation

```bash
cd build
cmake .. \
  -DBENCH_TOTAL=ON \
  -DBENCH_INFERENCE=ON \
  -DBENCH_BOOTSTRAP=ON \
  -DBENCH_LAYERS=ON \
  -DBENCH_MEMORY=ON
make -j4
```

### Build the batch accuracy harness

```bash
cd build
cmake .. -DBUILD_ACCURACY=ON
make -j4
```

Adds `./accuracy` next to `./main`.

### Combined

```bash
cmake .. -DBENCH_TOTAL=ON -DBENCH_INFERENCE=ON -DBENCH_BOOTSTRAP=ON \
         -DBENCH_LAYERS=ON -DBENCH_MEMORY=ON -DBUILD_ACCURACY=ON
make -j4
```

### Toggling a single flag without a full reconfigure

CMake remembers cached options between runs, so:

```bash
cd build
cmake -DBENCH_LAYERS=OFF .
make -j4
```

only recompiles the translation units that depend on the flag.

---

## 5. Running

### `main` — single-image inference

```bash
cd build
./main ../img_1.jpg
# (other samples: ../img_2.jpg ... ../img_5.jpg)
```

The driver also computes a plaintext reference using the same weights, so the
output ends with `(matches)` or `(MISMATCH)` and you immediately see whether
the FHE result is correct.

### `accuracy` — batch sweep

Built only when `-DBUILD_ACCURACY=ON`. Expects a directory laid out one
sub-directory per label:

```
test_root/
  0/some_zero.png
  0/another_zero.png
  1/...
  ...
  9/...
```

Then:

```bash
cd build
./accuracy <test_root>            # weights resolved from ../
./accuracy <test_root> <weights>  # explicit weights directory
```

It prints per-image predictions, an overall accuracy, and a 10×10 confusion
matrix at the end. The network is built and compiled exactly once and then
re-used for every image, so the cost is amortized.

---

## 6. Benchmark flag reference

Flags are mutually independent — turn on any subset. When a flag is `OFF`
its macros expand to `((void)0)` so there is **zero runtime overhead**.

### `BENCH_TOTAL`

Wraps `main()` with a single scoped timer. Reports the entire program
duration (key generation, encryption, inference, decryption, plaintext
reference check — everything).

Output line:
```
[BENCH] Total program                  195729.122 ms
```

Use when you want a top-level "wall clock" number including setup costs.

### `BENCH_INFERENCE`

Wraps the body of `Network::Run`, i.e. the work the *server* does for a
single inference: input encoding, every layer, output decoding. Excludes
`Compile()` (key generation, FBT setup), so this is the cleanest "amortizable
per-query" number.

Output line:
```
[BENCH] Inference (Run)                194483.505 ms
```

Use this when comparing two networks' runtime cost or measuring the
benefit of a new optimization that does not affect setup.

### `BENCH_BOOTSTRAP`

Wraps every `ActivationLayer::Apply` — the HomDecoding + CKKS↔RLWE refresh +
EvalMVBPrecompute + EvalMVBNoDecoding sequence, i.e. the bootstrap-equivalent
path. One line per activation in the network.

Output line:
```
[BENCH] Bootstrap (refresh+MVB)         70286.172 ms
```

Use this when profiling the FHE bottleneck. In a typical run the bootstrap
dominates total inference time.

### `BENCH_LAYERS`

Wraps every individual layer (`InputEncoder`, each `Linear`, each
`Activation`, each `DummyMult`, `OutputDecoder`). Useful for finding the
expensive layer in a deep network or seeing how much padding `DummyMult`
adds.

Output lines:
```
[BENCH] InputEncoder                    67884.312 ms
[BENCH] Linear                          48807.178 ms
[BENCH] Activation                      70286.521 ms
[BENCH] Linear                           5438.801 ms
[BENCH] OutputDecoder                    2065.538 ms
```

Note: with `BENCH_BOOTSTRAP` also enabled the activation prints both an
"Activation" line and a "Bootstrap (refresh+MVB)" line — they measure the
same code path but from different scopes.

### `BENCH_MEMORY`

Two effects:

1. Every `[BENCH] …` timer line gets an extra `(peak RSS …)` suffix
   reflecting the largest resident-set-size observed up to that point in the
   process.
2. Standalone `[BENCH][MEM] <tag>` lines are emitted at the
   instrumentation points sprinkled through the framework
   (`startup`, `after-compile`, `after-encrypt`, `after-input-encode`,
   `after-layers`, `after-output-decode`, `end`).

Output (with `BENCH_LAYERS=ON` and `BENCH_MEMORY=ON`):
```
[BENCH][MEM] startup                          RSS=6.38 MB    peak=6.38 MB
[BENCH][MEM] after-compile                    RSS=299.29 MB  peak=299.29 MB
[BENCH] InputEncoder                  67884.312 ms   (peak RSS 503.70 MB)
…
[BENCH][MEM] end                              RSS=369.66 MB  peak=515.35 MB
```

The reader uses `/proc/self/status` (`VmRSS:` and `VmHWM:`), so it works on
Linux and WSL; on other platforms it returns `0`.

### `BUILD_ACCURACY`

This is a **build option**, not a runtime instrumentation flag. When `ON`
CMake adds the `accuracy` target to the build. When `OFF` (the default)
`accuracy.cpp` is not compiled, keeping incremental builds fast.

---

## 7. Generalizing to other networks

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

## 8. Limitations / things to know

* `pInput` and `pOutput` default to `2^10`. Increasing them past this point
  breaks the current `Q`/`BIGQ` sizing.
* `levelsComputation` is bounded above by the depth budget left after FBT
  setup — networks whose deepest block exceeds `fbtDepth + levelsAvailableAfterBootstrap`
  cannot be supported without retuning `BIGQ`. `Compile()` computes the
  needed depth and passes it to OpenFHE; if your block is too deep you'll
  see a CCParams failure at `GenCryptoContext`.
* The activation pre-shift is currently applied to every CKKS slot (not
  just the slots holding active neuron values). This is fine because the
  next `LinearLayer`'s mask zeroes out the unused slots, but it differs
  cosmetically from the original implementation that shifted only the
  first `outDim` slots.
* `EvalFBTSetup` is called once with the coefficients of the deepest
  registered activation. If you find that the auto-detected "deepest"
  activation has uncharacteristically large overhead, you can hard-code
  the activation passed to `EvalFBTSetup` by editing
  `FHEContext::Build` in [`src/network/context.cpp`](src/network/context.cpp).
