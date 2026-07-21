# DNN_FHE — Homomorphic Neural-Network Inference on OpenFHE

A collection of small, self-contained projects that run quantized
neural-network inference **fully homomorphically** on top of
[OpenFHE](https://github.com/openfheorg/openfhe-development)'s CKKS scheme.

The repo evaluates the same MNIST DiNN topology (`784 → {30,100} → 10`) with
three different activation functions, using two different strategies for
evaluating the non-linearity under FHE:

| Strategy                | Mechanism                                              | Folders                                                                  |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| **MVB / FBT**           | Exact LUT via schemelet RLWE↔CKKS refresh + `EvalMVB*` | `MNIST_signal/`, `MNIST_heaviside/`, `MNIST_relu/`                       |
| **Polynomial + bootstrap** | Chebyshev approximation, then `EvalBootstrap`         | `poly_bootstraping_signal/`, `poly_bootstraping_heaviside/`, `poly_bootstraping_relu/` |

Each strategy is applied to three activations — `signal` (`Sign`, ±1),
`heaviside` (0/1 step), and `relu` — giving six sub-projects in total.

* The **MVB** projects share a small reusable framework
  (`include/` + `src/` → `libfhednn.a`) that exposes a PyTorch-style
  sequential builder. They use multi-value bootstrap via the schemelet
  RLWE↔CKKS refresh path to evaluate exact look-up-table activations.
* The **polynomial** projects are standalone single-file implementations
  that talk directly to OpenFHE: they approximate the activation with
  `EvalChebyshevFunction` (degree tuned per activation) and then restore
  precision with a full `EvalBootstrap`. They don't link `libfhednn`, but
  they do share the `bench/` instrumentation and support the same
  batch-accuracy workflow described in §6.

---

## 1. Repository layout

```
DNN_FHE/
├── include/                           public headers for the MVB framework
│   ├── bench/bench.h                  scoped timers + RSS reporter (compile-time gated)
│   ├── io/{csv,image,accuracy}.h      CSV / PNG-JPG loaders + shared batch harness
│   └── network/{activation,context,
│                layer,network}.h      builder, layers, FHEContext, activations
│   └── stb_image.h                    vendored image decoder
│
├── src/                               framework implementation (compiled into libfhednn.a)
│   ├── bench/bench.cpp
│   ├── io/{csv,image,accuracy}.cpp    accuracy.cpp = shared batch-eval harness (see §6)
│   └── network/{activation,context,layer,network}.cpp
│
├── MNIST_signal/                      MVB · Sign activation (±1)
├── MNIST_heaviside/                   MVB · Heaviside step (0/1)
├── MNIST_relu/                        MVB · ReLU
│
├── poly_bootstraping_signal/          polynomial · tanh Chebyshev approx of Sign
├── poly_bootstraping_heaviside/       polynomial · Chebyshev approx of Heaviside
├── poly_bootstraping_relu/            polynomial · Chebyshev approx of ReLU
│
├── data/mnist/                        bundled 50-image sample set, one folder per digit
├── run_all.sh                         batch driver: runs all six pre-built variants
└── run_logs/results/                  committed reference runs (accuracy + timing)
```

Each `MNIST_*` sub-project owns its own `main.cpp`, `CMakeLists.txt`, weight
CSVs, and sample images, and links against the shared library compiled from
`../src`. Each `poly_bootstraping_*` sub-project is a single independent
translation unit (`<activation>_running.cpp`) that borrows only
`src/bench` from the shared tree.

---

## 2. Prerequisites

* **OpenFHE** (development branch) installed system-wide so
  `find_package(OpenFHE REQUIRED)` resolves. The MVB projects additionally
  need the `schemelet` headers (RLWE↔CKKS refresh / `EvalFBT*`) bundled with
  recent OpenFHE.
* **CMake ≥ 3.10**, a **C++17** compiler, and the usual GMP / NTL stack
  pulled in by OpenFHE.
* Linux/WSL recommended (memory benchmarks read `/proc/self/status`).
* **Heavy runs.** These are not laptop-friendly demos: a full batch sweep
  over `data/mnist` can take on the order of hours per variant and use
  several GB (poly) to 10+ GB (MVB) of RSS — see `run_logs/results/` for
  concrete numbers from prior runs.

---

## 3. Common build & run idiom

Every sub-project produces a single executable called `main` that is
**dual-mode**: pass it an image to do a one-shot inference (with diagnostics
+ plaintext reference), or pass it a directory tree to run the batch
accuracy harness over a labeled set (see §6). All six sub-projects (MVB and
polynomial alike) follow the same idiom:

```bash
cd <subproject>                        # e.g. cd MNIST_signal
cmake -B build -S .
cmake --build build -j4
cd build
./main ../img_1.jpg                    # single image, hidden size 30 (default)
./main ../img_1.jpg 100                 # single image, hidden size 100
./main ../../data/mnist                 # batch + confusion matrix
```

**Run from inside `build/`.** Each `main.cpp` loads its weight CSVs from a
path relative to the current working directory (e.g. `"../signal30_W1.csv"`),
so invoking the binary from anywhere other than `<subproject>/build/` will
fail to find its weights.

`hidden_size` selects which weight files are loaded
(`<activation><hidden_size>_{W1,W2,b1,b2}.csv` in the sub-project directory)
and accepts `30` (default) or `100`.

### Build options (CMake cache flags)

Available in **all six** sub-projects — each `CMakeLists.txt` compiles a
static bench library from `../src/bench` and consumes these flags:

| Flag                | Effect                                                              |
| ------------------- | ------------------------------------------------------------------- |
| `-DBENCH_ALL=ON`    | Convenience umbrella — turns every BENCH_* flag below ON at once    |
| `-DBENCH_TOTAL=ON`  | Wraps `main()` with a single end-to-end timer                       |
| `-DBENCH_INFERENCE` | Times the per-query inference path (excludes one-time setup)        |
| `-DBENCH_BOOTSTRAP` | Times the activation/refresh step (MVB refresh or CKKS bootstrap)   |
| `-DBENCH_LAYERS`    | Times every individual layer/phase                                  |
| `-DBENCH_MEMORY`    | Adds `(peak RSS …)` to every timer line + `[BENCH][MEM]` checkpoints |

When a flag is `OFF`, its macros expand to `((void)0)` — zero runtime cost.

```bash
cmake -B build -S . -DBENCH_ALL=ON
cmake --build build -j4
```

Sample output with all flags on (MVB variant):

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

## 4. Batch driver (`run_all.sh`)

`run_all.sh` runs all six sub-projects back-to-back over the same data
directory. It does **not** build anything — each `<variant>/build/main`
must already exist (see §3) — it just invokes each binary with hidden size
30, captures stdout to a timestamped log, and prints a wall-time summary.

```bash
./run_all.sh                    # defaults to data/mnist
./run_all.sh /path/to/test_root
```

Logs land in `run_logs/<timestamp>/<variant>_h<size>.log`; any variant
whose binary isn't built is skipped with a note on how to build it.
`run_logs/results/` holds a set of previously committed reference runs
(accuracy + timing) for all six variants at hidden sizes 30 and 100.

---

## 5. Batch accuracy harness

MVB sub-projects have **no separate binary**. Each `main` accepts either a
single image **or a directory tree** — when it gets a directory it
delegates to a shared loop that lives in
[`src/io/accuracy.cpp`](src/io/accuracy.cpp) (declared in
[`include/io/accuracy.h`](include/io/accuracy.h)). The `poly_bootstraping_*`
projects don't call into this shared harness (they don't link `libfhednn`),
but each implements the same folder-mode / confusion-matrix behavior
directly in its own `main.cpp`.

This means each sub-project's `main.cpp` already encodes the *correct*
network shape, activation, and pixel encoding for that project — so the
harness inherits all of those automatically. Conversely, the MVB harness
does **not** know how to interpret pixels by itself; each `main.cpp` passes
a small `PixelLoader` lambda telling it which `io::LoadImage*` to call per
image, and optionally a `PlainScorer` lambda that re-runs the same
inference in plaintext for comparison.

### Expected directory layout

One sub-directory per integer label, each containing image files:

```
test_root/
  0/some_zero.png
  0/another_zero.png
  1/...
  ...
  9/...
```

`.png`, `.jpg`, and `.jpeg` are picked up; everything else is skipped.
`data/mnist/` in this repo follows this layout with a handful of sample
images per digit.

### Build & run

```bash
cd <subproject>
cmake -B build -S . -DBENCH_ALL=ON     # bench flags are optional
cmake --build build -j4
cd build
./main /path/to/test_root              # batch mode (directory)
./main ../img_1.jpg                    # single-image mode (file)
```

`Network::Compile` (MVB) / one-time key generation (poly) happens exactly
**once** before the loop, so the per-image cost in batch mode is
essentially the inference-only cost, scaled by image count.

### Output (batch mode)

```
[   1] some_zero.png  -> pred=0  plain=0  truth=0  OK
[   2] another_zero.png -> pred=8  plain=0  truth=0  MISS
…

=== FHE vs PlainText ===
Hit: 47 / 50  (94.00%)

=== PlainText Acc ===
Correct: 49 / 50  (98.00%)

=== Accuracy ===
Correct: 46 / 50  (92.00%)

Confusion matrix (row=truth, col=pred):
         0    1    2    3    4    5    6    7    8    9
  0 :    5    0    0    0    0    0    0    0    0    0
  1 :    0    5    0    0    0    0    0    0    0    0
  …
```

The `=== FHE vs PlainText ===` and `=== PlainText Acc ===` blocks only
appear when a `PlainScorer` was supplied (all current MVB `main.cpp`s
supply one); otherwise only `=== Accuracy ===` and the confusion matrix
print.

### Adding the dispatch to a new MVB driver

If you spin up a new MVB sub-project, add this block right after
`net.Compile(ctx)` in its `main.cpp`:

```cpp
#include "io/accuracy.h"
#include <filesystem>
namespace fs = std::filesystem;
...
if (fs::is_directory(argv[1])) {
    auto loadPixels = [](const std::string& p) {
        return io::LoadImageBinary(p.c_str(), IN_DIM);   // or LoadImageBipolar(...)
    };
    auto r = io::RunAccuracyLoop(net, argv[1], loadPixels, OUT_DIM);
    io::PrintAccuracySummary(r);
    return 0;
}
// ...existing single-image flow follows...
```

The lambda is the only place that encodes "what pixel encoding does this
network expect" — the rest of the harness is project-agnostic.

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
   .Linear(W1, b1)                 // 784 -> HID_DIM
   .Activate(activations::Sign(/*preShift=*/256, ctx.params().pInput))
   .Linear(W2, b2);                // HID_DIM -> 10

net.Compile(ctx);                  // sizes depth, generates keys, FBT setup
auto scores = net.Run(pixels);     // returns std::vector<int64_t>
```

#### Available activations (`include/network/activation.h`)

| Factory                                              | LUT                                                |
| ---------------------------------------------------- | -------------------------------------------------- |
| `activations::Sign(preShift, pInput)`                | `(x ≥ preShift) ? +1 : -1`                         |
| `activations::Heaviside(preShift, pInput)`           | `(x ≥ preShift) ? 1 : 0`                           |
| `activations::ReLU(preShift, pInput)`                | `max(0, x - preShift)`                             |
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
p.numSlots = 1024;
p.ringDim  = 1u << 16;
p.scaleTHI = 32;
p.lvlb     = {3, 3};
p.dnum     = 3;
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
changes. `io::LoadImageBipolar` and `io::LoadImageBinary` are merely
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
* **Polynomial projects.** Each one is fully self-contained — it does
  *not* link `libfhednn.a`. It *does* honor the `BENCH_*` flags (via its own
  `polybench` static library) and *does* support directory/batch mode with
  a confusion matrix, mirroring the MVB harness's output shape but
  implemented independently in each `<activation>_running.cpp`. Precision
  restoration uses a full CKKS `EvalBootstrap` (requires `UNIFORM_TERNARY`
  secret-key distribution — `GAUSSIAN` corrupts the heap in
  `EvalBootstrapKeyGen`), and the activation itself is a Chebyshev fit over
  a fixed input range (`[-8, 8]`, after scaling by `0.01`) whose degree is
  tuned per activation (`signal` deg 3, `heaviside` deg 7, `relu` deg 13 —
  the kink at 0 needs the extra degree). Accuracy is therefore sensitive to
  both the fitted range and the chosen degree, and can be noticeably lower
  than the exact MVB path — see `run_logs/results/` for measured numbers.
