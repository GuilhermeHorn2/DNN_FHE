# `poly_MNIST_30` — DiNN30 with polynomial-approx activation

Self-contained homomorphic inference for the MNIST DiNN30 network
(`784 → 30 → 10`), but with the hidden-layer activation evaluated via a
**degree-7 Chebyshev approximation of `tanh`** instead of the exact
`Sign` LUT used by [`MNIST_30/`](../MNIST_30/README.md).

This sub-project does **not** use the `libfhednn.a` framework. `main.cpp`
talks directly to OpenFHE CKKS — it's a single-file baseline meant to
contrast with the MVB pipeline.

For the project-level overview and the choice between MVB and polynomial
modes, see the [**root README**](../README.md).

---

## 1. What the polynomial mode trades

| Aspect              | MVB (`MNIST_30/`)              | Polynomial (this folder)        |
| ------------------- | ------------------------------ | ------------------------------- |
| Activation          | Exact `Sign` via LUT           | Approx `tanh` (Chebyshev deg 7) |
| Bootstrap           | RLWE↔CKKS refresh + MVB        | None                            |
| Setup time          | High (FBT precompute)          | Low                             |
| Per-inference time  | High (refresh dominates)       | Lower                           |
| Peak RAM            | High                           | Lower                           |
| Accuracy            | Exact bipolar                  | Approximate (boundary-sensitive) |
| Framework           | Uses shared `libfhednn.a`      | Standalone, single TU           |

The polynomial path is what you reach for when you want a quick MNIST
demo on a small machine; the MVB path is what you reach for when you want
the actual bipolar-DiNN result.

---

## 2. Topology & crypto context

| Layer       | Shape       | Notes                                         |
| ----------- | ----------- | --------------------------------------------- |
| Input       | 784 (28×28) | bipolar `{-1, +1}` after grayscale threshold  |
| Linear 1    | 784 → 30    | per-neuron CKKS matvec + bias (1 mask + 2 rescales) |
| Activation  | 30 → 30     | `tanh(0.01 · x)` via degree-7 Chebyshev on `[-8, 8]` |
| Linear 2    | 30 → 10     | output scores                                 |

CKKS context (set up in `setup_fhe_environment()`):

```cpp
parameters.SetSecurityLevel(HEStd_128_classic);
parameters.SetScalingModSize(50);
parameters.SetFirstModSize(60);
parameters.SetScalingTechnique(FLEXIBLEAUTO);
parameters.SetBatchSize(1024);
parameters.SetMultiplicativeDepth(12);
```

The `0.01` Chebyshev pre-scale is hand-tuned for DiNN30's layer-1
pre-activation magnitude — it maps the worst-case sums into roughly the
middle of the `[-8, 8]` Chebyshev domain.

---

## 3. Files in this folder

```
poly_MNIST_30/
├── CMakeLists.txt              standalone build script (no framework dependency)
├── README.md                   you are here
├── main.cpp                    self-contained driver (CKKS + Chebyshev tanh)
├── testar_com_64GB.cpp         experimental sibling (not built by default)
├── dinn30_W1.csv               hidden-layer weights  (784 × 30)
├── dinn30_b1.csv               hidden-layer biases   (30)
├── dinn30_W2.csv               output-layer weights  (30  × 10)
├── dinn30_b2.csv               output-layer biases   (10)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

`testar_com_64GB.cpp` is an experimental variant kept for reference and
is **not** wired into `CMakeLists.txt`; ignore it for normal use.

---

## 4. Build & run

```bash
cd poly_MNIST_30
cmake -B build -S .
cmake --build build -j4
cd build
./main ../img_1.jpg
```

Sample output:

```
--- DiNN OpenFHE Inference (Polynomial Mode) ---
Loading weights...
Setting up FHE Environment (Low RAM / Polynomial Approximation Mode)...
Generating keys (Fast & Low RAM)...
Evaluating Layer 1...
Applying Chebyshev Polynomial Approximation (Degree 7)...
Evaluating Layer 2...
Decrypting...

--- Final Class Scores ---
Digit 0: -1.93
Digit 1:  0.42
…

===============================
 PREDICTED DIGIT: 7
===============================
```

Note that this driver does **not** print a `(matches)` / `(MISMATCH)`
plaintext-reference line — it only reports the FHE-side prediction.

---

## 5. Things to watch out for

* **No `BENCH_*` flags.** This sub-project ignores the framework's
  benchmark macros entirely; if you want timings, wrap blocks in
  `std::chrono` calls inside `main.cpp` directly.
* **Pre-scale is hand-tuned.** The `0.01` factor is calibrated for the
  shipped DiNN30 weights. If you swap in a network with a different
  layer-1 dynamic range, the pre-activations will fall outside `[-8, 8]`
  and the Chebyshev approximation will saturate badly. The CIFAR poly
  variant ([`poly_cifar10/`](../poly_cifar10/README.md)) auto-tunes this
  factor from the weight matrix as a worked example.
* **Approximation error.** Chebyshev `tanh` is smooth; bipolar inputs
  near the activation boundary may flip sign relative to the exact MVB
  result. Expect a small accuracy gap vs. `MNIST_30/`.
