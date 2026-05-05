# `poly_MNIST_100` — DiNN100 with polynomial-approx activation

Self-contained homomorphic inference for the MNIST DiNN100 network
(`784 → 100 → 10`), with the hidden-layer activation evaluated via a
**degree-7 Chebyshev approximation of `tanh`** instead of the exact
`Sign` LUT used by [`MNIST_100/`](../MNIST_100/README.md).

This sub-project does **not** use the `libfhednn.a` framework. `main.cpp`
talks directly to OpenFHE CKKS — it's a single-file baseline meant to
contrast with the MVB pipeline.

For the project-level overview and the trade-off between MVB and
polynomial modes, see the [**root README**](../README.md). The trade-off
table from [`poly_MNIST_30/`](../poly_MNIST_30/README.md#1-what-the-polynomial-mode-trades)
applies verbatim here.

---

## 1. Topology & crypto context

| Layer       | Shape         | Notes                                         |
| ----------- | ------------- | --------------------------------------------- |
| Input       | 784 (28×28)   | bipolar `{-1, +1}` after grayscale threshold  |
| Linear 1    | 784 → 100     | per-neuron CKKS matvec + bias                 |
| Activation  | 100 → 100     | `tanh(0.01 · x)` via degree-7 Chebyshev on `[-8, 8]` |
| Linear 2    | 100 → 10      | output scores                                 |

CKKS context (`setup_fhe_environment()`):

```cpp
parameters.SetSecurityLevel(HEStd_128_classic);
parameters.SetScalingModSize(50);
parameters.SetFirstModSize(60);
parameters.SetScalingTechnique(FLEXIBLEAUTO);
parameters.SetBatchSize(1024);
parameters.SetMultiplicativeDepth(12);
```

`main.cpp` reuses the **same `0.01` Chebyshev pre-scale** as
`poly_MNIST_30/`. It survives the wider hidden layer because the input
width is unchanged (784 bipolar pixels), so the worst-case
pre-activation magnitude is unchanged too — see the comment near
`cheb_pre_scale` in `main.cpp`.

---

## 2. Files in this folder

```
poly_MNIST_100/
├── CMakeLists.txt              standalone build script (no framework dependency)
├── README.md                   you are here
├── main.cpp                    self-contained driver (CKKS + Chebyshev tanh)
├── dinn100_W1_0.csv            hidden-layer weights  (784 × 100)
├── dinn100_b1_0.csv            hidden-layer biases   (100)
├── dinn100_W2_0.csv            output-layer weights  (100 × 10)
├── dinn100_b2_0.csv            output-layer biases   (10)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

---

## 3. Build & run

```bash
cd poly_MNIST_100
cmake -B build -S .
cmake --build build -j4
cd build
./main ../img_1.jpg
```

Sample output:

```
--- DiNN100 OpenFHE Inference (Polynomial Mode) ---
Loading weights...
Setting up FHE Environment (Low RAM / Polynomial Approximation Mode)...
Generating keys (Fast & Low RAM)...
Evaluating Layer 1...
Applying Chebyshev Polynomial Approximation (Degree 7)...
Evaluating Layer 2...
Decrypting...

--- Final Class Scores ---
Digit 0: -2.10
Digit 1:  0.61
…

===============================
 PREDICTED DIGIT: 4
===============================
```

This driver, like `poly_MNIST_30/`, does **not** print a plaintext
reference line.

---

## 4. Things to watch out for

* **No `BENCH_*` flags.** Standalone build — the framework's benchmark
  macros aren't available. Use `std::chrono` inside `main.cpp` if you
  need numbers.
* **Layer-1 cost grows.** With 100 hidden neurons the
  `compute_linear_layer` loop performs 100 `EvalMult + EvalSum + Rescale +
  mask + Rescale + bias` cycles; this is the dominant per-inference cost.
* **Pre-scale carry-over.** The `0.01` factor is correct for the shipped
  DiNN100 checkpoint (input width is the same 784 as DiNN30). If you
  retrain with different layer-1 weight magnitudes, recompute it the way
  [`poly_cifar10/`](../poly_cifar10/README.md) does (`compute_pre_scale`
  picks `4.0 / max_row_l1`).
* **Accuracy gap.** As with `poly_MNIST_30/`, the Chebyshev `tanh` is a
  smooth surrogate for `Sign` — accuracy is slightly lower than the
  matching MVB build in `MNIST_100/`.
