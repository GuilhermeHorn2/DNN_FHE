# `poly_cifar10` — CIFAR-10 with polynomial-approx activation

Self-contained homomorphic inference for CIFAR-10 with the same DiNN30
topology used by [`cifar10/`](../cifar10/README.md), but with the
hidden-layer activation evaluated via a **degree-7 Chebyshev approximation
of `tanh`** instead of the exact `Sign` LUT.

Unlike the sibling `cifar10/` sub-project, this folder does **not** link
the `libfhednn.a` framework — `main.cpp` talks directly to OpenFHE CKKS,
in a single translation unit. Useful as a low-RAM baseline.

For the project-level overview and the MVB-vs-polynomial comparison, see
the [**root README**](../README.md). The trade-off table from
[`poly_MNIST_30/`](../poly_MNIST_30/README.md#1-what-the-polynomial-mode-trades)
applies here too.

---

## 1. Topology & crypto context

| Layer       | Shape           | Notes                                                            |
| ----------- | --------------- | ---------------------------------------------------------------- |
| Input       | 3072 (32×32×3)  | bipolar `{-1, +1}` after per-byte threshold (loaded RGB, channels=3) |
| Linear 1    | 3072 → 30       | per-neuron CKKS matvec + bias                                    |
| Activation  | 30 → 30         | `tanh(α · x)` via degree-7 Chebyshev on `[-8, 8]`, `α` auto-tuned |
| Linear 2    | 30 → 10         | output scores (one per CIFAR-10 class)                            |

CKKS context (`setup_fhe_environment()`):

```cpp
parameters.SetSecurityLevel(HEStd_128_classic);
parameters.SetScalingModSize(50);
parameters.SetFirstModSize(60);
parameters.SetScalingTechnique(FLEXIBLEAUTO);
parameters.SetBatchSize(4096);              // CIFAR-sized; ringDim ends up 8192
parameters.SetMultiplicativeDepth(12);
```

### Auto-tuned Chebyshev pre-scale

CIFAR's wider input fan-in makes the layer-1 pre-activation much larger
than MNIST's, so a hand-coded `0.01` doesn't work. `compute_pre_scale()`
picks the scale from the weights themselves:

```cpp
α = 4.0 / max_j (|b1[j]| + Σ_i |W1[j][i]|)
```

This guarantees the worst-case pre-activation lands inside
`[-4, 4]` ⊂ `[-8, 8]`, leaving headroom so Chebyshev edge effects don't
dominate. The chosen `α` is printed at startup.

---

## 2. Files in this folder

```
poly_cifar10/
├── CMakeLists.txt              standalone build script (no framework dependency)
├── README.md                   you are here
├── main.cpp                    self-contained driver (CKKS + Chebyshev tanh, auto-scale)
├── cifar10_weights_W1.csv      hidden-layer weights  (3072 × 30)
├── cifar10_weights_b1.csv      hidden-layer biases   (30)
├── cifar10_weights_W2.csv      output-layer weights  (30 × 10)
├── cifar10_weights_b2.csv      output-layer biases   (10)
└── cifar10_test_images.zip     ~100 sample CIFAR-10 PNGs, named "<idx>_<class>.png"
```

The `cifar10_test_images.zip` archive is identical to the one in
[`cifar10/`](../cifar10/README.md). Unzip it next to the binary before
running:

```bash
unzip cifar10_test_images.zip -d test_images
```

The polynomial driver does **not** parse the `<class>` from the filename
(only the MVB driver does that); it just prints the predicted class
index. CIFAR-10 class order is the canonical
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

---

## 3. Build & run

```bash
cd poly_cifar10
cmake -B build -S .
cmake --build build -j4
cd build
./main ../test_images/06_automobile.png
```

Sample output:

```
--- CIFAR-10 OpenFHE Inference (Polynomial Mode) ---
Loading weights...
Auto-tuned Chebyshev pre-scale = 0.000651466  (1 / 1535)
Setting up FHE Environment (Polynomial Approximation Mode, CIFAR-sized)...
  numSlots = 4096, ringDim = 8192
Generating keys...
Evaluating Layer 1...
Applying Chebyshev Polynomial Approximation (Degree 7), pre_scale = 0.000651466...
Evaluating Layer 2...
Decrypting...

--- Final Class Scores ---
Class 0: -0.42
Class 1:  1.18
…

===============================
 PREDICTED CLASS  : 1
 REFERENCE (plain): 1  (matches)
===============================
```

Unlike the MNIST poly drivers, this one **does** compute and print a
plaintext reference using exact `Sign` activation, so you can see the
gap between the polynomial FHE result and the bipolar reference for the
same input.

---

## 4. Things to watch out for

* **Same ringDim blowup as `cifar10/`.** `numSlots = 4096` forces
  `ringDim = 8192`, which is roughly 4× slower and 3-4× more
  memory-hungry than the MNIST poly drivers. Expect well over 1 GB RSS.
* **No `BENCH_*` flags / no `accuracy.cpp`.** This is a standalone build —
  the framework's benchmark and accuracy harness do not apply.
* **Polynomial vs. Sign.** Even with a well-chosen `α`, Chebyshev `tanh`
  is a smooth approximation of `Sign`. On CIFAR's already weak DiNN30
  baseline, this can shift a non-trivial fraction of predictions; treat
  the `(matches)` / `(MISMATCH)` count as a useful sanity check rather
  than as an accuracy guarantee.
* **Pre-scale logic is keyed to layer 1.** If you stack additional linear
  layers before the activation, replace the `compute_pre_scale` walk
  with one that iterates over all pre-activation layers — otherwise the
  Chebyshev domain won't cover the actual signal range.
