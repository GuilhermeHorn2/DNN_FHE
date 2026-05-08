# `poly_bootstraping_relu` — DiNN30 with ReLU-approx + CKKS bootstrap

Self-contained homomorphic inference for an MNIST DNN trained with a
**ReLU** hidden activation, evaluated under FHE using a **degree-13
Chebyshev approximation of `max(0, x)`** plus a **CKKS bootstrap**
between the activation and the second linear layer.

Sibling of [`poly_bootstraping_signal/`](../poly_bootstraping_signal/README.md)
and [`poly_bootstraping_heaviside/`](../poly_bootstraping_heaviside/README.md)
— all three share the same FHE setup; they only differ in **which
activation they approximate**, what **weights** they load, and how the
**input image** is binarized.

For the project-level overview see the [**root README**](../README.md).

---

## 1. What's different from the `signal` sibling

| Aspect              | `poly_bootstraping_signal/`            | `poly_bootstraping_relu/` (this folder)              |
| ------------------- | -------------------------------------- | ---------------------------------------------------- |
| Activation          | `Sign(x) ≈ tanh(x)`                    | **`ReLU(x) = max(0, x)`**                            |
| Chebyshev degree    | 7                                      | **13** (kink at 0 hurts low-degree fit)              |
| Pre-scale           | Hard-coded `0.01`                      | **Auto-tuned** from `||W1||_∞,1`                     |
| Image binarize      | `pixel > 0.5 ? +1 : -1`                | `pixel > 0.5 ? 1 : 0`                                |
| Weights             | `dinn{30,100}_*.csv` (W1 ∈ [-8, 9])    | `relu{30,100}_*.csv` (W1 ∈ [-61, 46] / [-55, 52])    |
| Output magnitudes   | Comparable to plaintext                | **Scaled** (still argmax-correct)                    |
| Hidden size         | CLI-selectable: 30 or 100              | **CLI-selectable: 30 or 100**                        |

Everything else — Ring Dim `2^16`, slots = 1024, mult depth ≈ 32,
bootstrap with `levelBudget = {4,4}`, `UNIFORM_TERNARY` keys,
`HEStd_NotSet`, ~10 GB base memory — is identical to the `signal`
variant.

---

## 2. Why ReLU needs more care than `tanh` / Heaviside

### 2.1 Auto-tuned pre-scale

ReLU weights are an **order of magnitude larger** than the bipolar /
Heaviside ones (`W1 ∈ [-61, 46]` vs. `[-8, 9]` and `[-11, 13]`). A fixed
`0.01` pre-scale would push pre-activations far outside the Chebyshev
domain `[-8, 8]`, where the polynomial wiggles wildly.

`compute_pre_scale` (ported from `poly_cifar10/main.cpp`) picks a
data-dependent pre-scale `s` so that the *worst-case* layer-1
pre-activation `(||W1[j]||_1 + |b1[j]|)` lands at half the Chebyshev
domain:

```cpp
s = 4.0 / max_j (||W1[j]||_1 + |b1[j]|)
```

The auto-tuned value is logged at startup:

```
Auto-tuned Chebyshev pre-scale = 0.000223  (1 / 4476)
```

### 2.2 Why the argmax is still correct despite the scale

ReLU is **positively homogeneous**: `ReLU(s·x) = s·ReLU(x)` for any
`s > 0`. So scaling the input to ReLU by `s` just rescales the hidden
activations by `s`, which then rescales the layer-2 score vector by `s`.
Argmax is preserved → `MATCH!` still holds against the (unscaled)
plaintext reference, even though FHE scores look ~`s` times smaller.

### 2.3 Why degree 13

ReLU has a kink at 0, which low-degree polynomials approximate very
poorly. Degree 7 leaves visible noise around the kink; degree 13 fits
the remaining depth budget comfortably:

| Op                        | Levels |
| ------------------------- | ------ |
| Linear 1 (mask · matvec)  | 2      |
| Pre-scale by `s`          | 1      |
| Chebyshev `ReLU` (deg 13) | ~5     |
| **Subtotal**              | **8**  |
| Available before bootstrap| **10** |
| Linear 2 (after bootstrap)| 2      |

If you observe degraded accuracy, bump `degree` to `27` — it still fits
(the subtotal becomes ~10, exactly the budget).

---

## 3. Pipeline

```
plaintext image
   └─► binarize each pixel to {0, 1}            (matches non-negative ReLU output)
   └─► CKKS-encrypt
   ├─► Linear 1 : 784 → 30
   ├─► Activate : ReLU(s · x), via degree-13 Chebyshev on [-8, 8]
   │                          (s auto-tuned from W1, b1 at startup)
   ├─► CKKS Bootstrap (refreshes noise budget)
   └─► Linear 2 : 30 → 10
   └─► Decrypt → argmax
```

The plaintext reference computes **true `ReLU(sum)`** with no pre-scale
(no Chebyshev domain to respect). FHE and plaintext score vectors will
therefore differ by a global multiplicative factor; their argmaxes
still agree.

---

## 4. CLI: hidden-layer size as an argument

```
./dinn_inference <image_or_folder_path> [hidden_size]

  <image_or_folder_path>:
      - a regular file -> single-image mode (verbose output, MATCH/DIVERGENCE
                          vs the plaintext reference)
      - a directory    -> folder mode, expects the layout
                          <root>/0/*.{png,jpg,jpeg}, ..., <root>/9/*.{png,jpg,jpeg}
                          and prints accuracy + confusion matrix + aggregate timing
  hidden_size: number of neurons in the hidden layer (default: 30).
               Supported values: 30, 100.
```

Mode is auto-detected via `std::filesystem::is_directory`; everything else
about the FHE setup (key gen, ~10 GB / minutes) and the auto-tuned
Chebyshev pre-scale is computed **once** before iterating, so per-image
cost in folder mode equals the single-image FHE eval time.

The chosen size drives:

* **CSV paths** — `relu30_*.csv` or `relu100_*.csv` (both copied into
  this folder so no relative-path acrobatics needed).
* **`load_csv_2d` shape checks** — `(784, H)` for `W1`, `(H, 10)` for `W2`.
* **`compute_linear_layer`** — its `num_neurons` argument for layer 1.
* **`plaintext_inference`** — the `hidden_size` parameter and both
  internal loops.
* **`compute_pre_scale`** — automatically picks a different `s` for
  each weight set (`relu30` and `relu100` have very similar magnitudes,
  so the values come out close, but the auto-tuner adapts in either
  case).

To add a new size, drop `relu64_*.csv` next to the existing ones and
add `64` to the validation guard near the top of `main()`. No other
code change needed.

---

## 5. Files in this folder

```
poly_bootstraping_relu/
├── CMakeLists.txt              standalone build script
├── README.md                   you are here
├── relu_30_running.cpp         driver — handles BOTH 30- and 100-neuron variants
├── relu30_W{1,2}.csv           DiNN30 hidden / output weights (ReLU-trained)
├── relu30_b{1,2}.csv           DiNN30 biases
├── relu100_W{1,2}.csv          DiNN100 hidden / output weights (ReLU-trained)
├── relu100_b{1,2}.csv          DiNN100 biases
├── img_{1..5}.jpg              sample MNIST inputs
├── result.txt                  reference run (older, pre-ReLU)
└── stb_image.h                 vendored single-header image decoder
```

The source filename still says `30` for historical reasons; it now drives
both sizes. Note: `result.txt` was captured before the file was adapted;
expect different score magnitudes (smaller, by `s`) when you re-run.

---

## 6. Build & run

```bash
cd poly_bootstraping_relu
cmake -B build -S .
cmake --build build -j4
cd build

# Single-image mode (regular file) — backward compatible
./dinn_inference ../img_1.jpg          # 30 neurons (default)
./dinn_inference ../img_1.jpg 30       # explicit 30
./dinn_inference ../img_1.jpg 100      # 100 neurons
./dinn_inference ../img_1.jpg 50       # rejected: "unsupported hidden_size 50"

# Folder mode (directory) — auto-detected
./dinn_inference /path/to/test_root        # folder mode, 30 neurons
./dinn_inference /path/to/test_root 100    # folder mode, 100 neurons
```

Expected first lines of output:

```
--- DiNN OpenFHE Inference (Single Image Mode) ---
Hidden layer size: 100
Loading weights...
Auto-tuned Chebyshev pre-scale = 0.000223  (1 / 4476)
Setting up FHE Environment (Security: HEStd_NotSet, Bootstrap)...
Ring Dimension (n): 65536
…
```

The `MATCH!` / `DIVERGENCE!` line at the end compares argmaxes — not
score magnitudes — so it remains the right correctness check despite
the scale difference.

---

## 6.1 Folder mode

When the first argument is a directory, the driver iterates over the
classic per-class layout:

```
<root>/0/*.{png,jpg,jpeg}
<root>/1/*.{png,jpg,jpeg}
...
<root>/9/*.{png,jpg,jpeg}
```

Each image goes through the same FHE inference path as single-image mode
— same encrypt → linear1 → activation → bootstrap → linear2 → decrypt
sequence, with the **same auto-tuned `cheb_pre_scale`** computed once
from the loaded weights. The per-image plaintext reference is skipped,
since the folder label already provides the ground truth. Per-image
output is a single (wide) line carrying per-phase timing and memory
deltas, then an aggregate summary at the end:

```
--- DiNN OpenFHE Inference (Folder Mode) ---
Hidden layer size: 30
Loading weights...
Auto-tuned Chebyshev pre-scale = 0.000223  (1 / 4476)
…
-> FHE Base Memory: 10341 MB

Iterating over /data/mnist_test:
[   1] zero_001.png                   -> pred=0 truth=0 OK     t=[l1 156.23 | a+b 34.87 | l2 11.69] s   rss=[l1 +5 | a+b +3 | l2 +2] MB   peak=[l1 +128 | a+b +64 | l2 +12] MB
[   2] zero_002.png                   -> pred=8 truth=0 MISS   t=[l1 155.91 | a+b 34.95 | l2 11.71] s   rss=[l1 +0 | a+b +0 | l2 +0] MB   peak=[l1 +0 | a+b +0 | l2 +0] MB
…

=== Accuracy ===
Correct: 92 / 100  (92.00%)

Confusion matrix (row=truth, col=pred):
        0    1    2    3    4    5    6    7    8    9
  0 :  10    0    0    0    0    0    0    0    0    0
  …

--- Aggregate Timing ---
Total Folder Time:         20300.42 s
Mean Linear Layer 1:       156.23 s
Mean Activation+Bootstrap: 34.87 s
Mean Linear Layer 2:       11.69 s
Mean FHE Eval per image:   202.79 s

--- Aggregate Memory (mean per-image deltas) ---
FHE Base Memory:           ~10341 MB
Linear Layer 1:            RSS +0.05 MB  Peak +1.28 MB
Activation + Bootstrap:    RSS +0.03 MB  Peak +0.64 MB
Linear Layer 2:            RSS +0.02 MB  Peak +0.12 MB
Global Peak Memory:        10856 MB
```

Peak deltas typically go to `+0` after image #1, since `getrusage`'s
`ru_maxrss` is process-wide monotonic — once the first inference has
established the high-water mark, subsequent images can't push it further.
The Chebyshev pre-scale is logged once at startup, so the per-image scale
is uniform across the folder run; use image #1's per-image line for the
true worst-case footprint of each phase.

---

## 7. Things to watch out for

* **Memory budget.** Same ~10 GB bootstrap-key footprint as the other
  two siblings — independent of `hidden_size`.
* **Score magnitudes are scaled by the pre-scale.** If you want
  comparable absolute scores between FHE and plaintext, multiply the FHE
  scores by `1 / s` after decryption (or scale the plaintext reference
  by `s`). The current code doesn't do this — it just lets the argmax
  speak.
* **Approximation error near 0.** Degree-13 Chebyshev still has visible
  wiggle near the ReLU kink. For inputs whose layer-1 sum is close to 0,
  the hidden activation may be slightly negative or noisily positive.
  This rarely flips the argmax for MNIST but can on hard examples.
* **Image binarization assumption.** This driver maps pixels to `{0, 1}`
  to match ReLU's non-negative output range. If your `relu*_*.csv`
  weights were trained with raw `pixel/255.0` real-valued inputs, switch
  the binarization loop in `main` to `real_image[p] = pixel;`.
