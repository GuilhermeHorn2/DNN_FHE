# `poly_bootstraping_heaviside` — DiNN with Heaviside-approx + CKKS bootstrap

Self-contained homomorphic inference for an MNIST DNN trained with a
**Heaviside step** activation (output in `{0, 1}`), evaluated under FHE
using a **degree-7 Chebyshev approximation of `0.5·(1 + tanh(k·x))`**
plus a **CKKS bootstrap** between the activation and the second linear
layer.

Sibling of [`poly_bootstraping_signal/`](../poly_bootstraping_signal/README.md)
and [`poly_bootstraping_relu/`](../poly_bootstraping_relu/README.md) —
all three share the same FHE setup (ring dim, depth budget, bootstrap
config); they only differ in **which activation they approximate**, what
**weights** they load, and how the **input image** is binarized to match
the trained network.

For the project-level overview see the [**root README**](../README.md).

---

## 1. What's different from the `signal` sibling

| Aspect           | `poly_bootstraping_signal/`            | `poly_bootstraping_heaviside/` (this folder) |
| ---------------- | -------------------------------------- | -------------------------------------------- |
| Activation range | `Sign(x) ∈ {-1, +1}`                   | `H(x) ∈ {0, +1}`                             |
| FHE Chebyshev    | `tanh(x)`                              | `0.5·(1 + tanh(x))`                          |
| Plaintext ref    | `tanh(0.01·x)`                         | `0.5·(1 + tanh(0.01·x))`                     |
| Image binarize   | `pixel > 0.5 ? +1 : -1`                | `pixel > 0.5 ? 1 : 0` (matches `H` output)   |
| Weights          | `dinn30_*.csv` (W1 ∈ [-8, 9])          | `heaviside{30,100}_*.csv` (W1 ∈ [-11, 13])   |
| Hidden size      | Fixed 30                               | **CLI-selectable: 30 or 100**                |

Everything else — Ring Dim `2^16`, slots = 1024, mult depth ≈ 32,
bootstrap with `levelBudget = {4,4}`, `UNIFORM_TERNARY` keys,
`HEStd_NotSet`, ~10 GB base memory — is identical to the `signal`
variant.

---

## 2. Pipeline

```
plaintext image
   └─► binarize each pixel to {0, 1}            (matches Heaviside output range)
   └─► CKKS-encrypt
   ├─► Linear 1 : 784 → H        (H ∈ {30, 100})
   ├─► Activate : 0.5·(1 + tanh(0.01·x)), via degree-7 Chebyshev on [-8, 8]
   ├─► CKKS Bootstrap (refreshes noise budget)
   └─► Linear 2 : H → 10
   └─► Decrypt → argmax
```

The `0.5·(1 + tanh(k·x))` family is a smooth surrogate for the Heaviside
step `H(x) = 1[x ≥ 0]`. With `k = 0.01`, any |sum| > ~800 saturates
`tanh` to ±1, so the surrogate output is essentially `0` or `1` — the
range the network was trained to emit.

We pre-compose this into the Chebyshev coefficients (one lambda
`x ↦ 0.5·(1 + tanh(x))`) instead of computing `tanh` and then doing two
extra ops. That keeps the depth identical to the `signal` sibling.

---

## 3. CLI: hidden-layer size as an argument

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
about the FHE setup (key gen, ~10 GB / minutes) runs **once** before
iterating, so per-image cost in folder mode equals the single-image FHE
eval time.

The chosen size drives:

* **CSV paths** — `heaviside30_*.csv` or `heaviside100_*.csv`
  (both copied into this folder so no relative-path acrobatics needed).
* **`load_csv_2d` shape checks** — `(784, H)` for `W1`, `(H, 10)` for `W2`.
* **`compute_linear_layer`** — its `num_neurons` argument for layer 1.
* **`plaintext_inference`** — the `hidden_size` parameter and both
  internal loops.

To add a new size (e.g. retrain a 64-neuron variant), drop
`heaviside64_*.csv` next to the existing ones and add `64` to the
validation guard near the top of `main()`. No other code change needed.

---

## 4. Files in this folder

```
poly_bootstraping_heaviside/
├── CMakeLists.txt              standalone build script
├── README.md                   you are here
├── heaviside_30_running.cpp    driver — handles BOTH 30- and 100-neuron variants
├── heaviside30_W{1,2}.csv      DiNN30 hidden weights (Heaviside-trained)
├── heaviside30_b{1,2}.csv      DiNN30 biases
├── heaviside100_W{1,2}.csv     DiNN100 hidden weights (copied from heavyside_mnist/)
├── heaviside100_b{1,2}.csv     DiNN100 biases
├── img_{1..5}.jpg              sample MNIST inputs
├── result.txt                  example run captured for the README
└── stb_image.h                 vendored single-header image decoder
```

The source filename still says `30` for historical reasons; it now drives
both sizes.

---

## 5. Build & run

```bash
cd poly_bootstraping_heaviside
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

The first lines of output now also report the chosen size and which mode
was selected:

```
--- DiNN OpenFHE Inference (Single Image Mode) ---
Hidden layer size: 100
Loading weights...
…
```

---

## 5.1 Folder mode

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
sequence — but the per-image plaintext reference is skipped, since the
folder label already provides the ground truth.

Per-image output is a single (wide) line carrying per-phase timing and
memory deltas, then an aggregate summary at the end:

```
--- DiNN OpenFHE Inference (Folder Mode) ---
Hidden layer size: 30
Loading weights...
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
The means therefore report the per-image cost amortized across the whole
folder, which under-states the per-image peak; use image #1's per-image
line for the true worst-case footprint of each phase.

---

## 6. Cost expectations

The expensive parts of FHE eval are:

* **`compute_linear_layer` for layer 1** — runs once per neuron. Going
  from 30 → 100 neurons multiplies this layer's runtime by ~3.3×. This
  is the dominant term, so the total FHE eval scales roughly the same.
* **Activation + Bootstrap** — operates on a single packed ciphertext;
  cost is **independent** of `hidden_size`.
* **`compute_linear_layer` for layer 2** — output is 10 neurons; cost is
  independent of `hidden_size` (each output is a dot product over the
  same 1024 slots).
* **Memory** — dominated by bootstrap keys; independent of
  `hidden_size`.

Rough projection vs. the 30-neuron run captured in `result.txt`:

| Hidden size | FHE Eval Time (estimate) | Peak RAM   |
| ----------- | ------------------------ | ---------- |
| 30          | ~200 s                   | ~10.8 GB   |
| 100         | ~600–650 s               | ~10.8 GB   |

---

## 7. Things to watch out for

* **Memory budget.** Same ~10 GB bootstrap-key footprint as the `signal`
  sibling — independent of `hidden_size`.
* **Plaintext reference uses the same surrogate.** The `MATCH!` check
  compares two evaluations of `0.5·(1 + tanh(0.01·x))` — one in
  plaintext doubles, one as Chebyshev under FHE — not against an exact
  `Heaviside` baseline. So `MATCH!` confirms the Chebyshev approximation
  agrees with itself, not that the network is exactly right.
* **Pre-scale is hand-tuned.** Heaviside-100's `W1` magnitude is similar
  to Heaviside-30's, so the same `0.01` scale and `[-8, 8]` Chebyshev
  domain still apply. If you observe divergence on some images, widen
  the domain (e.g. `[-12, 12]`) or bump the degree to 13 — both are
  one-line changes in `apply_approx_activation` and don't require any
  change to the FHE depth.
* **Image binarization assumption.** This driver maps pixels to `{0, 1}`
  to match the heaviside output range. If your weights were trained with
  a different input convention (e.g. raw `pixel/255.0`), edit the
  binarization loop in `main`.
