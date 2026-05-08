# `poly_bootstraping_signal` — DiNN30 with `tanh`-approx + CKKS bootstrap

Self-contained homomorphic inference for the MNIST DiNN30 network
(`784 → 30 → 10`), trained with a **bipolar `Sign`** activation and
evaluated under FHE using a **degree-7 Chebyshev approximation of `tanh`**
plus a **CKKS bootstrap** between the activation and the second linear
layer.

This is the original of the three `poly_bootstraping_*` siblings; the
`heaviside` and `relu` variants in this directory tree were derived from
this one. For the project-level overview see the
[**root README**](../README.md).

---

## 1. How it differs from `poly_MNIST_30/`

| Aspect          | `poly_MNIST_30/`              | `poly_bootstraping_signal/` (this folder) |
| --------------- | ----------------------------- | ----------------------------------------- |
| Bootstrap       | **None** (depth-12 budget)    | **CKKS bootstrap** between layers         |
| Security level  | `HEStd_128_classic`           | `HEStd_NotSet` (research mode)            |
| Ring dim        | Inferred (~`2^14`)            | `2^16 = 65536` (needed for bootstrap depth) |
| Mult depth      | 12                            | `10 + GetBootstrapDepth({4,4})` ≈ 32      |
| Per-image cost  | ~tens of seconds              | ~200 s, dominated by linear layers        |
| Peak RAM        | ~hundreds of MB               | ~10 GB (bootstrap keys)                   |
| Hidden size     | Fixed 30                      | **CLI-selectable: 30 or 100**             |

The bootstrap lets us "refresh" the noise budget *between* the activation
and the second linear layer, which is what the MVB pipeline does too —
just much more cheaply, and at the cost of an approximated activation.

---

## 2. Pipeline

```
plaintext image
   └─► binarize each pixel to {-1, +1}      (matches sign-trained network)
   └─► CKKS-encrypt
   ├─► Linear 1 : 784 → 30 (per-neuron matvec, mask, bias)
   ├─► Activate : tanh(0.01 · x), via degree-7 Chebyshev on [-8, 8]
   ├─► CKKS Bootstrap (refreshes noise budget)
   └─► Linear 2 : 30 → 10
   └─► Decrypt → argmax
```

Per-neuron `compute_linear_layer` performs `EvalMult(weights) →
EvalSum(slots) → Rescale → mask-mult slot i → Rescale → +bias`,
accumulating one slot per neuron in the output ciphertext. This pattern
is shared verbatim by all three siblings.

The activation is essentially `Sign(x) ≈ tanh(0.01 · x)`: the small
pre-scale pushes any |sum| > ~800 into the saturated region of `tanh`,
so the output is approximately ±1 — close to the bipolar values the
network was trained with.

---

## 3. Crypto context (`setup_fhe_environment`)

```cpp
SecretKeyDist secretKeyDist = UNIFORM_TERNARY;   // bootstrap REQUIRES this
parameters.SetSecretKeyDist(secretKeyDist);
parameters.SetSecurityLevel(HEStd_NotSet);
parameters.SetRingDim(1 << 16);
parameters.SetScalingTechnique(FLEXIBLEAUTO);
parameters.SetScalingModSize(59);
parameters.SetFirstModSize(60);

std::vector<uint32_t> levelBudget = {4, 4};
parameters.SetMultiplicativeDepth(
    /*levelsAvailableAfterBootstrap=*/10
    + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist));

parameters.SetBatchSize(/*desired_slots=*/1024);
```

The depth budget breaks down as:

* Linear 1: 2 levels (mask · matvec)
* Pre-scale by `0.01`: 1 level
* Chebyshev `tanh` (degree 7): ~4 levels
* (bootstrap consumes the rest of the chain and refreshes back to top)
* Linear 2: 2 levels

`UNIFORM_TERNARY` is **not optional** — with the OpenFHE default
(`GAUSSIAN`), `EvalBootstrapKeyGen` corrupts the heap.

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
about the FHE setup (key gen, ~10 GB / minutes) runs **once** before
iterating, so per-image cost in folder mode equals the single-image FHE
eval time.

The chosen size drives:

* **CSV paths** — `dinn30_*.csv` or `dinn100_*.csv` (both copied into
  this folder so no relative-path acrobatics needed).
* **`load_csv_2d` shape checks** — `(784, H)` for `W1`, `(H, 10)` for `W2`.
* **`compute_linear_layer`** — its `num_neurons` argument for layer 1.
* **`plaintext_inference`** — the `hidden_size` parameter and both
  internal loops.

To add a new size (e.g. retrain a 64-neuron variant), drop
`dinn64_*.csv` next to the existing ones and add `64` to the validation
guard near the top of `main()`. No other code change needed.

The 30- and 100-neuron weights have very similar magnitudes
(`W1 ∈ [-8, 9]` vs. `[-8, 7]`), so the fixed `0.01` Chebyshev pre-scale
remains valid for both — no retuning required.

---

## 5. Files in this folder

```
poly_bootstraping_signal/
├── CMakeLists.txt              standalone build script
├── README.md                   you are here
├── signal_30_running.cpp       driver — handles BOTH 30- and 100-neuron variants
├── signal_30.cpp               older variant, NOT built — kept for reference (still 30-only)
├── dinn30_W{1,2}.csv           bipolar-trained DiNN30 weights (784×30, 30×10)
├── dinn30_b{1,2}.csv           DiNN30 biases (30, 10)
├── dinn100_W{1,2}.csv          bipolar-trained DiNN100 weights (784×100, 100×10)
├── dinn100_b{1,2}.csv          DiNN100 biases (100, 10)
├── img_{1..5}.jpg              sample MNIST inputs
├── result.txt                  example run captured for the README
└── stb_image.h                 vendored single-header image decoder
```

The source filename still says `30` for historical reasons; it now drives
both sizes.

---

## 6. Build & run

```bash
cd poly_bootstraping_signal
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

Sample output (from `result.txt`):

```
--- DiNN OpenFHE Inference (Single Image Mode) ---
Loading weights...
Setting up FHE Environment (Security: HEStd_NotSet, Bootstrap)...
Ring Dimension (n): 65536
Configured Slots: 1024
Multiplicative Depth: 32
…
Bootstrapping keys generated successfully.
-> FHE Base Memory: 10341 MB

--- Scores ---
Digit 0: 48.31
Digit 1: -63.59
Digit 2: 95.75
…

MATCH!
Plaintext Prediction: 2
FHE Prediction: 2

Plaintext Time: 7e-05 s
FHE Eval Time: 202.79 s
Activation + Bootstrap: 34.83 s
Total FHE Time: 203.36 s
Global Peak Memory: 10856 MB
```

The script always runs both a plaintext reference and the FHE eval, then
asserts `MATCH!`/`DIVERGENCE!` on the argmax — useful when tuning the
Chebyshev pre-scale or domain.

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
[ 100] nine_99.png                    -> pred=9 truth=9 OK     t=[l1 156.04 | a+b 34.89 | l2 11.70] s   rss=[l1 +0 | a+b +0 | l2 +0] MB   peak=[l1 +0 | a+b +0 | l2 +0] MB

=== Accuracy ===
Correct: 92 / 100  (92.00%)

Confusion matrix (row=truth, col=pred):
        0    1    2    3    4    5    6    7    8    9
  0 :  10    0    0    0    0    0    0    0    0    0
  1 :   0    9    0    0    0    0    0    0    1    0
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

## 7. Things to watch out for

* **Memory.** Bootstrap-key generation needs ~10 GB resident — make sure
  your machine has it before launching.
* **Pre-scale is hand-tuned for these weights.** `0.01` matches DiNN30's
  weight magnitudes (`W1 ∈ [-8, 9]`, `b1 ∈ [-1, 0]`). For larger weights
  (e.g. the ReLU sibling), the pre-activations would saturate `tanh`
  outside `[-8, 8]` and Chebyshev's edge wiggle would dominate the
  output — see the auto-tuner used in `poly_bootstraping_relu/`.
* **Approximation error.** Chebyshev `tanh` is smooth; bipolar inputs
  near the activation boundary can flip sign relative to the exact MVB
  result. Expect a small accuracy gap vs. `MNIST_30/`. The match check
  only enforces argmax agreement against the same tanh-approximation
  plaintext reference — not against an exact `Sign` baseline.
* **`signal_30.cpp` is NOT built.** Only `signal_30_running.cpp` is wired
  into `CMakeLists.txt`. The shorter `signal_30.cpp` is kept as a
  reference variant (no plaintext check, no timing instrumentation).
