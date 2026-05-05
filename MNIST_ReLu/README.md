# `MNIST_ReLu` — DiNN100 driver with ReLU activation

MVB-based homomorphic inference for the DiNN100 MNIST topology
(`784 → 100 → 10`) using **ReLU** instead of `Sign`/`Heaviside`. Uses
the same shared framework (`../include`, `../src`) as
[`MNIST_signal/`](../MNIST_signal/README.md) and
[`MNIST_heaviside/`](../MNIST_heaviside/README.md).

---

## Build & run

```bash
cd MNIST_ReLu
cmake -B build -S .
cmake --build build -j4
./build/main ../MNIST_ReLu/img_1.jpg     # img_1..img_5 are sample inputs
```

The driver also computes a plaintext reference and prints `(matches)` /
`(MISMATCH)`. With the scaling baked into [`main.cpp`](main.cpp), all
five sample images report `(matches)`.

---

## The problem this folder had to solve

ReLU does not slot into the framework's `pInput`/`pOutput = 1024` budget
the way `Sign` and `Heaviside` do. With the raw integer weights, two
distinct overflows happen on every image:

1. **LUT wrap on the hidden side.** Pre-activations `y = W1·x + b1` reach
   roughly `[-900, +900]`, but the ReLU LUT only covers
   `[-preShift, pInput - preShift) = [-256, +768)`. Negative `y` wraps
   modulo `pInput`, so ReLU returns large positive garbage instead of `0`.

2. **Decoder wrap on the output side.** Final scores `W2·h + b2` reach
   roughly `±10⁵`, but `DecryptCoeff` interprets them mod `pOutput = 1024`
   as a signed window `(-512, +512)`. Anything outside that window is
   wrapped, destroying the argmax.

For `Sign`/`Heaviside` neither overflow happens: the activation only
needs the *sign* of `y` (so LUT wrap is harmless in most cases), and the
`{-1,+1}` / `{0,1}` hidden values combined with small `W2` weights keep
the scores inside `±512`. ReLU loses both of these properties.

---

## Solution: uniform pre-scaling of `W1/b1` and `W2/b2`

Argmax is invariant under any positive scalar, so we can shrink every
internal value into the safe windows by dividing the integer weights and
biases by two constants *before* handing them to `Network`:

```cpp
constexpr double HIDDEN_SCALE_K = 4.0;   // keeps |y'| < 256
constexpr double OUTPUT_SCALE_K = 64.0;  // keeps |scores'| < 512
for (auto& row : W1) for (auto& w : row) w /= HIDDEN_SCALE_K;
for (auto& v : b1)                       v /= HIDDEN_SCALE_K;
for (auto& row : W2) for (auto& w : row) w /= OUTPUT_SCALE_K;
for (auto& v : b2)                       v /= OUTPUT_SCALE_K;
```

`LinearLayer` already encodes weights as doubles, so non-integer values
are handled natively by CKKS. The plaintext reference uses the same
`W*/b*` vectors and so stays in lockstep with the FHE result.

`main.cpp` also prints two diagnostic blocks (raw vs. scaled) showing
the actual `pre-act y`, `hidden h`, and `refScores` ranges per image —
useful for retuning either constant if the model or the input changes.

### Why nothing else was needed

- A clipped ReLU would have helped the *output* side but not the *hidden*
  side: the LUT wraps before the lambda fires, so clipping happens too
  late. Scaling `W1/b1` is the only fix for the hidden overflow.
- Increasing `pOutput` past `2^10` is explicitly unsafe with the current
  `Q`/`BIGQ` budget (see the [root README §7](../README.md#7-limitations--things-to-know)).
- Switching the input convention to `{-1,+1}` would not help — the
  score-magnitude argument is the same.

---

## Sample run

```
[Diag] pre-activation y range : [  -886.0,    864.0]   (LUT safe range is [-256, +768))
[Diag] raw refScores range   : [-108828.0,  76222.0]   (decoder safe range is (-512, +512))
[Diag] scaled W1/b1 by 1/4.0, W2/b2 by 1/64.0 (argmax invariant)
[Diag] scaled pre-act y'    : [  -221.5,    216.0]   (need (-256, +768))
[Diag] scaled refScores     : [  -425.1,    297.7]   (need (-512, +512))
...
 PREDICTED DIGIT  : 2
 REFERENCE (plain): 2  (matches)
```

All five sample images:

| Image  | Truth | FHE | Plain | Status                                  |
| ------ | ----- | --- | ----- | --------------------------------------- |
| img_1  |   2   |  2  |   2   | matches, correct                        |
| img_2  |   0   |  0  |   0   | matches, correct                        |
| img_3  |   9   |  9  |   9   | matches, correct                        |
| img_4  |   0   |  9  |   9   | matches (model misclassifies, FHE = plain) |
| img_5  |   3   |  3  |   3   | matches, correct                        |

---

## Files in this folder

```
MNIST_ReLu/
├── CMakeLists.txt              build script (links ../src + ../include)
├── README.md                   you are here
├── main.cpp                    single-image inference driver (with scaling + diagnostics)
├── relu100_W1.csv              hidden-layer weights  (784 × 100)
├── relu100_b1.csv              hidden-layer biases   (100)
├── relu100_W2.csv              output-layer weights  (100 × 10)
├── relu100_b2.csv              output-layer biases   (10)
├── relu30_*.csv                DiNN30 alternative (not used by main.cpp)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

The framework-side change for this folder is small: a `ReLU(preShift,
pInput)` factory was added to
[`include/network/activation.h`](../include/network/activation.h) and
[`src/network/activation.cpp`](../src/network/activation.cpp), alongside
the existing `Sign`/`Heaviside`/`Identity`/`Step`/`Custom`.

---

## How `HIDDEN_SCALE_K` and `OUTPUT_SCALE_K` were chosen

Each constant has its own constraint, derived from the diagnostic block
in `main.cpp` rather than from any general theory.

### `HIDDEN_SCALE_K` — bounded by the LUT-safe window

The ReLU LUT only covers
`[-preShift, pInput - preShift) = [-256, +768)`, so:

```
max(|y|) / HIDDEN_SCALE_K  <  256
```

Across the five sample images the worst pre-activation range was
`img_2`'s `[-896, +659]`, giving `max(|y|) = 896`:

```
HIDDEN_SCALE_K  >  896 / 256  ≈  3.5
```

Next power of two → **`K1 = 4`**. Verified after the fact: scaled `y'`
ended up at `[-224, +165]`, just inside the window.

### `OUTPUT_SCALE_K` — bounded by the decoder window

`DecryptCoeff` reads class scores mod `pOutput = 1024` as a signed
`(-512, +512)` window. The `H1` scaling already shrinks the dominant
`W2·h` term by `K1` (the bias term `b2` is single-digit and negligible
next to `|W2·h|` ~ 10⁵), so the requirement on `K2` is:

```
max(|raw refScores|) / (HIDDEN_SCALE_K · OUTPUT_SCALE_K)  <  512
```

The worst raw range was `img_1`'s `[-108828, +76222]`. With `K1 = 4`:

```
OUTPUT_SCALE_K  >  108828 / (4 · 512)  ≈  53
```

Next power of two → **`K2 = 64`**. Verified: scaled `refScores` ended
up at `[-425, +298]`.

### Why powers of two

Three pragmatic reasons, none load-bearing:

1. CKKS rescaling is happiest with power-of-two scales (no extra
   rounding entering the modulus chain).
2. Easy to budget as bit-shifts of precision (`K1 = 2²` and
   `K2 = 2⁶` cost 8 bits combined; with `scaleTHI = 32 = 2⁵` the total
   sits well within double-precision CKKS slots).
3. Easy to step up if a new image overflows: `K1 → 8`, `K2 → 128`, etc.

---

## Tuning the scales for new weights or inputs

If you swap to `relu30_*` weights, train new ones, or feed denser
images, rerun once and re-derive `K1` and `K2` from the diagnostic
block:

1. Run with the existing `K`s once. Read the printed `[Diag]` lines.
2. `HIDDEN_SCALE_K  =  next_pow2( max(|y|) / 256 )`
3. `OUTPUT_SCALE_K  =  next_pow2( max(|raw refScores|) / (HIDDEN_SCALE_K · 512) )`
4. Rebuild and confirm both *scaled* diagnostic ranges sit inside their
   target windows with comfortable margin. If not, double whichever
   constant is tight.
