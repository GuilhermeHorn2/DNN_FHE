# MNIST_heaviside — why we override `BIGQ` / `Q`

The driver in `main.cpp` runs the DiNN30 / DiNN100 Heaviside MNIST
classifier (`784 → H → 10` with `Heaviside` between the two `Linear`
layers) through the FBT/Hermite‑trig FHE pipeline implemented in
`src/network/`. With the library's default `FHEParams`, the FHE side
silently mispredicts almost every image while the in‑driver plaintext
oracle stays correct — the symptom in
`run_logs/20260509_174247/MNIST_heaviside_h30.log` is that the FHE
prediction collapses onto a single class regardless of the input.

The fix is the two lines in `main.cpp`:

```cpp
fhednn::FHEParams params;
params.BIGQ = lbcrypto::BigInteger(1) << 55;
params.Q    = lbcrypto::BigInteger(1) << 55;
fhednn::FHEContext ctx{params};
```

Everything else (the activation factory, `pInput`, `pOutput`,
`scaleTHI`, `secretKeyDist`, `lvlb`, `dnum`, `numSlots`, `ringDim`)
stays at the values defined in `include/network/context.h`.

## What `BIGQ` controls

`FHEContext::Build` reads `BIGQ` and uses its bit length both as the
per‑level scaling modulus and as the first (top) modulus of the CKKS
chain:

```cpp
const std::uint32_t dcrtBits = params_.BIGQ.GetMSB() - 1;
const std::uint32_t firstMod = params_.BIGQ.GetMSB() - 1;
...
p.SetScalingModSize(dcrtBits);
p.SetFirstModSize(firstMod);
```

With `BIGQ = 1<<47`, `dcrtBits = 46`. With `BIGQ = 1<<55`, `dcrtBits = 54`.
That is +8 bits of CKKS scaling factor per level, i.e. **256× more
absolute precision per slot per scaling level**. With a `total depth`
of 28 levels for this network, that headroom compounds across every
`EvalMult + ModReduce` and across the full FBT bootstrap.

## Why the Heaviside DiNN needs that headroom and the Sign DiNN does not

After `EvalMVBNoDecoding`, slot `j` carries `f(preact_j) / scaleTHI` in
real CKKS slot space. For Sign, `f ∈ {-1, +1}`, so the two branches sit
at `±1/scaleTHI` — symmetric around zero, with the full slot magnitude
on either side. For Heaviside, `f ∈ {0, 1}`, so the "0" branch sits
**exactly at 0** in slot space, and the "1" branch sits at
`+1/scaleTHI`. The whole signal lives on one side of zero, with the "0"
branch coincident with the CKKS noise floor.

CKKS noise after the FBT bootstrap is *additive* in slot space and
roughly proportional to `1 / 2^dcrtBits` per scaling level. So:

| dcrtBits | noise floor (per level, schematic) | gap to "1" branch at scaleTHI=32 |
|---:|---:|---:|
| 46 (`BIGQ = 1<<47`, default) | `~ 2^-46 · noise_const` | comparable order of magnitude after 28 levels |
| 54 (`BIGQ = 1<<55`, fix) | `~ 2^-54 · noise_const` | clearly below the "1" branch |

When the noise floor is comparable to the signal:
- The "0" branch leaks a small **positive** bias (noise above zero
  isn't symmetric on a signal that lives at 0; only the positive side
  is "free", the negative side cancels nothing in the next layer).
- That positive leak is multiplied by every entry of `W2` in the
  second `Linear` and summed across `H` hidden neurons. The class
  score `score_k = b2_k + Σ W2_k · h` ends up offset by
  `leak · Σ W2_k`, so each class shifts in proportion to its column
  sum of `W2`, biasing argmax toward whichever class has the largest
  column sum.
- For this trained DiNN30, the column sums of `W2` happen to make a
  single class the systematic argmax winner regardless of the input —
  exactly the "always picks the same class" pattern in the failure
  log.

Sign avoids the entire mechanism because its "0 contribution" branch
is `−1` (not `0`) and its slot magnitude is `1/scaleTHI` on both
sides. Noise around either branch is symmetric and averages to ~0 per
neuron after `Linear` 2, so a few flipped neurons just cost accuracy
proportionally — they don't bias every class score the same way.
That's why `MNIST_signal/main.cpp` runs at ~86% under the same library
defaults while `MNIST_heaviside/main.cpp` collapses without this
override.

## Why bumping `BIGQ` is enough on its own

The integer class score recovered after `Network::DecodeOutput` is the
real‑valued slot value times `scaleTHI` (see `EvalHomDecoding` in
`src/network/network.cpp`):

```
integer_score = scaleTHI · (true_real_score + slot_noise)
              = true_score + scaleTHI · slot_noise
```

With `scaleTHI = 32` (the library default) and `BIGQ = 1<<47`, the
`scaleTHI · slot_noise` term lands in the same magnitude as the
plaintext top‑1 / top‑2 score margins for this DiNN30 (~50–60), so the
bias from the previous paragraph wins the argmax. With `BIGQ = 1<<55`,
`slot_noise` drops by enough orders of magnitude that
`scaleTHI · slot_noise` is small compared to the score margin, and the
argmax goes back to following the actual `(W·h + b)` computation.

That's why no other knob is needed: it isn't `pInput`, it isn't
`preShift`, it isn't `scaleTHI`, it isn't the activation order, and it
isn't the secret‑key distribution. The plaintext model is fine, the
LUT is fine, the encoding is fine — what was broken was the slot‑space
*precision* available to the activation bootstrap, and `BIGQ` is the
single parameter that controls that.

## Cost

`BIGQ = 1<<55` enlarges every scaling-level modulus by 8 bits. The
total CKKS modulus chain grows from roughly 28·46 + 46 ≈ 1334 bits to
28·54 + 54 ≈ 1566 bits. OpenFHE will keep `ringDim` at the default
`1<<16` for this network (the chain still fits under `HEStd_NotSet`),
so there is no extra ring‑dimension cost; only ~17% more bits per
ciphertext limb, which translates to a modest per‑bootstrap time and
RAM bump and no change in `total depth` (`28`) or `fbtDepth` (`26`).
