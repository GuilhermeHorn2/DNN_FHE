# Homomorphic Neural Network — MNIST Digit Classification

A minimal-but-complete C++ implementation of a **two-layer neural network** that
runs its activation function **entirely under encryption** using
[OpenFHE](https://openfhe.org) CKKS with **Functional Bootstrapping (FBT)**.

---

## Architecture

```
Input (784 pixels, 8-bit)
        │
        ▼  [BFV coefficients — integer plaintext mod 256]
  ┌─────────────────────────────┐
  │  Layer 1 Linear  784 → 128  │  (plaintext weights, dot-product)
  └─────────────────────────────┘
        │  z₁ = W₁·x + b₁   [integers mod 256]
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  BFV → CKKS conversion  (ConvertRLWEToCKKS)                 │
  │  ReLU via EvalFBT  (Trigonometric Hermite Interpolation)     │
  │  CKKS → BFV conversion  (ConvertCKKSToRLWE)                 │
  └──────────────────────────────────────────────────────────────┘
        │  h₁ = ReLU(z₁)   [integers mod 256, encrypted]
        ▼
  ┌─────────────────────────────┐
  │  Layer 2 Linear  128 → 10   │  (plaintext weights, dot-product)
  └─────────────────────────────┘
        │  logits = W₂·h₁ + b₂
        ▼
     argmax → predicted digit [0–9]
```

---

## Scheme Flow (BFV ↔ CKKS Bridge)

The core innovation in this code follows the functional bootstrapping approach from
the provided reference:

| Step | Description |
|------|-------------|
| **1** | Quantize pixel values to 8-bit integers, pack into BFV coefficient ciphertext |
| **2** | `SchemeletRLWEMP::EncryptCoeff` — encrypt z₁ with large initial modulus `QBFVINIT` |
| **3** | `SchemeletRLWEMP::ModSwitch` — reduce modulus from `QBFVINIT` to `Q` |
| **4** | `SchemeletRLWEMP::ConvertRLWEToCKKS` — lift BFV ciphertext into CKKS slots |
| **5** | `cc->EvalFBT` — evaluate ReLU via Trigonometric Hermite Interpolation over ℤ₂₅₆ |
| **6** | `SchemeletRLWEMP::ConvertCKKSToRLWE` — project CKKS slots back to BFV coefficients |
| **7** | `SchemeletRLWEMP::DecryptCoeff` — decrypt h₁ for the next layer (or keep encrypted) |

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PInput` / `POutput` | 256 | 8-bit quantized activations |
| `RING_DIM` | 4096 | CKKS ring dimension |
| `NUM_SLOTS` | 128 | HIDDEN_DIM neurons packed per ciphertext |
| `order` | 1 | THI interpolation order (first-order, faster) |
| `scaleTHI` | 32 | Hermite coefficient scale-down factor |
| `lvlb` | {3, 3} | Levels for homomorphic encoding/decoding |
| `secretKeyDist` | `SPARSE_ENCAPSULATED` | Tighter security / slightly more noise |

---

## ReLU via Functional Bootstrapping

The activation function is the **ReLU over ℤ₂₅₆** (8-bit signed interpretation):

```
relu(x) = x      if x ∈ [0, 127]   (positive half)
        = 0      if x ∈ [128, 255] (negative half, i.e. x − 256 < 0)
```

`GetHermiteTrigCoefficients` computes the Trigonometric Hermite Interpolation
coefficients of this function over the discrete domain {0, …, 255}, and
`EvalFBT` evaluates the resulting polynomial homomorphically inside a single
bootstrapping call.

---

## Prerequisites

- **OpenFHE** ≥ 1.2 (must include `schemelet/rlwe-mp.h` and `math/hermite.h`)
- CMake ≥ 3.16
- C++17 compiler (GCC ≥ 10, Clang ≥ 12)

---

## Build & Run

```bash
# 1. Clone and build OpenFHE (if not already installed)
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install

# 2. Build this project
mkdir build && cd build
cmake .. -DOPENFHE_ROOT=/usr/local
make -j$(nproc)

# 3. Run
./he_neural_network
```

---

## Expected Output

```
============================================================
  Homomorphic Neural Network — MNIST Digit Classification
  OpenFHE CKKS + Functional Bootstrapping (BFV <-> CKKS)
============================================================

[Info] Pixel input (first 8 values): [143, 27, 231, ...]

[Plaintext] Logits: [...]
[Plaintext] Predicted digit: 3

[Setup] Building CKKS crypto context for FBT ReLU...
[Setup] CKKS ring dimension = 4096, multiplicative depth = N
[Setup] Keys and FBT parameters ready.

[Layer 1] Pre-activations z1 (first 8): [...]
[Layer 1] Encrypting pre-activations as BFV coefficients...
[Layer 1] Converting BFV -> CKKS...
[Layer 1] Applying ReLU via EvalFBT (Functional Bootstrapping)...
[Layer 1] Converting CKKS -> BFV coefficients...
[Layer 1] HE ReLU activations h1 (first 8): [...]
[Layer 1] Max absolute error (HE vs plain ReLU): 0

[Layer 2] Logits (HE h1): [...]
[Layer 2] Predicted digit (HE): 3
[Layer 2] Predicted digit (Plain): 3

============================================================
  Summary
  ...
  Match: YES ✓
============================================================
```

---

## Extending to a Full HE Network

This demo decrypts after Layer 1 for clarity. To keep everything encrypted:

1. **Layer 2 in CKKS** — encode W₂ as CKKS plaintexts, use `EvalMult` + `EvalSum`
   (inner-product via rotations) to compute logits homomorphically.
2. **Argmax** — use the iterative comparison circuit or FBT step function to find
   the maximum index without decryption.
3. **Batching** — encrypt multiple images simultaneously using the remaining
   CKKS slots (up to `RING_DIM / 2`).
4. **Trained weights** — load pretrained quantization-aware-trained (QAT) MNIST
   weights from a `.bin` or `.json` file instead of random weights.
