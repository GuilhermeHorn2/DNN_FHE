# ReLU Bootstrapping Runner

This runner follows the same main pipeline as `poly_bootstraping_signal`:

```text
encrypt image -> linear layer 1 -> approximate activation -> bootstrap -> linear layer 2 -> decrypt
```

The differences are mostly caused by ReLU being harder to approximate under CKKS than `tanh`.

## Main Differences From Signal

- **Input encoding**
  - Signal uses bipolar pixels: `{-1, +1}`.
  - ReLU uses binary pixels: `{0, +1}`.
  - Reason: the ReLU model was trained for non-negative MNIST inputs.

- **Weights**
  - Signal loads `signal<hidden_size>_*.csv`.
  - ReLU loads `relu<hidden_size>_*.csv`.
  - Reason: these are different trained models, not the same model with a swapped activation.

- **CKKS parameters**
  - Signal uses smaller modulus settings: `dcrtBits = 50`, `firstMod = 51`, `levelsAvailableAfterBootstrap = 7`.
  - ReLU uses larger settings: `dcrtBits = 59`, `firstMod = 60`, `levelsAvailableAfterBootstrap = 10`.
  - Reason: ReLU approximation needs more depth and precision.

- **Activation scaling**
  - Signal uses a fixed pre-scale of `0.01`.
  - ReLU computes `pre_scale` from `W1` and `b1` using `compute_pre_scale_distributional()`.
  - Reason: ReLU is approximated well only when layer-1 values are mapped into a useful Chebyshev range.

- **Chebyshev degree**
  - Signal uses degree `3`.
  - ReLU uses degree `13`.
  - Reason: ReLU has a kink at zero, so a higher-degree polynomial is needed to reduce approximation error.

## Important Note

The FHE ReLU path approximates `ReLU(pre_scale * y)`, while the plaintext reference computes `ReLU(y)`.
Since `pre_scale > 0`, this preserves signs and class behavior may still match, but the hidden activation scale is different unless layer 2 or the training process accounts for it.
