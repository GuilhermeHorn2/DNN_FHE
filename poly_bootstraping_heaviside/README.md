# Heaviside Bootstrapping Runner

This runner follows the same encrypted inference flow as `poly_bootstraping_signal`:

```text
image -> encrypt -> linear layer 1 -> activation approximation -> bootstrap -> linear layer 2 -> decrypt
```

The main differences are intentional and come from the network being trained for a Heaviside-style activation instead of a sign/signal activation.

## Differences From Signal

- **Input encoding**
  - Heaviside uses binary inputs: `{0, 1}`.
  - Signal uses bipolar inputs: `{-1, 1}`.
  - Reason: the Heaviside network expects non-negative binary pixels, while the signal network expects signed/bipolar pixels.

- **Activation output range**
  - Heaviside approximates `0.5 * (1 + tanh(x))`, producing values close to `{0, 1}`.
  - Signal approximates `tanh(x)`, producing values close to `{-1, 1}`.
  - Reason: each approximation matches the activation used when training its own model.

- **Chebyshev degree**
  - Heaviside uses degree `7`.
  - Signal uses degree `3`.
  - Reason: the Heaviside-like curve is sharper/harder to approximate accurately, so it needs a higher-degree polynomial.

- **FHE parameters**
  - Heaviside uses larger CKKS modulus sizes and more available levels after bootstrap.
  - Signal uses smaller parameters.
  - Reason: the higher-degree Heaviside approximation consumes more multiplicative depth and needs more precision.

- **Weight files**
  - Heaviside loads `heaviside*_W*.csv` and `heaviside*_b*.csv`.
  - Signal loads `signal*_W*.csv` and `signal*_b*.csv`.
  - Reason: they are separately trained models, not the same model with only the runtime activation changed.

Everything else in the runner is mostly the same: linear layers, bootstrapping setup, image/folder modes, benchmarking, and final argmax prediction.
