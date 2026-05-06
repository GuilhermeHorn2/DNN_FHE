# `MNIST_100` — DiNN100 driver on the MVB framework

MVB-based homomorphic inference for the wider **DiNN100** MNIST network:
`784 → 100 → 10`, hard-`Sign` activation, exact look-up-table evaluation
through the schemelet RLWE↔CKKS refresh path.

For framework internals, build-flag reference, and the dual-mode batch
accuracy harness, see the [**root README**](../README.md).

---

## 1. Topology

| Layer       | Shape        | Notes                                       |
| ----------- | ------------ | ------------------------------------------- |
| Input       | 784 (28×28)  | bipolar `{-1, +1}` after grayscale threshold |
| Linear 1    | 784 → 100    | exact integer matvec + bias                 |
| `Sign`      | 100 → 100    | `(x ≥ 0) ? +1 : -1` via MVB LUT             |
| Linear 2    | 100 → 10     | output scores                               |

Crypto context uses the framework defaults from `FHEParams`
(`numSlots = 1024`, `ringDim = 2048`) — both layers fit in the same slot
budget as DiNN30.

---

## 2. Files in this folder

```
MNIST_signal/
├── CMakeLists.txt              build script (links ../src + ../include)
├── README.md                   you are here
├── main.cpp                    dual-mode driver: single image or labeled directory tree
├── signal100_W1.csv            hidden-layer weights  (784 × 100)
├── signal100_b1.csv            hidden-layer biases   (100)
├── signal100_W2.csv            output-layer weights  (100 × 10)
├── signal100_b2.csv            output-layer biases   (10)
├── signal30_*.csv              alternative DiNN30 weights (not used by main.cpp by default)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

---

## 3. Build & run

```bash
cd MNIST_signal
cmake -B build -S .
cmake --build build -j4

# Single-image mode: prints scores + plaintext reference + (matches/MISMATCH)
./build/main ../img_1.jpg
# (other samples: ../img_2.jpg … ../img_5.jpg)

# Batch mode: scores every image under <root>/<label>/*.{png,jpg,jpeg},
# prints OK/MISS per image and ends with accuracy + confusion matrix.
./build/main /path/to/test_root
```

Single-image mode also computes a plaintext reference using the same
weights and ends with `(matches)` or `(MISMATCH)` so you can immediately
tell whether the FHE result is correct.

### Bench / accuracy build options

Identical to the rest of the MVB sub-projects — see
[root README §3](../README.md#3-common-build--run-idiom) (bench flags) and
[§4](../README.md#4-batch-accuracy-harness) (batch harness).

---

## 4. DiNN30 vs DiNN100

* **Weights / accuracy.** DiNN100 has roughly 3× the parameters of DiNN30
  and reaches noticeably higher accuracy on MNIST.
* **Runtime.** Layer-1 has 3× more neurons but is still a single matvec at
  the same depth — the per-inference time grows sub-linearly. The
  bootstrap dominates total runtime in both networks.
* **Memory.** Same `ringDim = 2048` and `numSlots = 1024` as DiNN30; peak
  RSS rises only modestly because the weight matrices are stored in
  plaintext.
