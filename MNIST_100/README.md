# `MNIST_100` — DiNN100 driver on the MVB framework

MVB-based homomorphic inference for the wider **DiNN100** MNIST network:
`784 → 100 → 10`, hard-`Sign` activation, exact look-up-table evaluation
through the schemelet RLWE↔CKKS refresh path.

Same pipeline as [`MNIST_30/`](../MNIST_30/README.md), just with a 100-
neuron hidden layer in place of 30. For framework internals, build-flag
reference, and `accuracy.cpp`, see the [**root README**](../README.md).

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
MNIST_100/
├── CMakeLists.txt              build script (links ../src + ../include, optionally ../accuracy.cpp)
├── README.md                   you are here
├── main.cpp                    single-image inference driver
├── dinn100_W1_0.csv            hidden-layer weights  (784 × 100)
├── dinn100_b1_0.csv            hidden-layer biases   (100)
├── dinn100_W2_0.csv            output-layer weights  (100 × 10)
├── dinn100_b2_0.csv            output-layer biases   (10)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

The `_0` suffix on the weight filenames is just how this checkpoint was
serialized — `LoadCsv2D` doesn't care, it transposes either CSV
orientation automatically.

---

## 3. Build & run

```bash
cd MNIST_100
cmake -B build -S .
cmake --build build -j4
cd build
./main ../img_1.jpg
# (other samples: ../img_2.jpg … ../img_5.jpg)
```

The driver also computes a plaintext reference using the same weights and
ends with `(matches)` or `(MISMATCH)` so you can immediately tell whether
the FHE result is correct.

### Bench / accuracy build options

Identical to the rest of the MVB sub-projects — see
[root README §3](../README.md#3-common-build--run-idiom) and
[§4 (`accuracy.cpp`)](../README.md#4-accuracycpp--batch-accuracy-harness).

> `accuracy.cpp` as shipped is wired for **DiNN30** weights
> (`dinn30_*.csv`, `HID_DIM = 30`). To use it from `MNIST_100/` you need
> to flip the three `static constexpr int` lines and the weight filenames
> at the top of `../accuracy.cpp` to match.

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
