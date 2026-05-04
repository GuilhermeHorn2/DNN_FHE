# `MNIST_30` — DiNN30 driver on the MVB framework

MVB-based homomorphic inference for an MNIST-style **DiNN30** network:
`784 → 30 → 10`, hard-`Sign` activation, exact look-up-table evaluation
through the schemelet RLWE↔CKKS refresh path.

For the framework internals, the build-flag reference, and `accuracy.cpp`
documentation, see the [**root README**](../README.md).

---

## 1. Topology

| Layer       | Shape       | Notes                                       |
| ----------- | ----------- | ------------------------------------------- |
| Input       | 784 (28×28) | bipolar `{-1, +1}` after grayscale threshold |
| Linear 1    | 784 → 30    | exact integer matvec + bias                 |
| `Sign`      | 30 → 30     | `(x ≥ 0) ? +1 : -1` via MVB LUT             |
| Linear 2    | 30 → 10     | output scores                               |

Crypto context uses the framework defaults from `FHEParams`
(`numSlots = 1024`, `ringDim = 2048`).

---

## 2. Files in this folder

```
MNIST_30/
├── CMakeLists.txt              build script (links ../src + ../include, optionally ../accuracy.cpp)
├── README.md                   you are here
├── main.cpp                    single-image inference driver
├── dinn30_W1.csv               hidden-layer weights  (784 × 30)
├── dinn30_b1.csv               hidden-layer biases   (30)
├── dinn30_W2.csv               output-layer weights  (30  × 10)
├── dinn30_b2.csv               output-layer biases   (10)
└── img_{1..5}.jpg              sample MNIST-style inputs
```

CSV orientation: `LoadCsv2D` accepts both `out × in` and `in × out` and
transposes automatically, so either layout works.

---

## 3. Build & run

```bash
cd MNIST_30
cmake -B build -S .
cmake --build build -j4
./build/main img_1.jpg
# (other samples: img_2.jpg … img_5.jpg)
```

> The `main` driver expects weights at `../dinn30_*.csv` *relative to its
> working directory*, so it must be invoked from `build/` — or just use
> `cd build && ./main ../img_1.jpg`.

The driver also computes a plaintext reference using the same weights and
ends with `(matches)` or `(MISMATCH)` so you can immediately tell whether
the FHE result is correct.

### Bench / accuracy build options

Identical to the rest of the MVB sub-projects — see
[root README §3](../README.md#3-common-build--run-idiom) and
[§4 (`accuracy.cpp`)](../README.md#4-accuracycpp--batch-accuracy-harness).

```bash
cmake -B build -S . -DBENCH_TOTAL=ON -DBENCH_BOOTSTRAP=ON -DBUILD_ACCURACY=ON
cmake --build build -j4
./build/main img_1.jpg
./build/accuracy <test_root>     # weights resolved from ../
```
