# `new_models/` — Parametric trainers for the FHE C++ pipeline

Containerised JupyterLab sandbox that trains DiNN-style integer-quantized
MLPs on **MNIST**, **CIFAR-10**, and **CIFAR-100**, and exports their weights
as plain CSV files (`fmt="%d"`, no header) that the C++ FHE inference projects
in this repo (`MNIST_30/`, `MNIST_100/`, `cifar10/`, …) read out-of-the-box.

Three of the notebooks share a single **parametric template**: pick the
network shape and hidden activation in a `CONFIG` dict at the top, run all
cells, copy the CSVs into a sub-project. A fourth file
(`train_cifar100_baseline.ipynb`) is the original hand-written, fully
narrated worked example kept around as a pedagogical reference.

---

## 1. Quickstart

```bash
cd new_models
make setup     # one-time: build the docker image
make up        # start the jupyter service in the background
make logs s=jupyter   # grab the access URL/token
```

Then in the browser:

1. Open `http://localhost:8888` (token from the log line above).
2. Pick one of `notebooks/train_mnist.ipynb`, `train_cifar10.ipynb`,
   `train_cifar100.ipynb`, or the read-only reference
   `train_cifar100_baseline.ipynb`.
3. Edit the `CONFIG` dict in § 2 (topology, activation, quantization, output
   directory, prefix). Defaults reproduce the artifact shipped with the
   matching C++ sub-project.
4. **Run all cells.** The notebook ends with an *exact-round-trip* assertion
   (§ 11): if it passes, the CSVs you just wrote are bit-for-bit equivalent
   to the in-memory PyTorch model.

Other useful targets: `make data` (downloads MNIST + CIFAR-10 + CIFAR-100
into `./data/`), `make export-images` (writes per-class PNGs into
`<repo>/test_data/<dataset>/<count>/<label>/` so the C++ accuracy harness can
consume them), `make sh` (shell into the running container), `make down`
(stop), `make clean` (stop and drop volumes).

---

## 2. The integer-quantized STE pattern (one paragraph)

Each `nn.Linear` keeps continuous **shadow weights** that Adam updates as
usual; on every forward pass the shadow tensors are projected to integers via
straight-through estimators — **`TernarizeSTE`** for ternary `{-1, 0, +1}`
layers, **`IntQuantSTE`** for layers in `[-max_int, +max_int]`, and
**`RoundSTE`** for biases — so the loss gradient flows through, but the
forward already operates on the *exact* integer weights that will be exported
to CSV. There is no post-training quantization step, no calibration phase,
and no PTQ accuracy cliff: the test accuracy you read off the live PyTorch
model in § 8 is exactly the test accuracy the deployed integer-only C++
inference path computes on the same inputs (verified end-to-end by the
NumPy-vs-PyTorch parity check in § 11).

---

## 3. Notebooks at a glance

| Notebook                                  | Dataset    | Default topology  | Default `prefix`   | Default `output_dir` | Pairs naturally with    |
| ----------------------------------------- | ---------- | ----------------- | ------------------ | -------------------- | ----------------------- |
| `notebooks/train_mnist.ipynb`             | MNIST      | `784 → 30 → 10`   | `dinn30`           | `mnist_dinn30/`      | `MNIST_30/`, `MNIST_100/` *(swap CONFIG to `dinn100` + `[784,100,10]`)* |
| `notebooks/train_cifar10.ipynb`           | CIFAR-10   | `3072 → 30 → 10`  | `cifar10_weights`  | `cifar10/`           | `cifar10/`              |
| `notebooks/train_cifar100.ipynb`          | CIFAR-100  | `3072 → 1024 → 100` | `cifar100_weights` | `cifar100/`        | a future `cifar100/` sub-project |
| `notebooks/train_cifar100_baseline.ipynb` | CIFAR-100  | `3072 → 1024 → 100` *(hardcoded)* | `cifar100_weights` | `cifar100/`  | reference / worked example for the CIFAR-100 case |

All three parametric notebooks have **the same 13-section structure** (title
→ imports → CONFIG → loader → preprocessing → activation factory → MLP →
training → plots → eval → headroom → export → reload check → summary). The
only differences are the dataset loader, the bipolar preprocessing
(grayscale-784 vs. HWC-3072), and the default CONFIG. The baseline notebook
keeps the same section layout but is non-parametric: every value is hard-wired
to the original `3072 → 1024 → 100` sign-activated CIFAR-100 setup.

Hidden-layer activations supported by the parametric notebooks (set
`CONFIG["activation"]` to one of these strings; some take extra
`CONFIG["act_params"]` knobs — see the table in § 5 of any parametric
notebook):

```
sign | heaviside | ternary_act | staircase | hardtanh_q | sigmoid_q | relu_q | square_q
```

---

## 4. CONFIG cookbook

Five ready-to-paste `CONFIG` dicts covering the canonical sub-project
shapes. Drop any of them into the corresponding notebook's § 2 cell, run
all, and the resulting CSVs land in `new_models/<output_dir>/<prefix>_*.csv`.

### 4.1 DiNN30 sign — `train_mnist.ipynb` → `MNIST_30/`

```python
CONFIG = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":   "./data",
    "output_dir": "mnist_dinn30",
    "prefix":     "dinn30",
    "topology":   [784, 30, 10],
    "activation": "sign",
    "act_params": {},
    "layer_quant": [
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "int",     "max_int": 15},
    ],
    "preShift": 256, "pOutput": 1024,
}
```

### 4.2 DiNN100 sign — `train_mnist.ipynb` → `MNIST_100/`

```python
CONFIG = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":   "./data",
    "output_dir": "mnist_dinn100",
    "prefix":     "dinn100",
    "topology":   [784, 100, 10],
    "activation": "sign",
    "act_params": {},
    "layer_quant": [
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "int",     "max_int": 15},
    ],
    "preShift": 256, "pOutput": 1024,
}
```

> **Note.** `MNIST_100/` ships its weights as `dinn100_*_0.csv` (note the
> trailing `_0`). To consume notebook output unmodified, edit
> `MNIST_100/main.cpp` (and `poly_MNIST_100/main.cpp`) to drop the `_0`
> suffix from the four `dinn100_*_0.csv` paths so they accept the standard
> `<prefix>_W{1,2}.csv` / `<prefix>_b{1,2}.csv` pattern emitted here.

### 4.3 MNIST stacked sign — `train_mnist.ipynb` → custom 4-layer MLP

```python
CONFIG = {
    "epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":   "./data",
    "output_dir": "mnist_stacked",
    "prefix":     "mnist_stacked",
    "topology":   [784, 100, 30, 10],   # two hidden layers
    "activation": "sign",
    "act_params": {},
    "layer_quant": [
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "int",     "max_int": 15},
    ],
    "preShift": 256, "pOutput": 1024,
}
```

### 4.4 CIFAR-10 sign baseline — `train_cifar10.ipynb` → `cifar10/`

```python
CONFIG = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":   "./data",
    "output_dir": "cifar10",
    "prefix":     "cifar10_weights",
    "topology":   [3072, 30, 10],
    "activation": "sign",
    "act_params": {},
    "layer_quant": [
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "int",     "max_int": 15},
    ],
    "preShift": 256, "pOutput": 1024,
}
```

### 4.5 CIFAR-100 baseline — `train_cifar100.ipynb` → future `cifar100/`

```python
CONFIG = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":   "./data",
    "output_dir": "cifar100",
    "prefix":     "cifar100_weights",
    "topology":   [3072, 1024, 100],
    "activation": "sign",
    "act_params": {},
    "layer_quant": [
        {"kind": "ternary", "thresh": 0.7},
        {"kind": "int",     "max_int": 15},
    ],
    "preShift": 256, "pOutput": 1024,
}
```

This reproduces the artifact written by `train_cifar100_baseline.ipynb`,
just driven by `CONFIG` instead of hard-coded constants.

---

## 5. Wiring the CSVs into a sub-project

Each parametric notebook ends with a § 12 cell that prints a paste-ready
`Network` builder snippet assembled from `CONFIG`. The mechanical part of
hooking it into a sub-project is a one-liner: copy or symlink the CSVs from
`new_models/<output_dir>/` next to the sub-project's `CMakeLists.txt`, then
build as usual.

```bash
# Example: replace the shipped MNIST_30 weights with a fresh notebook run.
cp new_models/mnist_dinn30/dinn30_W{1,2}.csv \
   new_models/mnist_dinn30/dinn30_b{1,2}.csv \
   MNIST_30/

cd MNIST_30
cmake -B build -S . && cmake --build build -j
./build/main img_1.jpg
```

`MNIST_30/main.cpp` already loads `../dinn30_W{1,2}.csv` and
`../dinn30_b{1,2}.csv`, so dropping the four files into the sub-project
folder is all you need. Use the same recipe (with the matching prefix and
output dir) for `MNIST_100/` (`dinn100_*`), `cifar10/` (`cifar10_weights_*`),
or any future sub-project.
