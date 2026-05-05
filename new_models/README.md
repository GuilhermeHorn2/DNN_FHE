# `new_models/` — Trainers for the FHE C++ pipeline

Containerised JupyterLab sandbox that trains DiNN-style integer-quantized
MLPs on **MNIST** and **CIFAR-10**, and exports their weights as plain CSV
files (`fmt="%d"`, no header) that the C++ FHE inference projects in this
repo (`MNIST_30/`, `MNIST_100/`, `cifar10/`) read out-of-the-box.

Both notebooks share a single **parametric template** structurally based on
`plaintext_training1.ipynb`: train a continuous Keras MLP with
`hard_sigmoid` hidden activations on bipolarized inputs, then **post-hoc
discretize** every weight via `round(w · τ) / τ`, then evaluate a strict
`sign(x)` model on the discretized weights, then export the integer copies
`round(W · τ)` as `2 · N` headerless CSVs (plus a companion
`<prefix>_config.json` summary). Pick the topology, hidden activation, `τ`,
and training hyper-parameters in the `CONFIG` dict at the top — defaults
reproduce the artifact shipped with the matching C++ sub-project.

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
2. Pick `notebooks/train_mnist.ipynb` or `notebooks/train_cifar10.ipynb`.
3. Edit the `CONFIG` dict in § 2 (topology, activation, `τ`, training
   hyper-parameters, output directory, prefix). Defaults reproduce the
   artifact shipped with the matching C++ sub-project.
4. **Run all cells.** The export cell writes the weight CSVs **and** a
   companion `<prefix>_config.json` summary alongside them.

Other useful targets: `make data` (downloads MNIST + CIFAR-10 + CIFAR-100
into `./data/`), `make export-images` (writes per-class PNGs into
`<repo>/test_data/<dataset>/<count>/<label>/` so the C++ accuracy harness can
consume them), `make sh` (shell into the running container), `make down`
(stop), `make clean` (stop and drop volumes).

---

## 2. The "train continuous → post-hoc discretize" pipeline

Mirrors `plaintext_training1.ipynb` exactly, structurally:

1. **Build a continuous Keras `Sequential` MLP.** The hidden activation
   (`hard_sigmoid` by default) is a smooth, differentiable proxy for the
   strict `sign(x)` we'll deploy.
2. **Train** with Adam + cross-entropy on bipolarized inputs.
3. **Post-hoc discretize.** Snap every weight to the nearest 1/τ lattice
   point via `round(w · τ) / τ` and push the discretized tensors back into
   the model.
4. **Strict-sign evaluation.** Build a *second* Keras model with `tf.sign`
   between Dense layers, copy the discretized weights into it, and
   evaluate. The accuracy printed here is the cleartext accuracy the FHE
   C++ pipeline will reproduce on the same inputs.
5. **Export.** Write the integer copies `round(W · τ)` and `round(b · τ)`
   as headerless `fmt='%d'` CSVs, plus a JSON config (incl. per-layer
   L1 / L2 column norms — `max(L1)` is the OpenFHE message-space bound `B`
   you should size the plaintext modulus around).

What's parametric vs `plaintext_training1.ipynb`: `topology`,
`hidden_activation`, `tau`, `epochs`, `batch_size`, `validation_split`,
`binarize_threshold`, `prefix`, `output_dir` — all knobs live in `CONFIG`
at the top of the notebook. Setting `topology = [784, 30, 10]` (the
default for `train_mnist.ipynb`) reproduces `plaintext_training1`'s
DiNN-30; setting `[784, 100, 10]` gives DiNN-100; `[784, 100, 30, 10]`
stacks two hidden layers; `tau` widens / shrinks the integer dynamic range.

> **Caveat.** Because discretization is post-hoc (no STE / quantization-
> aware training), expect a small accuracy drop between the continuous
> model and the strict-sign / discretized evaluation. The discretized
> accuracy printed in § 7 is the one the C++ side will see.

---

## 3. Notebooks at a glance

| Notebook                                  | Dataset    | Default topology  | Default `prefix`   | Default `output_dir`    | Pairs naturally with    |
| ----------------------------------------- | ---------- | ----------------- | ------------------ | ----------------------- | ----------------------- |
| `notebooks/train_mnist.ipynb`             | MNIST      | `784 → 30 → 10`   | `dinn30`           | `weights/mnist_dinn30/` | `MNIST_30/`, `MNIST_100/` *(swap CONFIG to `dinn100` + `[784,100,10]`)* |
| `notebooks/train_cifar10.ipynb`           | CIFAR-10   | `3072 → 30 → 10`  | `cifar10_weights`  | `weights/cifar10/`      | `cifar10/`              |
| `notebooks/plaintext_training1.ipynb`     | MNIST      | `784 → 30 → 10` *(hard-coded)* | `dinn30`  | `weights/`              | reference / worked example |

Both parametric notebooks have the same nine-section structure (title →
deps → CONFIG → load + binarize → `build_model` → train → discretize →
strict-sign eval → export + JSON). The only differences are the dataset
loader, the bipolar preprocessing (grayscale-784 vs. HWC-3072), and the
default CONFIG. The reference `plaintext_training1.ipynb` keeps the same
section flow but is non-parametric.

Each parametric notebook's § 8 export cell writes `2 · N` weight CSVs **plus
a single companion `<output_dir>/<prefix>_config.json`** that records the
dataset, topology, hidden activation, `τ`, training hyper-parameters,
final cleartext discretized accuracy, the per-layer L1 / L2 norms (and the
recommended `B = max(L1)`), the list of weight files, the TensorFlow
version, and a timestamp — everything needed to reproduce or audit the run
from the artifact dir alone.

---

## 4. CONFIG cookbook

Four ready-to-paste `CONFIG` dicts covering the canonical sub-project
shapes. Drop any of them into the corresponding notebook's § 2 cell, run
all, and the resulting CSVs land in `new_models/<output_dir>/<prefix>_*.csv`.

### 4.1 DiNN30 — `train_mnist.ipynb` → `MNIST_30/` *(default)*

```python
CONFIG = {
    "dataset":       "MNIST",
    "topology":      [784, 30, 10],
    "hidden_activation": "hard_sigmoid",
    "tau":           10,
    "epochs":        10,
    "batch_size":    128,
    "validation_split": 0.1,
    "binarize_threshold": 128,
    "seed":          0,
    "prefix":        "dinn30",
    "output_dir":    "weights/mnist_dinn30",
}
```

### 4.2 DiNN100 — `train_mnist.ipynb` → `MNIST_100/`

```python
CONFIG = {
    "dataset":       "MNIST",
    "topology":      [784, 100, 10],
    "hidden_activation": "hard_sigmoid",
    "tau":           10,
    "epochs":        15,
    "batch_size":    128,
    "validation_split": 0.1,
    "binarize_threshold": 128,
    "seed":          0,
    "prefix":        "dinn100",
    "output_dir":    "weights/mnist_dinn100",
}
```

> **Note.** `MNIST_100/` ships its weights as `dinn100_*_0.csv` (note the
> trailing `_0`). To consume notebook output unmodified, edit
> `MNIST_100/main.cpp` (and `poly_MNIST_100/main.cpp`) to drop the `_0`
> suffix from the four `dinn100_*_0.csv` paths so they accept the standard
> `<prefix>_W{1,2}.csv` / `<prefix>_b{1,2}.csv` pattern emitted here.

### 4.3 MNIST stacked — `train_mnist.ipynb` → custom 4-layer MLP

```python
CONFIG = {
    "dataset":       "MNIST",
    "topology":      [784, 100, 30, 10],   # two hidden layers
    "hidden_activation": "hard_sigmoid",
    "tau":           10,
    "epochs":        20,
    "batch_size":    128,
    "validation_split": 0.1,
    "binarize_threshold": 128,
    "seed":          0,
    "prefix":        "mnist_stacked",
    "output_dir":    "weights/mnist_stacked",
}
```

### 4.4 CIFAR-10 baseline — `train_cifar10.ipynb` → `cifar10/` *(default)*

```python
CONFIG = {
    "dataset":       "CIFAR10",
    "topology":      [3072, 30, 10],
    "hidden_activation": "hard_sigmoid",
    "tau":           10,
    "epochs":        10,
    "batch_size":    128,
    "validation_split": 0.1,
    "binarize_threshold": 128,
    "seed":          0,
    "prefix":        "cifar10_weights",
    "output_dir":    "weights/cifar10",
}
```

---

## 5. Wiring the CSVs into a sub-project

The mechanical part of hooking notebook output into a C++ sub-project is a
one-liner: copy or symlink the CSVs from `new_models/<output_dir>/` next to
the sub-project's `CMakeLists.txt`, then build as usual. The companion
`<prefix>_config.json` is informational — copy it along too if you want the
sub-project folder to remain self-describing, but the C++ side does not
read it (yet).

```bash
# Example: replace the shipped MNIST_30 weights with a fresh notebook run.
cp new_models/weights/mnist_dinn30/dinn30_W{1,2}.csv \
   new_models/weights/mnist_dinn30/dinn30_b{1,2}.csv \
   new_models/weights/mnist_dinn30/dinn30_config.json \
   MNIST_30/

cd MNIST_30
cmake -B build -S . && cmake --build build -j
./build/main img_1.jpg
```

`MNIST_30/main.cpp` already loads `../dinn30_W{1,2}.csv` and
`../dinn30_b{1,2}.csv`, so dropping the four CSVs into the sub-project
folder is all you need. Use the same recipe (with the matching prefix and
output dir) for `MNIST_100/` (`dinn100_*`) and `cifar10/`
(`cifar10_weights_*`).

Also note `openfhe_params.B_recommended` in the emitted JSON: that is
`max(L1)` across all Linear layers, the recommended message-space bound
for the OpenFHE plaintext modulus.
