# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=raw \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /workspace

# Strictly the minimum needed to run the notebooks under new_models/notebooks/:
#   - torch + torchvision (CPU): model, training loop, CIFAR-10/100 datasets
#     (train_cifar10.ipynb, train_mnist.ipynb)
#   - tensorflow (CPU): keras model + tensorflow.keras.datasets.mnist
#     (plaintext_training1.ipynb)
#   - numpy: HWC preprocessing and CSV export/reload
#   - matplotlib: training-history plots (loss / accuracy curves)
#   - jupyterlab: notebook server
#
# Each pip install lives in its own RUN so that:
#   1. BuildKit can cache each layer independently. If the proxy / BuildKit
#      gRPC stream dies mid-build, only the failing layer needs to be retried;
#      successfully-installed layers (notably the heavy torch / tensorflow
#      ones) are preserved across rebuilds.
#   2. The pip cache mount keeps already-downloaded wheels on the host
#      builder, so even the *failing* layer resumes almost instantly on
#      the next `make build` invocation.
#
# Recovery from a dropped BuildKit connection is therefore just:
#   make build      # re-run; cached layers + pip cache resume the work

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install tensorflow-cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install numpy matplotlib

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
