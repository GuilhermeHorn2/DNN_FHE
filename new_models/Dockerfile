FROM python:3.12-slim AS development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=raw

WORKDIR /workspace

# Strictly the minimum needed to run new_models/notebooks/train_cifar100.ipynb:
#   - torch + torchvision (CPU): model, training loop, CIFAR-100 dataset
#   - numpy: HWC preprocessing and CSV export/reload
#   - jupyterlab: notebook server
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --retries 10 --timeout 180 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install --no-cache-dir --retries 10 --timeout 180 \
        numpy \
        jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
