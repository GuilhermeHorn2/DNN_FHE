# Lightweight image for running scripts/download_test_data.py.
#
# The project root is expected to be mounted at /workspace at runtime
# (see scripts/docker-compose.yml). The script writes its outputs to
# ./data and ./data/_downloads, so mounting the project root means the
# downloaded MNIST/CIFAR-10 PNGs end up directly in <repo>/data on the
# host.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Install any corporate root/intermediate CAs found in ./corp-ca/ so
# pip / urllib / requests trust the BCN proxy's MITM certificate. The
# directory is optional and host-specific (it is .gitignored); on
# machines without a corporate proxy it can be empty and this step is
# a no-op.
COPY corp-ca/ /usr/local/share/ca-certificates/extra/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Default working directory matches the volume mount point so the
# script's relative `data/` output path lands in the host repo root.
WORKDIR /workspace

ENTRYPOINT ["python", "scripts/download_test_data.py"]
CMD []
