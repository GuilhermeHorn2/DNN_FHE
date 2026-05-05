PORT ?= 8888
PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# DOCKER_BUILDKIT=1 is enabled defensively: the Dockerfile uses BuildKit
# cache mounts (--mount=type=cache) which silently no-op without it.
# COMPOSE_BAKE=false keeps `docker compose build` on a single sequential
# BuildKit session, which gives clean per-layer caching during retries.
DOCKER_COMPOSE = HOST_UID=$$(id -u) HOST_GID=$$(id -g) PORT=$(PORT) DOCKER_BUILDKIT=1 COMPOSE_BAKE=false BUILDKIT_PROGRESS=plain docker compose -f "$(PROJECT_DIR)/docker-compose.yml" --project-directory "$(PROJECT_DIR)"

# How many times `make build` will retry on transient BuildKit failures
# (most commonly `rpc error ... EOF` behind corporate proxies in WSL2).
# Each retry reuses already-built layers + the pip wheel cache mount, so
# retries are very cheap.
BUILD_RETRIES     ?= 5
BUILD_RETRY_DELAY ?= 3

.PHONY: setup up up-all down logs logs-split build restart clean sh data export-images

# First-time setup: build the image
setup: build
	@echo ""
	@echo "Setup complete. Run:"
	@echo "  'make up-all': to start all services."
	@echo "  'make up': to start only the jupyter server."

# Start only the jupyter service
up:
	$(DOCKER_COMPOSE) up -d jupyter
	@echo "Jupyter is running at http://localhost:$(PORT)"
	@echo "Run 'make logs s=jupyter' to see the access token."

# Start all services
up-all:
	$(DOCKER_COMPOSE) up -d
	@echo "Jupyter is running at http://localhost:$(PORT)"
	@echo "Run 'make logs s=jupyter' to see the access token."

# Stop all services
down:
	$(DOCKER_COMPOSE) down

# View logs (all services or specific: make logs s=jupyter)
logs:
	$(DOCKER_COMPOSE) logs -f $(s)

# View logs segmented. TMUX is required.
SERVICES ?= $(shell $(DOCKER_COMPOSE) config --services 2>/dev/null)
logs-split:
	@services="$(SERVICES)"; \
	tmux new-session -d -s logs 2>/dev/null || true; \
	first=1; \
	for svc in $$services; do \
		if [ $$first -eq 1 ]; then \
			tmux send-keys -t logs "$(DOCKER_COMPOSE) logs -f $$svc" Enter; \
			first=0; \
		else \
			tmux split-window -t logs; \
			tmux send-keys -t logs "$(DOCKER_COMPOSE) logs -f $$svc" Enter; \
		fi; \
	done; \
	tmux select-layout -t logs tiled; \
	tmux attach-session -t logs

# Build images.
#
# Wrapped in a retry loop because BuildKit behind a corporate proxy in WSL2
# frequently drops its CLI<->daemon gRPC stream mid-build with:
#   failed to receive status: rpc error: code = Unavailable desc = ... EOF
# The Dockerfile is structured so each pip install is its own (small) layer
# and shares a host-side pip cache mount, so every retry resumes from where
# the previous attempt died -- usually finishing in seconds, not minutes.
#
# Tunables:
#   make build BUILD_RETRIES=10 BUILD_RETRY_DELAY=5
build:
	@n=1; \
	while [ $$n -le $(BUILD_RETRIES) ]; do \
		echo ">>> Build attempt $$n / $(BUILD_RETRIES)..."; \
		if $(DOCKER_COMPOSE) build; then \
			exit 0; \
		fi; \
		echo ""; \
		echo ">>> Attempt $$n failed (likely BuildKit gRPC EOF)."; \
		echo ">>> Cached layers + pip cache will resume on retry."; \
		echo ">>> Sleeping $(BUILD_RETRY_DELAY)s before next attempt..."; \
		echo ""; \
		sleep $(BUILD_RETRY_DELAY); \
		n=$$((n + 1)); \
	done; \
	echo ""; \
	echo "!!! Build failed after $(BUILD_RETRIES) attempts."; \
	echo "!!! Try restarting Docker Desktop, then re-run 'make build'."; \
	exit 1

# Restart all services
restart:
	$(DOCKER_COMPOSE) restart

# Stop and remove volumes
clean:
	$(DOCKER_COMPOSE) down -v

# Open a shell inside the jupyter container
sh:
	$(DOCKER_COMPOSE) exec jupyter bash

# Download MNIST, CIFAR-10 and CIFAR-100 into ./data/ (uses the running jupyter container).
# Usage: make data            -> all three datasets
#        make data ONLY=mnist -> just MNIST (or cifar10 / cifar100)
data:
	$(DOCKER_COMPOSE) exec jupyter python scripts/download_datasets.py \
		--data-dir data \
		$(if $(ONLY),--only $(ONLY))

# Export test PNGs grouped by class into <repo>/test_data/<dataset>/<count>/<label>/
# so the C++ apps (cifar10/build/main, accuracy, ...) can consume them directly.
# The host folder is the repo-root test_data/ (gitignored), bind-mounted at /repo/test_data.
# Usage: make export-images
#        make export-images COUNTS=50,100,500,1000
#        make export-images DATASET=cifar100 COUNTS=200
export-images:
	$(DOCKER_COMPOSE) exec jupyter python scripts/export_test_images.py \
		--data-dir data \
		$(if $(COUNTS),--counts $(COUNTS)) \
		$(if $(DATASET),--dataset $(DATASET))
