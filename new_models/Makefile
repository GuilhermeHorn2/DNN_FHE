PORT ?= 8888
PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DOCKER_COMPOSE = HOST_UID=$$(id -u) HOST_GID=$$(id -g) PORT=$(PORT) COMPOSE_BAKE=false BUILDKIT_PROGRESS=plain docker compose -f "$(PROJECT_DIR)/docker-compose.yml" --project-directory "$(PROJECT_DIR)"

.PHONY: setup up up-all down logs logs-split build restart clean sh data export-images

# First-time setup: build the image
setup:
	$(DOCKER_COMPOSE) build
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

# Build images
build:
	$(DOCKER_COMPOSE) build

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
