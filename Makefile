# Makefile for LLM Compressor
# Multi-agent system for LLM compression and optimization

.PHONY: help install build run clean test lint format docker setup-dev

# Variables
PYTHON := python3
PIP := pip
DOCKER_IMAGE := llm-compressor
DOCKER_TAG := latest
CONFIG_FILE := configs/default.yaml
OUTPUT_DIR := reports

# Default target
help:
	@echo "LLM Compressor - Multi-agent optimization system"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  build        Build Docker image"
	@echo "  run          Run optimization with default config"
	@echo "  run-baseline Run baseline recipes only"
	@echo "  run-docker   Run in Docker container"
	@echo "  clean        Clean up generated files"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code"
	@echo "  setup-dev    Setup development environment"
	@echo "  export       Export and analyze results"
	@echo "  quickstart   Quick setup and run"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG_FILE=$(CONFIG_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  DOCKER_IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)"

# Installation
install:
	@echo "Installing LLM Compressor dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Installation completed!"

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install pytest pytest-cov black isort flake8 mypy
	@echo "Development installation completed!"

# Build
build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# Run targets
run:
	@echo "Running LLM compression optimization..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --output $(OUTPUT_DIR)

run-baseline:
	@echo "Running baseline recipes only..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes baseline --output $(OUTPUT_DIR)/baseline

run-search:
	@echo "Running search optimization only..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes search --output $(OUTPUT_DIR)/search

run-docker:
	@echo "Running in Docker container..."
	docker run --gpus all -it --rm \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/evals:/app/evals \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		python3 scripts/run_search.py --config $(CONFIG_FILE) --output reports/docker

run-interactive:
	@echo "Running interactive Docker session..."
	docker run --gpus all -it --rm \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/evals:/app/evals \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		/bin/bash

# Export and analysis
export:
	@echo "Exporting and analyzing results..."
	$(PYTHON) scripts/export_report.py --db experiments.db --output analysis_report

export-json:
	@echo "Exporting from JSON summary..."
	$(PYTHON) scripts/export_report.py --input $(OUTPUT_DIR)/execution_summary.json --output analysis_report

# Testing
test:
	@echo "Running test suite..."
	$(PYTHON) -m pytest tests/ -v --cov=llm_compressor --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick tests..."
	$(PYTHON) -m pytest tests/ -v -x --tb=short

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 llm_compressor/ scripts/ --max-line-length=100 --ignore=E203,W503
	mypy llm_compressor/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black llm_compressor/ scripts/ --line-length=100
	isort llm_compressor/ scripts/ --profile black

format-check:
	@echo "Checking code formatting..."
	black llm_compressor/ scripts/ --line-length=100 --check
	isort llm_compressor/ scripts/ --profile black --check-only

# Development setup
setup-dev: install-dev
	@echo "Setting up development environment..."
	@mkdir -p reports artifacts logs evals
	@echo "Creating sample evaluation data..."
	@bash scripts/quickstart.sh --create-data
	@echo "Development environment ready!"

setup-data:
	@echo "Setting up sample datasets..."
	@mkdir -p evals
	@bash scripts/quickstart.sh --create-data
	@echo "Sample datasets created!"

# Quickstart
quickstart:
	@echo "Running quickstart setup and optimization..."
	@bash scripts/quickstart.sh setup
	@bash scripts/quickstart.sh baseline

quickstart-docker: build
	@echo "Running quickstart in Docker..."
	docker run --gpus all -it --rm \
		-v $(PWD)/reports:/app/reports \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		bash scripts/quickstart.sh baseline

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	rm -rf reports/* artifacts/* logs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup completed!"

clean-docker:
	@echo "Cleaning up Docker resources..."
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker system prune -f

clean-all: clean clean-docker
	@echo "Full cleanup completed!"

# Development utilities
check: format-check lint test-quick
	@echo "All checks passed!"

validate-config:
	@echo "Validating configuration..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --dry-run

# Environment information
env-info:
	@echo "Environment Information:"
	@echo "========================"
	@$(PYTHON) --version
	@$(PIP) --version
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "CUDA: $$(nvcc --version 2>/dev/null | grep 'release' || echo 'Not available')"

# Performance benchmarking
benchmark:
	@echo "Running performance benchmark..."
	$(PYTHON) scripts/run_search.py \
		--config configs/default.yaml \
		--recipes baseline \
		--output reports/benchmark_$$(date +%Y%m%d_%H%M%S) \
		--log-level INFO

# Documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	@echo "# LLM Compressor Documentation\n" > docs/index.md
	@echo "Auto-generated documentation would go here..." >> docs/index.md

# Custom recipes
run-custom:
	@if [ -z "$(RECIPE)" ]; then \
		echo "Usage: make run-custom RECIPE=path/to/recipe.yaml"; \
		exit 1; \
	fi
	@echo "Running custom recipe: $(RECIPE)"
	$(PYTHON) scripts/run_search.py --config $(RECIPE) --output $(OUTPUT_DIR)/custom

# Multi-GPU support
run-multi-gpu:
	@echo "Running with multiple GPUs..."
	CUDA_VISIBLE_DEVICES=0,1 $(PYTHON) scripts/run_search.py \
		--config $(CONFIG_FILE) \
		--output $(OUTPUT_DIR)/multi_gpu

# Monitoring and debugging
debug:
	@echo "Running in debug mode..."
	$(PYTHON) scripts/run_search.py \
		--config $(CONFIG_FILE) \
		--log-level DEBUG \
		--output $(OUTPUT_DIR)/debug

monitor:
	@echo "Monitoring system resources..."
	watch -n 5 'nvidia-smi; echo ""; ps aux | grep python | head -10'

# CI/CD targets
ci-test: install-dev format-check lint test
	@echo "CI tests completed!"

ci-build: build
	@echo "CI build completed!"

# Help for specific targets
help-docker:
	@echo "Docker Commands:"
	@echo "  make build         Build Docker image"
	@echo "  make run-docker    Run optimization in container"
	@echo "  make run-interactive  Start interactive container"
	@echo "  make clean-docker  Remove Docker image"

help-dev:
	@echo "Development Commands:"
	@echo "  make setup-dev     Setup development environment"
	@echo "  make test          Run full test suite"
	@echo "  make lint          Run code linting"
	@echo "  make format        Format code"
	@echo "  make check         Run all quality checks"