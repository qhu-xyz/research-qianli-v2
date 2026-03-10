# --------------- Variables ----------------
PROJECT_NAME := research_template
TOKEN ?= ""
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := . $(VENV_DIR)/bin/activate
BUMP := bump-my-version
TICKET ?= "VER-0000"
PART ?= patch
SHELL := /bin/bash
export UV_CACHE_DIR := $(abspath ./.uv_cache)

# --------------- Phony Targets ----------------
.PHONY: init init-legacy pre-commit sync update lint format precommit clean

# --------------- Commands ----------------

init:
	# @echo "🔧 Init project: $(PROJECT_NAME)..."
	uv venv $(VENV_DIR)
	source .venv/bin/activate
	uv pip install /wheels/dist/numpy*  # install numpy first
	uv pip install /wheels/dist/scipy*
	uv sync --dev --group types --group dev
	@echo "🔧 Setting up pre-commit hooks..."
	uvx pre-commit install
	uvx pre-commit install --hook-type commit-msg
	@echo "✅ To activate virtual environment for $(PROJECT_NAME), run: 'source $(VENV_DIR)/bin/activate'"

init-legacy:
	@echo "🔧 Init project using generic packages (numpy, scipy): $(PROJECT_NAME)..."
	@echo "⚠️ warning: IF YOU ARE USING AMD, USE 'make init' INSTEAD"
	# @echo "🔧 Init project: $(PROJECT_NAME)..."
	uv venv $(VENV_DIR)

	@echo "🔧 Installing project dependencies..."
	uv sync --dev --group types

	@echo "🔧 Setting up pre-commit hooks..."
	uvx pre-commit install
	uvx pre-commit install --hook-type commit-msg

	@echo "✅ To activate virtual environment for $(PROJECT_NAME), run: 'source $(VENV_DIR)/bin/activate'"

pre-commit:
	@echo "🔧 Setting up pre-commit hooks..."
	uvx pre-commit install
	uvx pre-commit install --hook-type commit-msg

sync:
	@echo "🔄 Syncing project dependencies..."
	uv sync --dev --group types --group dev

update:
	@echo "🔄 Updating project dependencies to latest versions..."
	uv sync --update

lint:
	@echo "🔄 Syncing project dependencies for $(PROJECT_NAME)..."
	uv sync --dev --group types
	@echo "🔎 Linting with ruff..."
	uv run ruff check --fix  src tests
	@echo "🔎 Linting with mypy..."
	uv run mypy src tests

format:
	@echo "🎨 Formatting with ruff..."
	uvx ruff format src tests

precommit:
	@echo "🔍 Running pre-commit hooks..."
	uvx pre-commit run --all-files

clean:
	@echo "🧹 Cleaning environment..."
	rm -rf $(VENV_DIR)
