.PHONY: install clean build dev lint format sync all add add-dev remove regen-lock list lab help

# Default Python version
PYTHON_VERSION ?= 3.12.8

# Detect platform
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

# Set JAX extras based on platform
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        JAX_PLATFORM = metal
    else
        ifeq ($(UNAME_M),x86_64)
            JAX_PLATFORM = cuda
        else
            $(error Unsupported architecture: $(UNAME_M))
        endif
    endif
else
    JAX_PLATFORM = cuda
endif

print-platform:
	@echo "JAX_PLATFORM: $(JAX_PLATFORM)"

# Install dependencies from lockfile with platform-specific JAX
install:	
	uv sync --extra $(JAX_PLATFORM)

# Install development dependencies
dev: install
	uv sync --extra dev --extra $(JAX_PLATFORM)
	uv run python -m ipykernel install --user --name=jaxpt --display-name "Python $(PYTHON_VERSION) (jaxpt)"

# Regenerate lockfile from scratch
regen-lock:
	rm -f uv.lock
	uv sync --extra $(JAX_PLATFORM) --extra dev

# Add a production dependency (usage: make add pkg=package_name)
add:
	uv add $(pkg)

# Add a development dependency (usage: make add-dev pkg=package_name)
add-dev:
	uv add --dev $(pkg)

# Remove a dependency (usage: make remove pkg=package_name)
remove:
	uv remove $(pkg)

# Clean build artifacts and cache
clean:
	rm -rf uv.lock
	rm -rf build/
	rm -rf dist/
	rm -rf src/jaxpt.egg-info/
	rm -rf .pytest_cache/
	rm -rf src/jaxpt/__pycache__/
	rm -rf src/jaxpt/**/__pycache__/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .venv/
	rm -rf notebooks/.ipynb_checkpoints/
	rm -rf notebooks/**/.ipynb_checkpoints/


# Run linting
lint:
	uv run ruff check src/jaxpt/ --fix

# Format code
format:
	uv run ruff format src/jaxpt/

# Create wheel
wheel:
	uv build --wheel .

# Create source distribution
sdist:
	uv build --sdist .

# Build package
build: wheel sdist

# Show installed packages
list:
	uv pip list

jupyter-ssh-tunnel:
	ssh -L 8888:localhost:8888 -i '${k}' ubuntu@${h}

# Run Jupyter lab
lab:
	cd notebooks && nohup uv run jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10 --NotebookApp.rate_limit_window=10.0 --no-browser --port=8888 > jupyter.log 2>&1 &
	sleep 3
	jupyter server list 

# Help command
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies from lockfile"
	@echo "  make dev       - Install all dependencies including dev from lockfile"
	@echo "  make regen-lock - Regenerate lockfile from scratch"
	@echo "  make add       - Add a production dependency (make add pkg=package_name)"
	@echo "  make add-dev   - Add a development dependency (make add-dev pkg=package_name)"
	@echo "  make remove    - Remove a dependency (make remove pkg=package_name)"
	@echo "  make clean     - Clean build artifacts and cache"
	@echo "  make build     - Build package"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"
	@echo "  make wheel     - Create wheel distribution"
	@echo "  make sdist     - Create source distribution"
	@echo "  make list      - Show installed packages"
	@echo "  make lab       - Run Jupyter lab" 
	@echo "  make jupyter-ssh-tunnel - SSH tunnel to Jupyter lab"
	@echo "  make help      - Show this help message"
