.PHONY: install clean build dev lint format sync all add add-dev remove regen-lock list lab roofline plot-roofline help

# Default Python version
PYTHON_VERSION ?= 3.12.8

# Default SSH host (from ~/.ssh/config)
host ?= runpod1

# Detect platform
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

# Set JAX extras based on platform
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        JAX_PLATFORM = metal
    else ifeq ($(UNAME_M),x86_64)
        ifeq ($(shell command -v nvidia-smi > /dev/null 2>&1 && echo yes),yes)
            JAX_PLATFORM = gpu
        else
            JAX_PLATFORM = tpu
        endif
    else
        $(error Unsupported architecture: $(UNAME_M))
    endif
else ifeq ($(UNAME_S),Linux)
    ifeq ($(shell command -v nvidia-smi > /dev/null 2>&1 && echo yes),yes)
        JAX_PLATFORM = gpu
    else ifeq ($(shell command -v rocminfo > /dev/null 2>&1 && echo yes),yes)
        JAX_PLATFORM = gpu
    else
        JAX_PLATFORM = tpu
    endif
else
    $(error Unsupported operating system: $(UNAME_S))
endif

print-platform:
	@echo "JAX_PLATFORM: $(JAX_PLATFORM)"

# Install dependencies from lockfile with platform-specific JAX
install:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
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

# SSH tunnel for Jupyter (usage: make jupyter-ssh-tunnel [host=runpod1])
jupyter-ssh-tunnel:
	ssh -L 8888:localhost:8888 -L 6006:localhost:6006 $(host)

# SSH tunnel for xprof profiling (usage: make xprof-tunnel [host=runpod1])
xprof-tunnel:
	@echo "Starting SSH tunnel for xprof on port 8791..."
	@echo "Open http://localhost:8791 in your browser"
	ssh -L 8791:localhost:8791 $(host)

# Start xprof server locally (usage: make xprof-serve dir=<trace_path>)
# Example: make xprof-serve dir=traces/2026-01-24/attention_fwd_B4_H8_T1024_D64_flash_attention
xprof-serve:
	@if [ -z "$(dir)" ]; then \
		echo "Available traces:"; \
		find traces -name "*.xplane.pb" -exec dirname {} \; 2>/dev/null | sed 's|/plugins/profile/.*||' | sort -u; \
		echo ""; \
		echo "Usage: make xprof-serve dir=<trace_path>"; \
	else \
		uv run xprof --port 8791 $(dir); \
	fi

# List available traces
xprof-list:
	@uv run python -m high_performance_jax.profiling list

# Download traces from remote machine (usage: make download-traces [host=runpod1])
download-traces:
	@echo "Downloading traces from $(host)..."
	@mkdir -p traces
	scp -r $(host):/root/high_performance_jax/traces/* ./traces/
	@echo "Traces downloaded to ./traces/"
	@echo "View with: make xprof-serve"

# ============================================================================
# NVIDIA Nsight Profiling (for detailed kernel analysis)
# ============================================================================

# Run Nsight Compute on remote (usage: make nsight-compute-remote [host=runpod1] [script=...])
# Profiles kernel-level metrics: memory throughput, occupancy, warp stalls, etc.
nsight-compute-remote:
	@echo "Running Nsight Compute on $(host)..."
	ssh $(host) 'cd /root/high_performance_jax && \
		mkdir -p nsight_profiles && \
		ncu --set full --export nsight_profiles/profile_$$(date +%Y%m%d_%H%M%S) \
		python $(script:-scripts/profile_attention.py)'
	@echo "Profile complete. Download with: make download-nsight"

# Run Nsight Systems on remote (usage: make nsight-systems-remote [host=runpod1] [script=...])
# Profiles system-level timeline: CPU/GPU activity, memory transfers, kernel launches
nsight-systems-remote:
	@echo "Running Nsight Systems on $(host)..."
	ssh $(host) 'cd /root/high_performance_jax && \
		mkdir -p nsight_profiles && \
		nsys profile --stats=true --output=nsight_profiles/timeline_$$(date +%Y%m%d_%H%M%S) \
		python $(script:-scripts/profile_attention.py)'
	@echo "Profile complete. Download with: make download-nsight"

# Download Nsight profiles from remote (usage: make download-nsight [host=runpod1])
download-nsight:
	@echo "Downloading Nsight profiles from $(host)..."
	@mkdir -p nsight_profiles
	scp -r $(host):/root/high_performance_jax/nsight_profiles/* ./nsight_profiles/
	@echo "Profiles downloaded to ./nsight_profiles/"
	@echo "Open .ncu-rep files with: ncu-ui <file>"
	@echo "Open .nsys-rep files with: nsys-ui <file>"

# List Nsight profiles
nsight-list:
	@echo "Local Nsight profiles:"
	@ls -la nsight_profiles/ 2>/dev/null || echo "  No profiles found. Run make nsight-compute-remote or make nsight-systems-remote first."

# Run Jupyter lab
lab:
	cd notebooks && nohup uv run jupyter lab --NotebookApp.iopub_data_rate_limit=1.0e10 --NotebookApp.rate_limit_window=10.0 --no-browser --port=8888 > jupyter.log 2>&1 &
	sleep 3
	uv run jupyter server list

# Generate roofline plot for attention (usage: make roofline [batch=4] [heads=8] [head-dim=64] [seq-lengths=128,256,512,1024,2048,4096])
roofline:
	uv run python scripts/roofline_attention.py

# Generate roofline plots from JSON file (usage: make plot-roofline json=traces/roofline_data_*.json [output-dir=traces])
plot-roofline:
	uv run python scripts/plot_roofline.py $(json) --output-dir=$(output-dir) 

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
	@echo "  make jupyter-ssh-tunnel - SSH tunnel to Jupyter lab [host=runpod1]"
	@echo "  make download-traces - Download traces from remote [host=runpod1]"
	@echo "  make xprof-serve  - Start xprof server locally"
	@echo "  make xprof-list   - List available traces"
	@echo "  make xprof-tunnel - SSH tunnel for xprof [host=runpod1]"
	@echo "  make roofline - Generate roofline plot for attention batch=4 heads=8 head-dim=64 seq-lengths=128,256,512,1024,2048,4096"
	@echo "  make plot-roofline - Generate roofline plots from JSON json=traces/roofline_data_*.json [output-dir=traces]"
	@echo "  make nsight-compute-remote - Run Nsight Compute [host=runpod1]"
	@echo "  make nsight-systems-remote - Run Nsight Systems [host=runpod1]"
	@echo "  make download-nsight - Download Nsight profiles [host=runpod1]"
	@echo "  make nsight-list  - List local Nsight profiles"
	@echo "  make help      - Show this help message"
