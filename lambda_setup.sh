git config --global user.email "pvikram035@gmail.com"
git config --global user.name "Vikram Pawar"
export TERM=xterm-256color

# Detect and export GPU model if available
if command -v nvidia-smi &> /dev/null; then
    export GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
    echo "Detected GPU: $GPU_MODEL"
else
    echo "No NVIDIA GPU detected (nvidia-smi not found)"
    export GPU_MODEL="none"
fi

rm -rf .venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
make regen-lock
make install
make dev
