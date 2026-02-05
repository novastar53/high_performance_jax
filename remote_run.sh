#!/bin/bash
# remote_run.sh - Run Python scripts on remote machines with log streaming
#
# Usage: ./remote_run.sh [-d] [-n SESSION] [-b BRANCH] <ssh-host> <script-path> [script-args...]
#        ./remote_run.sh <ssh-host> <command>
#
# Options:
#   -d           - Detached mode (run in background)
#   -n SESSION   - Custom tmux session name (default: jax_remote)
#   -b BRANCH    - Git branch or commit to checkout before running
#
# Commands (when no script path provided):
#   attach       - Attach to running tmux session
#   stream       - Stream logs from remote to local
#   status       - Check if script is running
#   stop         - Stop script and download logs
#
# Examples:
#   ./remote_run.sh runpod1 src/high_performance_jax/pallas/pallas_softmax.py
#   ./remote_run.sh -d runpod1 scripts/roofline_attention.py run
#   ./remote_run.sh -d -n my_experiment runpod1 src/high_performance_jax/moe.py
#   ./remote_run.sh -b feature-branch runpod1 src/high_performance_jax/moe.py
#   ./remote_run.sh -b abc123 runpod1 src/high_performance_jax/moe.py
#   ./remote_run.sh runpod1 attach
#   ./remote_run.sh runpod1 status
#   ./remote_run.sh runpod1 stop

set -e

# Configuration
REMOTE_DIR="~/high_performance_jax"
REPO_URL="https://github.com/novastar53/high_performance_jax.git"
DEFAULT_HOST="runpod1"
DEFAULT_SESSION="jax_remote"
LOG_DIR="remote_logs"
REMOTE_LOG_DIR="~/.cache/jax_remote"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse options
DETACH=false
SESSION_NAME="$DEFAULT_SESSION"
GIT_REF=""

while getopts "dn:b:" opt; do
    case $opt in
        d)
            DETACH=true
            ;;
        n)
            SESSION_NAME="$OPTARG"
            ;;
        b)
            GIT_REF="$OPTARG"
            ;;
        \?)
            echo -e "${RED}Invalid option: -$OPTARG${NC}" >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# Parse positional arguments
SSH_HOST=${1:-}
SECOND_ARG=${2:-}

if [ -z "$SSH_HOST" ]; then
    echo -e "${RED}Error: SSH host is required${NC}"
    echo "Usage: $0 [-d] [-n SESSION] [-b BRANCH] <ssh-host> <script-path> [script-args...]"
    echo "   or: $0 <ssh-host> <command>"
    echo ""
    echo "Options:"
    echo "  -d           Detached mode (run in background)"
    echo "  -n SESSION   Custom tmux session name (default: jax_remote)"
    echo "  -b BRANCH    Git branch or commit to checkout before running"
    echo ""
    echo "Commands: attach, stream, status, stop"
    exit 1
fi

# Detect if second argument is a command or script path
COMMANDS="attach|stream|status|stop"
if [[ "$SECOND_ARG" =~ ^($COMMANDS)$ ]]; then
    COMMAND="$SECOND_ARG"
    SCRIPT_PATH=""
    SCRIPT_ARGS=""
else
    COMMAND="run"
    SCRIPT_PATH="$SECOND_ARG"
    shift 2
    SCRIPT_ARGS="$@"
fi

# Generate timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Helper function to run commands on remote
remote_exec() {
    ssh "$SSH_HOST" "$@"
}

# Helper function to get script name from path
get_script_name() {
    local path="$1"
    basename "$path" .py
}

# Helper function to validate script path
validate_script() {
    local script_path="$1"
    
    # Check extension
    if [[ ! "$script_path" =~ \.py$ ]]; then
        echo -e "${RED}Error: Script must be a .py file${NC}"
        exit 1
    fi
    
    # Check if file exists locally (for validation only)
    local local_path="./$script_path"
    if [ ! -f "$local_path" ]; then
        echo -e "${YELLOW}Warning: Script not found locally at $local_path${NC}"
        echo -e "${YELLOW}Will attempt to run on remote anyway...${NC}"
    fi
}

# Helper function to setup remote environment
setup_remote() {
    echo -e "${GREEN}Setting up remote environment...${NC}"
    if [ -n "$GIT_REF" ]; then
        echo -e "${BLUE}Will checkout: $GIT_REF${NC}"
    fi

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

# Expand REMOTE_DIR
REMOTE_DIR_EXPANDED=\$(eval echo ~/high_performance_jax)
PARENT_DIR=\$(dirname "\$REMOTE_DIR_EXPANDED")

# Clone or update main repo
if [ -d "\$REMOTE_DIR_EXPANDED" ]; then
    echo "Updating existing repo..."
    cd "\$REMOTE_DIR_EXPANDED"
    git fetch origin
    git checkout main
    git pull origin main
else
    echo "Cloning repo..."
    mkdir -p "\$PARENT_DIR"
    git clone https://github.com/novastar53/high_performance_jax.git "\$REMOTE_DIR_EXPANDED"
    cd "\$REMOTE_DIR_EXPANDED"
fi

# Checkout specified branch/commit if provided
GIT_REF="$GIT_REF"
if [ -n "\$GIT_REF" ]; then
    echo "Checking out: \$GIT_REF"
    git checkout "\$GIT_REF"
fi

# Clone or update deepkit dependency (required sibling directory)
DEEPKIT_DIR="\$PARENT_DIR/deepkit"
if [ -d "\$DEEPKIT_DIR" ]; then
    echo "Updating deepkit dependency..."
    cd "\$DEEPKIT_DIR"
    git fetch origin
    git pull origin main
else
    echo "Cloning deepkit dependency..."
    mkdir -p "\$PARENT_DIR"
    git clone https://github.com/novastar53/deepkit "\$DEEPKIT_DIR"
fi

# Create log directory
mkdir -p ~/.cache/jax_remote

# Source uv environment
if [ -f "\$HOME/.local/bin/env" ]; then
    source "\$HOME/.local/bin/env"
fi

# Install tmux if needed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y tmux
    elif command -v yum &> /dev/null; then
        sudo yum install -y tmux
    elif command -v apk &> /dev/null; then
        apk add tmux
    else
        echo "Warning: Could not install tmux - unknown package manager"
    fi
fi

# Install dependencies if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "\$HOME/.local/bin/env"
fi

# Install project dependencies
echo "Ensuring dependencies are installed..."
cd "\$REMOTE_DIR_EXPANDED"
uv sync --extra gpu 2>/dev/null || uv sync --extra tpu 2>/dev/null || uv sync

echo "Remote setup complete!"
REMOTE_SCRIPT
}

run_script_foreground() {
    local script_path="$1"
    local script_args="$2"
    local log_file="$REMOTE_LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
    
    echo -e "${GREEN}Running script on $SSH_HOST (foreground mode)...${NC}"
    echo -e "${BLUE}Script: $script_path${NC}"
    echo -e "${BLUE}Args: $script_args${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Create local log directory
    mkdir -p "$LOG_DIR"
    local local_log="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
    
    # Kill existing tmux session if it exists
    remote_exec "tmux kill-session -t $SESSION_NAME 2>/dev/null" || true
    
    # Setup and run script in foreground
    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)
cd "\$REMOTE_DIR_EXPANDED"

# Source uv environment
source \$HOME/.local/bin/env 2>/dev/null

# Run script with tee for logging
echo "Starting: uv run python $script_path $script_args"
uv run python $script_path $script_args 2>&1 | tee $log_file
REMOTE_SCRIPT
    
    # Download log file after completion
    echo ""
    echo -e "${GREEN}Script completed. Downloading logs...${NC}"
    remote_exec "cat $log_file" > "$local_log"
    echo -e "${GREEN}Logs saved to: $local_log${NC}"
}

run_script_background() {
    local script_path="$1"
    local script_args="$2"
    local log_file="$REMOTE_LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
    
    echo -e "${GREEN}Running script on $SSH_HOST (background mode)...${NC}"
    echo -e "${BLUE}Script: $script_path${NC}"
    echo -e "${BLUE}Args: $script_args${NC}"
    echo -e "${BLUE}Session: $SESSION_NAME${NC}"
    
    # Kill existing tmux session if it exists
    remote_exec "tmux kill-session -t $SESSION_NAME 2>/dev/null" || true
    
    # Setup and run script in background
    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)
cd "\$REMOTE_DIR_EXPANDED"

# Source uv environment
source \$HOME/.local/bin/env 2>/dev/null

# Kill existing tmux session
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new tmux session and run script
tmux new-session -d -s $SESSION_NAME -c "\$REMOTE_DIR_EXPANDED"
tmux send-keys -t $SESSION_NAME "source \$HOME/.local/bin/env 2>/dev/null" Enter
tmux send-keys -t $SESSION_NAME "uv run python $script_path $script_args 2>&1 | tee $log_file" Enter

echo "Script started in tmux session '$SESSION_NAME'"
echo "Log file: $log_file"
REMOTE_SCRIPT
    
    echo -e "${GREEN}Script started in background.${NC}"
    echo -e "${YELLOW}Use '$0 $SSH_HOST attach' to attach to the session${NC}"
    echo -e "${YELLOW}Use '$0 $SSH_HOST stream' to stream logs${NC}"
    echo -e "${YELLOW}Use '$0 $SSH_HOST status' to check status${NC}"
    echo -e "${YELLOW}Use '$0 $SSH_HOST stop' to stop and download logs${NC}"
}

cmd_run() {
    if [ -z "$SCRIPT_PATH" ]; then
        echo -e "${RED}Error: Script path is required for 'run' command${NC}"
        exit 1
    fi
    
    validate_script "$SCRIPT_PATH"
    
    # Setup remote environment
    setup_remote
    
    # Run script in foreground or background
    if [ "$DETACH" = true ]; then
        run_script_background "$SCRIPT_PATH" "$SCRIPT_ARGS"
    else
        run_script_foreground "$SCRIPT_PATH" "$SCRIPT_ARGS"
    fi
}

cmd_attach() {
    echo -e "${GREEN}Attaching to tmux session '$SESSION_NAME' on $SSH_HOST...${NC}"
    ssh -t "$SSH_HOST" "tmux attach-session -t $SESSION_NAME"
}

cmd_stream() {
    echo -e "${GREEN}Streaming logs from $SSH_HOST...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop streaming (script will continue)${NC}"
    echo ""
    
    # Create local log directory
    mkdir -p "$LOG_DIR"
    local local_log="$LOG_DIR/${SESSION_NAME}_stream_${TIMESTAMP}.log"
    
    # Get latest log file from remote
    local remote_log=$(remote_exec "ls -t ~/.cache/jax_remote/${SESSION_NAME}_*.log 2>/dev/null | head -1" || echo "")
    
    if [ -z "$remote_log" ]; then
        echo -e "${RED}No log files found for session '$SESSION_NAME'${NC}"
        echo -e "${YELLOW}Use '$0 $SSH_HOST status' to check if script is running${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Streaming log: $remote_log${NC}"
    echo -e "${BLUE}Saving to: $local_log${NC}"
    echo ""
    
    # Stream logs to both terminal and file
    ssh "$SSH_HOST" "tail -f $remote_log" | tee "$local_log"
}

cmd_status() {
    echo -e "${GREEN}Checking status on $SSH_HOST...${NC}"
    echo ""
    
    remote_exec "bash -l" << 'REMOTE_SCRIPT'
# Check if tmux session exists
if tmux has-session -t SESSION_NAME 2>/dev/null; then
    echo -e "${GREEN}Tmux session 'SESSION_NAME': RUNNING${NC}"
else
    echo -e "${RED}Tmux session 'SESSION_NAME': NOT RUNNING${NC}"
fi

# Check if Python process is running
if pgrep -f "uv run python" > /dev/null; then
    echo -e "${GREEN}Python process: RUNNING${NC}"
    echo ""
    echo "Process info:"
    ps aux | grep "[u]v run python" | head -5
else
    echo -e "${RED}Python process: NOT RUNNING${NC}"
fi

# Show log files
echo ""
echo "Log files:"
ls -lt ~/.cache/jax_remote/SESSION_NAME_*.log 2>/dev/null | head -5 || echo "  No logs found"

# Show last few lines of latest log
LATEST_LOG=$(ls -t ~/.cache/jax_remote/SESSION_NAME_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    echo "Last 10 lines of log:"
    tail -10 "$LATEST_LOG"
fi
REMOTE_SCRIPT
}

cmd_stop() {
    echo -e "${GREEN}Stopping script on $SSH_HOST...${NC}"
    
    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

# Check if tmux session exists
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "No tmux session '$SESSION_NAME' found."
    exit 0
fi

# Send Ctrl+C to gracefully stop
echo "Sending interrupt signal to session '$SESSION_NAME'..."
tmux send-keys -t $SESSION_NAME C-c

# Wait for process to exit (with timeout)
echo "Waiting for script to stop..."
for i in {1..30}; do
    if ! pgrep -f "uv run python" > /dev/null; then
        echo "Script stopped."
        break
    fi
    sleep 1
    echo "  Still waiting... ($i)"
done

# Force kill if still running
if pgrep -f "uv run python" > /dev/null; then
    echo "Force killing process..."
    pkill -9 -f "uv run python" || true
fi

# Kill tmux session
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
echo "Tmux session closed."
REMOTE_SCRIPT
    
    # Download logs
    echo ""
    echo -e "${GREEN}Downloading logs...${NC}"
    mkdir -p "$LOG_DIR"
    
    remote_exec "bash -lc 'ls -t ~/.cache/jax_remote/${SESSION_NAME}_*.log 2>/dev/null | head -1'" | while read -r remote_log; do
        if [ -n "$remote_log" ]; then
            local filename=$(basename "$remote_log")
            local local_log="$LOG_DIR/$filename"
            echo -e "${BLUE}Downloading: $remote_log${NC}"
            scp "$SSH_HOST:$remote_log" "$local_log" 2>/dev/null || true
        fi
    done
    
    echo -e "${GREEN}Logs downloaded to: $LOG_DIR${NC}"
    echo -e "${GREEN}Script stopped successfully.${NC}"
}

# Execute the requested command
case "$COMMAND" in
    run)
        cmd_run
        ;;
    attach)
        cmd_attach
        ;;
    stream)
        cmd_stream
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo "Available commands: run, attach, stream, status, stop"
        echo "Or provide a script path to run."
        exit 1
        ;;
esac
