#!/bin/bash
# MorphiNet Hyperparameter Sweep Execution Script
#
# This script provides a convenient way to launch and manage WandB sweeps for MorphiNet.
# It handles sweep creation, agent execution, and provides monitoring capabilities.

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/conda.env" ]]; then
    set -a
    source "$PROJECT_ROOT/conda.env"
    set +a
fi

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-morphinet}"

SWEEP_CONFIG="sweep_config.yaml"
PROJECT_NAME="MorphiNet-Sweep"
AGENT_COUNT=5  # Number of sequential runs (parameter combinations to try)
# Note: These runs execute SEQUENTIALLY, not concurrently, to avoid OOM

# Memory optimization settings to prevent OOM
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_MODULE_LOADING=LAZY

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if conda environment is activated
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]]; then
        print_warning "MorphiNet conda environment not activated"
        print_status "Activating $CONDA_ENV_NAME environment..."
        source "$CONDA_ROOT/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
    fi
    
    # Check if WandB is installed
    if ! python -c "import wandb" 2>/dev/null; then
        print_error "WandB is not installed. Please install it with: pip install wandb"
        exit 1
    fi
    
    # Check if sweep config exists
    if [[ ! -f "$SWEEP_CONFIG" ]]; then
        print_error "Sweep configuration file not found: $SWEEP_CONFIG"
        exit 1
    fi
    
    # Check if GPU is available
    if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_warning "GPU not available. Training will be very slow."
    fi
    
    print_status "Prerequisites check completed"
}

# Function to create and start sweep
create_and_run_sweep() {
    local run_count="${1:-$AGENT_COUNT}"  # Allow override via argument
    
    print_header "Creating and Running Hyperparameter Sweep"
    
    # Create sweep and capture sweep ID
    print_status "Creating sweep from configuration: $SWEEP_CONFIG"
    SWEEP_ID=$(python sweep_agent.py --create_sweep --sweep_config "$SWEEP_CONFIG" --project "$PROJECT_NAME" | grep "Created sweep with ID:" | cut -d' ' -f5)
    
    if [[ -z "$SWEEP_ID" ]]; then
        print_error "Failed to create sweep"
        exit 1
    fi
    
    print_status "Sweep created with ID: $SWEEP_ID"
    
    # Save sweep ID for reference
    echo "$SWEEP_ID" > sweep_id.txt
    print_status "Sweep ID saved to sweep_id.txt"
    
    # Start sweep agent with specified run count
    print_status "Starting sweep agent with $run_count sequential runs..."
    print_status "Each run will try a different parameter combination from Bayesian optimization"
    
    # Use process isolation to prevent memory accumulation
    if [[ $run_count -gt 1 ]]; then
        print_status "Using process isolation to prevent OOM (running $run_count separate agents)"
        for ((i=1; i<=run_count; i++)); do
            print_status "Running parameter combination $i of $run_count..."
            python sweep_agent.py --sweep_id "$SWEEP_ID" --count 1 --project "$PROJECT_NAME"
            if [[ $? -ne 0 ]]; then
                print_error "Run $i failed, but continuing with remaining runs..."
            fi
        done
    else
        # Single run - use direct agent
        python sweep_agent.py --sweep_id "$SWEEP_ID" --count "$run_count" --project "$PROJECT_NAME"
    fi
}

# Function to join existing sweep
join_existing_sweep() {
    if [[ -z "$1" ]]; then
        print_error "Please provide sweep ID"
        exit 1
    fi
    
    local sweep_id="$1"
    local run_count="${2:-$AGENT_COUNT}"  # Allow override via second argument
    
    print_header "Joining Existing Sweep: $sweep_id"
    
    print_status "Starting sweep agent with $run_count sequential runs..."
    
    # Use process isolation for multiple runs
    if [[ $run_count -gt 1 ]]; then
        print_status "Using process isolation to prevent OOM (running $run_count separate agents)"
        for ((i=1; i<=run_count; i++)); do
            print_status "Running parameter combination $i of $run_count..."
            python sweep_agent.py --sweep_id "$sweep_id" --count 1 --project "$PROJECT_NAME"
            if [[ $? -ne 0 ]]; then
                print_error "Run $i failed, but continuing with remaining runs..."
            fi
        done
    else
        python sweep_agent.py --sweep_id "$sweep_id" --count "$run_count" --project "$PROJECT_NAME"
    fi
}

# Function to run multiple combinations on existing sweep
run_multiple_combinations() {
    local run_count="${1:-10}"  # Default 10 combinations
    
    if [[ ! -f "sweep_id.txt" ]]; then
        print_error "No sweep ID file found. Create a sweep first with: $0 create"
        exit 1
    fi
    
    SWEEP_ID=$(cat sweep_id.txt)
    print_header "Running $run_count Parameter Combinations on Sweep: $SWEEP_ID"
    
    print_status "Each combination will run sequentially to avoid OOM issues"
    print_status "Bayesian optimization will suggest optimal parameter combinations"
    
    python sweep_agent.py --sweep_id "$SWEEP_ID" --count "$run_count" --project "$PROJECT_NAME"
}

# Function to show sweep status
show_sweep_status() {
    if [[ -f "sweep_id.txt" ]]; then
        SWEEP_ID=$(cat sweep_id.txt)
        print_header "Sweep Status: $SWEEP_ID"
        print_status "WandB URL: https://wandb.ai/$(whoami)/$PROJECT_NAME/sweeps/$SWEEP_ID"
    else
        print_warning "No sweep ID file found. Have you created a sweep yet?"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION] [COUNT]"
    echo ""
    echo "Options:"
    echo "  create [count]           Create and run a new hyperparameter sweep"
    echo "  join <sweep_id> [count]  Join an existing sweep"
    echo "  multi [count]            Run multiple combinations on existing sweep"
    echo "  status                   Show current sweep status"
    echo "  clean                    Clean up temporary files"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create                    # Create and run $AGENT_COUNT combinations"
    echo "  $0 create 10                 # Create and run 10 combinations"
    echo "  $0 join abc123def456         # Join existing sweep with $AGENT_COUNT runs"
    echo "  $0 join abc123def456 15      # Join existing sweep with 15 runs"
    echo "  $0 multi 20                  # Run 20 more combinations on existing sweep"
    echo "  $0 status                    # Show sweep status"
    echo ""
    echo "Key Points:"
    echo "  • All runs execute SEQUENTIALLY to avoid OOM errors"
    echo "  • Bayesian optimization selects parameter combinations"
    echo "  • Each run tries different hyperparameter values"
    echo "  • Results are logged to WandB for comparison"
    echo ""
    echo "Configuration:"
    echo "  SWEEP_CONFIG: $SWEEP_CONFIG"
    echo "  PROJECT_NAME: $PROJECT_NAME"
    echo "  DEFAULT_COUNT: $AGENT_COUNT"
}

# Function to clean up
clean_up() {
    print_header "Cleaning Up"
    
    files_to_remove=("sweep_id.txt" "wandb/")
    
    for file in "${files_to_remove[@]}"; do
        if [[ -e "$file" ]]; then
            rm -rf "$file"
            print_status "Removed: $file"
        fi
    done
    
    print_status "Cleanup completed"
}

# Main execution
main() {
    case "${1:-help}" in
        "create")
            check_prerequisites
            create_and_run_sweep "$2"  # Pass optional count
            ;;
        "join")
            check_prerequisites
            join_existing_sweep "$2" "$3"  # Pass sweep_id and optional count
            ;;
        "multi")
            check_prerequisites
            run_multiple_combinations "$2"  # Pass optional count
            ;;
        "status")
            show_sweep_status
            ;;
        "clean")
            clean_up
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Execute main function
main "$@"