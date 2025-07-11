#!/bin/bash
"""
MorphiNet Hyperparameter Sweep Execution Script

This script provides a convenient way to launch and manage WandB sweeps for MorphiNet.
It handles sweep creation, agent execution, and provides monitoring capabilities.
"""

set -e  # Exit on error

# Configuration
SWEEP_CONFIG="sweep_config.yaml"
PROJECT_NAME="MorphiNet-Sweep"
AGENT_COUNT=1

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
    if [[ "$CONDA_DEFAULT_ENV" != "morphinet" ]]; then
        print_warning "MorphiNet conda environment not activated"
        print_status "Activating morphinet environment..."
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate morphinet
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
    
    # Start sweep agent
    print_status "Starting sweep agent with $AGENT_COUNT runs..."
    python sweep_agent.py --sweep_id "$SWEEP_ID" --count "$AGENT_COUNT" --project "$PROJECT_NAME"
}

# Function to join existing sweep
join_existing_sweep() {
    if [[ -z "$1" ]]; then
        print_error "Please provide sweep ID"
        exit 1
    fi
    
    SWEEP_ID="$1"
    print_header "Joining Existing Sweep: $SWEEP_ID"
    
    print_status "Starting sweep agent..."
    python sweep_agent.py --sweep_id "$SWEEP_ID" --count "$AGENT_COUNT" --project "$PROJECT_NAME"
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
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  create          Create and run a new hyperparameter sweep"
    echo "  join <sweep_id> Join an existing sweep"
    echo "  status          Show current sweep status"
    echo "  clean           Clean up temporary files"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create                    # Create and run new sweep"
    echo "  $0 join abc123def456         # Join existing sweep"
    echo "  $0 status                    # Show sweep status"
    echo ""
    echo "Configuration:"
    echo "  SWEEP_CONFIG: $SWEEP_CONFIG"
    echo "  PROJECT_NAME: $PROJECT_NAME"
    echo "  AGENT_COUNT: $AGENT_COUNT"
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
            create_and_run_sweep
            ;;
        "join")
            check_prerequisites
            join_existing_sweep "$2"
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