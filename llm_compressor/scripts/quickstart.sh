#!/bin/bash
# Quickstart script for LLM Compressor
# Sets up environment and runs basic optimization

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq "3" ] && [ "$PYTHON_MINOR" -ge "10" ]; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
        return 0
    else
        print_warning "nvidia-smi not found, GPU may not be available"
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_warning "requirements.txt not found, installing basic dependencies"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install transformers accelerate datasets
        pip install plotly pandas numpy scikit-learn
        pip install pynvml GPUtil psutil
        pip install pyyaml
    fi
}

# Function to create sample data
create_sample_data() {
    print_status "Creating sample evaluation datasets..."
    
    mkdir -p evals
    
    # Create MMLU subset
    cat > evals/mmlu_subset.jsonl << 'EOF'
{"id": "mmlu_math_1", "subject": "mathematics", "question": "What is the derivative of x^2?", "choices": ["A) 2x", "B) x", "C) 2", "D) x^2"], "answer": "A", "difficulty": "easy"}
{"id": "mmlu_physics_1", "subject": "physics", "question": "What is Newton's first law?", "choices": ["A) F=ma", "B) Objects in motion stay in motion", "C) E=mc^2", "D) F=G(m1*m2)/r^2"], "answer": "B", "difficulty": "medium"}
{"id": "mmlu_chemistry_1", "subject": "chemistry", "question": "What is the chemical symbol for water?", "choices": ["A) H2O", "B) CO2", "C) O2", "D) H2"], "answer": "A", "difficulty": "easy"}
EOF

    # Create GSM8K subset
    cat > evals/gsm8k_subset.jsonl << 'EOF'
{"id": "gsm8k_1", "question": "John has 10 apples. He gives 3 to his friend. How many apples does John have left?", "answer": 7, "solution_steps": ["Step 1: John starts with 10 apples", "Step 2: He gives away 3 apples", "Step 3: 10 - 3 = 7", "Step 4: John has 7 apples left"]}
{"id": "gsm8k_2", "question": "A store sells pencils for $2 each. If Sarah buys 5 pencils, how much does she spend?", "answer": 10, "solution_steps": ["Step 1: Each pencil costs $2", "Step 2: Sarah buys 5 pencils", "Step 3: 5 × $2 = $10", "Step 4: Sarah spends $10"]}
EOF

    # Create MT-Bench subset
    cat > evals/mtbench_subset.jsonl << 'EOF'
{"id": "mtbench_writing_1", "category": "writing", "conversation": [{"role": "user", "content": "Write a creative story about a robot learning to paint."}], "reference_answer": "A creative story about artistic expression and technology.", "scoring_criteria": ["creativity", "coherence", "engagement"]}
{"id": "mtbench_math_1", "category": "math", "conversation": [{"role": "user", "content": "Explain how to solve quadratic equations."}], "reference_answer": "Detailed explanation of quadratic equation solving methods.", "scoring_criteria": ["accuracy", "clarity", "completeness"]}
EOF

    print_success "Sample datasets created"
}

# Function to run basic optimization
run_optimization() {
    local CONFIG_FILE=${1:-"configs/default.yaml"}
    local MODE=${2:-"baseline"}
    
    print_status "Running LLM compression optimization..."
    print_status "Config: $CONFIG_FILE"
    print_status "Mode: $MODE"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        return 1
    fi
    
    # Run optimization
    python scripts/run_search.py \
        --config "$CONFIG_FILE" \
        --recipes "$MODE" \
        --output "reports/quickstart_$(date +%Y%m%d_%H%M%S)" \
        --log-level INFO
    
    if [ $? -eq 0 ]; then
        print_success "Optimization completed successfully!"
        return 0
    else
        print_error "Optimization failed"
        return 1
    fi
}

# Function to show results
show_results() {
    local REPORTS_DIR="reports"
    
    if [ -d "$REPORTS_DIR" ]; then
        print_status "Recent optimization results:"
        ls -la "$REPORTS_DIR" | tail -5
        
        # Find the most recent report directory
        LATEST_REPORT=$(find "$REPORTS_DIR" -name "quickstart_*" -type d | sort | tail -1)
        
        if [ -n "$LATEST_REPORT" ]; then
            print_status "Latest report location: $LATEST_REPORT"
            
            if [ -f "$LATEST_REPORT/summary.md" ]; then
                print_status "Summary report preview:"
                head -20 "$LATEST_REPORT/summary.md"
            fi
        fi
    else
        print_warning "No reports directory found"
    fi
}

# Function to display usage
usage() {
    cat << EOF
LLM Compressor Quickstart Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    setup           Set up environment and dependencies
    baseline        Run baseline recipes only (default)
    search          Run search-based optimization
    full           Run full optimization (baseline + search)
    results        Show recent results
    help           Show this help message

Options:
    --config FILE   Configuration file (default: configs/default.yaml)
    --gpu-check     Check GPU availability before running
    --install-deps  Install dependencies
    --create-data   Create sample datasets

Examples:
    $0 setup                    # Set up environment
    $0 baseline                 # Run baseline optimization
    $0 full --gpu-check        # Run full optimization with GPU check
    $0 --config my_config.yaml baseline  # Use custom config

Environment Variables:
    LLM_COMPRESSOR_GPU         Set to 'false' to disable GPU requirements
    LLM_COMPRESSOR_LOG_LEVEL   Set logging level (DEBUG, INFO, WARNING, ERROR)

EOF
}

# Main execution
main() {
    echo "=========================================="
    echo "       LLM Compressor Quickstart"
    echo "=========================================="
    echo ""
    
    # Parse command line arguments
    COMMAND="baseline"
    CONFIG_FILE="configs/default.yaml"
    GPU_CHECK=false
    INSTALL_DEPS=false
    CREATE_DATA=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            setup)
                COMMAND="setup"
                shift
                ;;
            baseline|search|full|results|help)
                COMMAND="$1"
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --gpu-check)
                GPU_CHECK=true
                shift
                ;;
            --install-deps)
                INSTALL_DEPS=true
                shift
                ;;
            --create-data)
                CREATE_DATA=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Handle help command
    if [ "$COMMAND" = "help" ]; then
        usage
        exit 0
    fi
    
    # Handle results command
    if [ "$COMMAND" = "results" ]; then
        show_results
        exit 0
    fi
    
    # Check Python version
    if ! check_python; then
        exit 1
    fi
    
    # GPU check if requested
    if [ "$GPU_CHECK" = true ] || [ "$LLM_COMPRESSOR_GPU" != "false" ]; then
        check_gpu
    fi
    
    # Handle setup command
    if [ "$COMMAND" = "setup" ]; then
        print_status "Setting up LLM Compressor environment..."
        
        install_dependencies
        create_sample_data
        
        print_success "Setup completed!"
        print_status "You can now run: $0 baseline"
        exit 0
    fi
    
    # Install dependencies if requested
    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
    fi
    
    # Create sample data if requested
    if [ "$CREATE_DATA" = true ]; then
        create_sample_data
    fi
    
    # Check if configuration file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        print_status "Please run: $0 setup"
        exit 1
    fi
    
    # Run the optimization
    case $COMMAND in
        baseline)
            run_optimization "$CONFIG_FILE" "baseline"
            ;;
        search)
            run_optimization "$CONFIG_FILE" "search"
            ;;
        full)
            run_optimization "$CONFIG_FILE" "full"
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
    
    # Show results after successful execution
    if [ $? -eq 0 ]; then
        echo ""
        print_status "=== Execution Summary ==="
        show_results
        echo ""
        print_success "Quickstart completed successfully!"
        print_status "Check the reports directory for detailed results."
    fi
}

# Run main function with all arguments
main "$@"