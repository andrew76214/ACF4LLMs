# Quick Start Guide

Get started with the Agentic Compression Framework in 5 minutes!

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Conda or pip for package management

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n greenai python=3.10
conda activate greenai

# Install dependencies
pip install -r requirements.txt

# Optional GPU quantizers (requires CUDA + CUDA_HOME)
export CUDA_HOME=/usr/local/cuda  # Adjust if CUDA lives elsewhere
pip install -r requirements.quantization.txt
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional GPU quantizers (requires CUDA + CUDA_HOME)
export CUDA_HOME=/usr/local/cuda  # Adjust if CUDA lives elsewhere
pip install -r requirements.quantization.txt
```

## Your First Compression

### 1. Test with a Small Model (GPT-2)

Start with GPT-2 to verify everything works:

```bash
# Run with mock implementation (no GPU needed)
python scripts/run_pipeline.py --model gpt2 --dataset gsm8k --mock --episodes 3
```

Expected output:
- 3 compression strategies tested
- Pareto frontier with optimal solutions
- Results saved to `data/experiments/`

### 2. View Model Specification

See what the framework infers about your model:

```bash
python scripts/run_pipeline.py --model meta-llama/Meta-Llama-3-8B-Instruct --dataset commonsenseqa --show-spec
```

This shows:
- Model size and family
- Hardware requirements
- Recommended compression methods
- Optimization objectives

### 3. Run Real Compression (GPU Required)

For actual model compression:

```bash
# Install the GPU-backed quantizers (if not already done)
export CUDA_HOME=/usr/local/cuda  # Ensure this points to your CUDA toolkit
pip install -r requirements.quantization.txt

# Run compression optimization
python scripts/run_pipeline.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset gsm8k \
  --episodes 5 \
  --interactive
```

> ⚠️ `gptqmodel` builds CUDA extensions (Cutlass, Triton, etc.). Make sure the
CUDA toolkit (with `nvcc`) is installed locally and that `CUDA_HOME` points to
its root (e.g., `/usr/local/cuda` or the path from `which nvcc`). If the build
fails, retry with `pip install -v gptqmodel --no-build-isolation`.

### 4. Analyze Results

After optimization completes:

```bash
# View summary
python scripts/run_pipeline.py analyze data/experiments/your_experiment_dir

# View detailed results
python scripts/run_pipeline.py analyze data/experiments/your_experiment_dir --format detailed
```

## Common Use Cases

### Memory-Constrained Deployment

Optimize for smallest model size:

```bash
python scripts/run_pipeline.py \
  --model your-model \
  --dataset your-dataset \
  --episodes 10 \
  --experiment-name "minimize_size"
```

### Latency-Critical Applications

Optimize for fastest inference:

```bash
python scripts/run_pipeline.py \
  --model your-model \
  --dataset your-dataset \
  --episodes 10 \
  --experiment-name "minimize_latency"
```

### Balanced Optimization

Find the best trade-off:

```bash
python scripts/run_pipeline.py \
  --model your-model \
  --dataset your-dataset \
  --episodes 15 \
  --experiment-name "balanced"
```

## Understanding Results

### Pareto Frontier

The framework finds solutions that are optimal across multiple objectives:

- **Accuracy**: Model performance on benchmarks
- **Latency**: Inference speed (ms)
- **Memory**: Peak memory usage (GB)
- **Size**: Compressed model size (GB)

A solution is "Pareto optimal" if no other solution is better in all objectives.

### Output Files

```
data/experiments/your_experiment/
├── model_spec.json           # Your model's specifications
├── pareto_frontier.json      # All Pareto optimal solutions
├── final_results.json        # Summary and best solutions
├── pareto_visualization.html # Interactive chart (open in browser)
└── episode_001/
    ├── strategy.json         # What compression was tried
    └── results.json          # How well it performed
```

### Visualization

Open the HTML visualization to see:
- Accuracy vs Latency trade-offs
- Accuracy vs Model Size trade-offs
- Interactive exploration of solutions

```bash
# Open in browser
firefox data/experiments/your_experiment/pareto_visualization.html
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'deepagents'"

DeepAgents is optional for MVP. Use `--mock` flag:

```bash
python scripts/run_pipeline.py --model gpt2 --dataset gsm8k --mock
```

### "CUDA out of memory"

Try these solutions:
1. Use smaller batch size in evaluation
2. Use CPU instead: `--device cpu`
3. Start with 8-bit quantization instead of 4-bit

### "Model not found"

Ensure you're using the correct HuggingFace model name:
- ✅ `meta-llama/Meta-Llama-3-8B-Instruct`
- ❌ `llama3-8b`

Check available models at: https://huggingface.co/models

## Next Steps

1. **Explore Different Models**: Try various model families
2. **Adjust Episodes**: More episodes = better optimization
3. **Custom Benchmarks**: Add your own evaluation datasets
4. **Production Deployment**: Use the best checkpoint in your application

## Getting Help

- Check the [full documentation](../README.md)
- Open an issue on GitHub
- Review example experiments in `scripts/examples/`

Happy compressing! 🚀
