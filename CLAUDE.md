# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Compressor is a multi-agent system for optimizing Large Language Models across multiple objectives: accuracy, latency, VRAM usage, energy consumption, and CO₂ emissions. The system uses Pareto frontier analysis to find optimal trade-offs between these objectives.

## Core Architecture

### Multi-Agent System Structure
- **Orchestrator** (`llm_compressor/core/orchestrator.py`): Main coordinator that manages agent execution and workflow
- **Registry** (`llm_compressor/core/registry.py`): Experiment tracking and artifact management
- **Pareto Analyzer** (`llm_compressor/core/pareto.py`): Multi-objective optimization analysis
- **Metrics Collector** (`llm_compressor/core/metrics.py`): Performance and resource monitoring

### Agent Types
Located in `llm_compressor/agents/`:
- `quantization.py`: AWQ, GPTQ, BitsAndBytes quantization methods
- `kv_longcontext.py`: FlashAttention, PagedAttention optimizations
- `pruning_sparsity.py`: Structured/unstructured pruning and N:M sparsity
- `distillation.py`: Knowledge distillation with LoRA/QLoRA
- `perf_carbon.py`: Performance monitoring and carbon footprint measurement
- `eval_safety.py`: Safety evaluation and red-teaming
- `recipe_planner.py`: Pipeline planning and recipe generation
- `search.py`: Bayesian and evolutionary optimization

### Model Runners
- Abstract interface in `llm_compressor/core/runners.py` supports both vLLM and TensorRT-LLM backends
- Default backend is vLLM with GPU acceleration

## Common Development Commands

### Installation and Setup
```bash
# Quick setup and run
make quickstart

# Manual installation
make install
make setup-data

# Development setup
make setup-dev
```

### Running Optimization
```bash
# Default optimization
make run
python llm_compressor/scripts/run_search.py --config llm_compressor/configs/default.yaml

# Baseline recipes only
make run-baseline
python llm_compressor/scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes baseline

# Search optimization
make run-search
python llm_compressor/scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes search

# Export results
make export
python llm_compressor/scripts/export_report.py --db experiments.db --output analysis_report
```

### Testing and Quality Assurance
```bash
# Full test suite
make test
python -m pytest tests/ -v --cov=llm_compressor --cov-report=html --cov-report=term

# Quick tests
make test-quick
python -m pytest tests/ -v -x --tb=short

# Linting
make lint
flake8 llm_compressor/ --max-line-length=100 --ignore=E203,W503
mypy llm_compressor/ --ignore-missing-imports

# Code formatting
make format
black llm_compressor/ --line-length=100
isort llm_compressor/ --profile black

# All quality checks
make check
```

### Docker Operations
```bash
# Build Docker image
make build

# Run in Docker
make run-docker

# Interactive Docker session
make run-interactive
```

## Configuration System

### Main Configuration
- Primary config: `llm_compressor/configs/default.yaml`
- Baseline recipes: `llm_compressor/configs/recipes_baseline.yaml`
- Model-specific configs available (e.g., `gemma3_270m.yaml`)

### Key Configuration Sections
- `model`: Base model settings, sequence length
- `hardware`: GPU type, VRAM limits, parallelization
- `constraints`: Accuracy drop limits, latency thresholds, carbon budget
- `agents`: Individual agent configurations and parameters
- `search`: Optimization method (bayesian, evolutionary, grid, random)
- `evaluation`: MMLU, GSM8K, MT-Bench, Safety benchmarks

## Agent Development

### Adding New Agents
1. Inherit from `BaseAgent` in `llm_compressor/agents/base.py`
2. Implement the `execute()` method returning `AgentResult`
3. Register in orchestrator's `_initialize_agents()`
4. Add configuration in YAML files

### Agent Result Structure
```python
@dataclass
class AgentResult:
    success: bool
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
```

## Hardware Requirements
- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100/H100 recommended)
- **CUDA**: 11.8+
- **Python**: 3.10+
- **Docker**: 20.10+ (optional)

## Evaluation Datasets
Built-in support for:
- **MMLU**: Multi-task Language Understanding
- **GSM8K**: Mathematical reasoning
- **MT-Bench**: Multi-turn conversations
- **Safety**: Red-teaming and toxicity evaluation

## Monitoring and Debugging
```bash
# Debug mode with verbose logging
make debug
python llm_compressor/scripts/run_search.py --log-level DEBUG

# System monitoring
make monitor

# Validate configuration
make validate-config
```

## Optimization Techniques
- **Quantization**: AWQ (4-bit), GPTQ, BitsAndBytes (INT8/FP16)
- **Attention**: FlashAttention, PagedAttention for memory efficiency  
- **Pruning**: Structured/unstructured, N:M sparsity patterns
- **Distillation**: LoRA, QLoRA with temperature scaling
- **KV Optimization**: Sliding window, compression techniques

## Key Dependencies
- PyTorch ≥ 2.1.0 with CUDA support
- Transformers ≥ 4.36.0
- Quantization: auto-gptq, autoawq, bitsandbytes
- Serving: vLLM ≥ 0.3.0 (default) or TensorRT-LLM
- Optimization: scipy, optuna, DEAP
- Visualization: plotly, matplotlib, seaborn
- Monitoring: pynvml, GPUtil, psutil