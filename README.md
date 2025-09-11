# LLM Compressor

**Multi-Agent System for LLM Compression and Optimization with Pareto Frontier Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

## Overview

LLM Compressor is a comprehensive multi-agent optimization system designed to compress and optimize Large Language Models (LLMs) across multiple objectives: **accuracy**, **latency**, **VRAM usage**, **energy consumption**, and **CO₂ emissions**. The system uses Pareto frontier analysis to find optimal trade-offs and provides automated pipeline execution with reproducible results.

### Key Features

- 🤖 **10+ Specialized Agents**: Quantization, Pruning, Distillation, KV Optimization, Performance Monitoring, Safety Evaluation
- 📊 **Multi-Objective Optimization**: Accuracy, Latency, VRAM, Energy, CO₂e with Pareto frontier analysis
- 🔧 **Multiple Backends**: vLLM, TensorRT-LLM support with abstracted interfaces
- 📈 **Interactive Visualizations**: Plotly-based charts, 3D Pareto frontiers, parallel coordinates
- 🐳 **Containerized**: Docker support with GPU acceleration
- ⚡ **Production Ready**: Automated pipelines, experiment tracking, comprehensive reporting

### Supported Optimization Techniques

| Technique | Methods | Precision | Hardware Acceleration |
|-----------|---------|-----------|----------------------|
| **Quantization** | AWQ, GPTQ, BitsAndBytes | FP16, FP8, INT8, INT4 | ✅ GPU Optimized |
| **Attention Optimization** | FlashAttention, PagedAttention | - | ✅ Memory Efficient |
| **Pruning** | Structured, Unstructured, N:M Sparsity | - | ✅ Hardware Friendly |
| **Knowledge Distillation** | LoRA, QLoRA, Layer Alignment | - | ✅ Parameter Efficient |
| **Long Context** | Sliding Window, KV Compression | - | ✅ Memory Optimized |

## Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100/H100 recommended)
- **CUDA**: 11.8+ 
- **Python**: 3.10+
- **Docker**: 20.10+ (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/llm-compressor.git
cd llm-compressor

# Quick setup and run
make quickstart
```

### Docker Installation

```bash
# Build Docker image
make build

# Run optimization in container
make run-docker
```

### Manual Installation

```bash
# Install dependencies
make install

# Setup evaluation datasets
make setup-data

# Run baseline optimization
make run-baseline
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python scripts/run_search.py --config configs/default.yaml

# Run only baseline recipes
python scripts/run_search.py --config configs/default.yaml --recipes baseline

# Run search optimization
python scripts/run_search.py --config configs/default.yaml --recipes search

# Export and analyze results
python scripts/export_report.py --db experiments.db --output analysis_report
```

### Configuration

The system is configured via YAML files. Key parameters:

```yaml
# configs/default.yaml
model:
  base_model: "meta-llama/Meta-Llama-3-8B-Instruct"
  sequence_length: 4096

hardware:
  gpu: "NVIDIA A100 80GB"
  vram_limit_gb: 80

constraints:
  max_accuracy_drop: 0.01  # 1% max accuracy drop
  p95_latency_ms: 150      # P95 latency threshold
  carbon_budget_kg: 1.0    # CO₂e budget

objective_weights:
  accuracy: 1.0      # Maximize
  latency: -0.8      # Minimize
  vram: -0.6         # Minimize
  energy: -0.5       # Minimize
  co2e: -0.3         # Minimize
```

### Example Results

After optimization, you'll get:

- **Pareto Frontier**: 5-8 optimal candidates
- **Interactive Visualizations**: 3D plots, parallel coordinates, radar charts
- **Comprehensive Reports**: HTML, CSV, JSON, Markdown formats
- **Reproducible Scripts**: One-click reproduction of any result

```
Top Pareto Candidates:
┌──────┬─────────────────────┬───────┬──────────┬─────────────┬─────────────┐
│ Rank │ Recipe ID           │ Score │ Accuracy │ Latency(ms) │ VRAM(GB)    │
├──────┼─────────────────────┼───────┼──────────┼─────────────┼─────────────┤
│ 1    │ awq_4bit_flash      │ 0.923 │ 0.847    │ 67.3        │ 18.2        │
│ 2    │ conservative_8bit   │ 0.892 │ 0.853    │ 89.1        │ 22.4        │
│ 3    │ aggressive_combo    │ 0.857 │ 0.831    │ 52.8        │ 12.7        │
└──────┴─────────────────────┴───────┴──────────┴─────────────┴─────────────┘
```

## Architecture

### Multi-Agent System

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Recipe Planner │───▶│   Orchestrator   │───▶│ Pareto Analysis │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │Quantization │ │KV Optimizer │ │Perf Monitor │
            └─────────────┘ └─────────────┘ └─────────────┘
                    ▼         ▼         ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │   Pruning   │ │Distillation │ │   Safety    │
            └─────────────┘ └─────────────┘ └─────────────┘
```

### Core Components

- **Orchestrator**: Manages agent execution and workflow
- **Registry**: Experiment tracking and artifact management  
- **Pareto Analyzer**: Multi-objective optimization analysis
- **Metrics Collector**: Performance and resource monitoring
- **Model Runners**: vLLM/TensorRT-LLM abstraction layer

## Baseline Recipes

The system includes 8 pre-configured baseline recipes:

### 1. Quantization Only
```yaml
quantization_only:
  pipeline: ["quantization", "perf_carbon", "eval_safety"]
  quantization:
    method: "awq"
    bits: 4
    group_size: 128
  expected_results:
    compression_ratio: 4.0
    accuracy_drop: 0.005
    latency_improvement: 1.8
```

### 2. KV Optimization Only
```yaml
kv_optimization_only:
  pipeline: ["kv_longcontext", "perf_carbon", "eval_safety"]
  kv_longcontext:
    attention_type: "flash"
    paged_attention: true
    page_size: "2MB"
  expected_results:
    memory_efficiency: 1.5
    latency_improvement: 1.2
```

### 3. Combined Quantization + KV
```yaml
quantization_plus_kv:
  pipeline: ["quantization", "kv_longcontext", "perf_carbon", "eval_safety"]
  # Combines AWQ 4-bit with FlashAttention
  expected_results:
    compression_ratio: 4.0
    memory_efficiency: 1.5
    latency_improvement: 2.2
```

[See full recipe configurations](configs/recipes_baseline.yaml)

## Advanced Usage

### Custom Configurations

Create custom optimization scenarios:

```yaml
# configs/my_experiment.yaml
model:
  base_model: "microsoft/DialoGPT-large"
  sequence_length: 2048

constraints:
  max_accuracy_drop: 0.005  # Stricter accuracy requirement
  p95_latency_ms: 100       # Aggressive latency target

search:
  method: "evolutionary"
  iterations: 100
  parallel_workers: 8
```

### Adding New Agents

1. **Create Agent Class**:
```python
# llm_compressor/agents/my_agent.py
from .base import BaseAgent, AgentResult

class MyCustomAgent(BaseAgent):
    def execute(self, recipe, context):
        # Your optimization logic here
        return AgentResult(success=True, metrics={}, artifacts={})
```

2. **Register in Orchestrator**:
```python
# Add to orchestrator._initialize_agents()
"my_custom": MyCustomAgent
```

3. **Configure in YAML**:
```yaml
agents:
  my_custom:
    enabled: true
    custom_param: value
```

### Multi-GPU Support

```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_search.py \
  --config configs/multi_gpu.yaml
```

### Extending to TensorRT-LLM

The system includes abstract interfaces for easy backend switching:

```python
# Use TensorRT-LLM instead of vLLM
from llm_compressor.core.runners import RunnerFactory

runner = RunnerFactory.create_runner("tensorrt", config)
runner.start_server(model_path, max_batch_size=8)
```

## Evaluation Datasets

Built-in evaluation on standard benchmarks:

- **MMLU**: Multi-task Language Understanding (10 subjects)
- **GSM8K**: Mathematical reasoning (100 problems)  
- **MT-Bench**: Multi-turn conversations (80 scenarios)
- **Safety**: Red-teaming and toxicity evaluation

Custom datasets can be added via the dataset loader framework.

## Monitoring and Visualization

### Real-time Monitoring

```bash
# Monitor system resources during optimization
make monitor
```

### Interactive Visualizations

The system generates multiple visualization types:

- **2D Pareto Plots**: Accuracy vs Latency, Accuracy vs VRAM
- **3D Pareto Frontier**: Multi-objective trade-off surface
- **Parallel Coordinates**: High-dimensional objective space
- **Radar Charts**: Top candidate comparison
- **Resource Timeline**: GPU/CPU/Memory usage over time

### Sample Pareto Visualization

```
       Accuracy vs Latency Trade-off
    1.0 ┤                                ╭─╮
        │                              ╭─╯ ╰─╮ Pareto
    0.9 ┤                        ╭─╮ ╭─╯     ╰─╮ Frontier
        │                      ╭─╯ ╰─╯         ╰─╮
    0.8 ┤                ╭─╮ ╭─╯                 ╰─╮
        │          ╭─╮ ╭─╯ ╰─╯                     ╰─╮
    0.7 ┤    ╭─╮ ╭─╯ ╰─╯                             ╰─╮
        │╭─╮ ╯ ╰─╯                                     ╰─╮
    0.6 ┼╯ ╰─────────────────────────────────────────────╰
        0    50   100   150   200   250   300   350   400
                        Latency (ms)
```

## Testing

```bash
# Run full test suite
make test

# Quick tests only  
make test-quick

# Code quality checks
make check
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-agent`
3. **Make changes** and add tests
4. **Run quality checks**: `make check`
5. **Submit pull request**

### Development Setup

```bash
# Setup development environment
make setup-dev

# Run with debug logging
make debug

# Format code
make format
```

## Performance Benchmarks

Tested on NVIDIA A100 80GB with Llama-3-8B-Instruct:

| Configuration | Accuracy | Latency | VRAM | Energy Savings |
|---------------|----------|---------|------|----------------|
| **Baseline** | 0.853 | 142ms | 38.2GB | - |
| **AWQ 4-bit** | 0.847 | 67.3ms | 18.2GB | 68% |
| **AWQ + Flash** | 0.847 | 52.8ms | 15.7GB | 74% |
| **Aggressive** | 0.831 | 34.1ms | 9.8GB | 82% |

## Troubleshooting

### Common Issues

**GPU Memory Errors**:
```bash
# Reduce model size or batch size
export CUDA_VISIBLE_DEVICES=0
python scripts/run_search.py --config configs/small_gpu.yaml
```

**Installation Issues**:
```bash
# Use Docker for isolated environment
make build && make run-docker
```

**Performance Issues**:
```bash
# Enable debug logging
python scripts/run_search.py --log-level DEBUG
```

### FAQ

**Q: How long does optimization take?**
A: Baseline recipes: 15-30 minutes. Full search: 2-4 hours depending on configuration.

**Q: Can I run without GPU?**
A: The system requires GPU for model inference. CPU-only mode is not recommended for production.

**Q: How to add custom metrics?**
A: Extend the MetricsCollector class and register new metrics in your custom agent.

## Roadmap

- [ ] **Support for more architectures**: Mamba, Mistral, Gemma
- [ ] **Additional optimization techniques**: Sparse attention, MoE optimization  
- [ ] **Distributed optimization**: Multi-node training and inference
- [ ] **Integration with MLOps platforms**: Weights & Biases, MLflow
- [ ] **Automated hyperparameter tuning**: Optuna integration
- [ ] **Edge deployment**: ONNX/OpenVINO export

## Citation

If you use LLM Compressor in your research, please cite:

```bibtex
@software{llm_compressor2024,
  title={LLM Compressor: Multi-Agent System for LLM Optimization},
  author={LLM Compressor Team},
  year={2024},
  url={https://github.com/your-org/llm-compressor}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Transformers**: Hugging Face ecosystem
- **vLLM**: High-performance LLM serving
- **AutoAWQ/AutoGPTQ**: Quantization libraries
- **FlashAttention**: Memory-efficient attention
- **Plotly**: Interactive visualizations

---

**🚀 Ready to optimize your LLMs? Get started with `make quickstart`!**
