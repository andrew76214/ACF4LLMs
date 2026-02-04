# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agentic Compression Framework** - An LLM-driven multi-agent system using LangGraph and GPT-4o to automatically find optimal compression strategies through multi-objective optimization.

Takes minimal input (model name + dataset) and automatically:
- Infers model specifications and hardware requirements
- Uses GPT-4o to autonomously decide compression strategies
- Executes quantization (AutoRound, GPTQ, AWQ, INT8)
- Evaluates on benchmarks (GSM8K, CommonsenseQA, TruthfulQA, HumanEval, BIG-Bench Hard)
- Tracks Pareto-optimal solutions across accuracy, latency, memory, and size

## Build & Run Commands

```bash
# Setup environment
conda create -n greenai python=3.10
conda activate greenai
pip install -r requirements.txt

# Set OpenAI API key (required)
export OPENAI_API_KEY=sk-...

# Optional: GPU quantizers (requires CUDA toolkit with nvcc)
export CUDA_HOME=/usr/local/cuda
pip install -r requirements.quantization.txt

# Run compression optimization
python scripts/run_pipeline.py --model gpt2 --dataset gsm8k --episodes 5

# Run with more episodes and interactive mode
python scripts/run_pipeline.py -m meta-llama/Meta-Llama-3-8B-Instruct -d gsm8k -e 10 -i

# With budget and custom output directory
python scripts/run_pipeline.py -m MODEL -d DATASET -e 10 --budget 4.0 --output-dir ./results

# Show inferred model spec
python scripts/run_pipeline.py --model MODEL_NAME --dataset DATASET --show-spec

# Analyze experiment results
python scripts/run_pipeline.py analyze data/experiments/EXPERIMENT_DIR --format detailed

# Standalone baseline evaluation with energy/CO2 estimation
python scripts/run_manual_eval.py --model gpt2 --device cpu --proxy

# Run tests
python tests/test_basic.py

# Run tests with pytest
pytest tests/

# Run a single test function
pytest tests/test_basic.py::test_spec_inference -v
```

## Architecture

### LangGraph State Machine

```
                    ┌─────────────────┐
                    │     START       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Coordinator    │ ◄─────────────────┐
                    │   (GPT-4o)      │                   │
                    │  分析 Pareto    │                   │
                    │  決定下一步     │                   │
                    └────────┬────────┘                   │
                             │                            │
              ┌──────────────┼──────────────┐             │
              ▼              ▼              ▼             │
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │
     │ Quantization│ │  Evaluation │ │   Search    │     │
     │   Node      │ │   Node      │ │   Node      │     │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘     │
            │               │               │             │
            └───────────────┼───────────────┘             │
                            ▼                             │
                   ┌─────────────────┐                    │
                   │  Update State   │────────────────────┘
                   │  (Pareto更新)    │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      END        │
                   └─────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **LangGraph Coordinator** | `src/coordinator/langgraph_coordinator.py` | Main LLM-driven orchestrator using GPT-4o |
| **State Schema** | `src/coordinator/state.py` | LangGraph state definition and utilities |
| **Spec Inference** | `src/coordinator/spec_inference.py` | Auto-infers model specs from model name |
| **Pareto Frontier** | `src/coordinator/pareto.py` | Multi-objective optimization tracking |
| **Quantization Tools** | `src/agents/quantization_agent.py` | AutoRound, GPTQ, AWQ, INT8, LoRA, QLoRA tools |
| **Evaluation Tools** | `src/agents/evaluation_agent.py` | Benchmark evaluation tools |
| **Search Tools** | `src/agents/search_agent.py` | Bayesian, evolutionary, bandit search |
| **Pruning Tools** | `src/agents/pruning_agent.py` | Magnitude and structured pruning |
| **Quantization Wrappers** | `src/tools/quantization_wrapper.py` | Real quantization implementations |
| **lm-eval Integration** | `src/evaluation/evaluators/lm_eval_evaluator.py` | EleutherAI lm-eval harness wrapper |
| **Schemas** | `src/common/schemas.py` | Pydantic data models |
| **Advanced Config** | `src/common/config.py` | Unified configuration system |

### Episode-Based Workflow

Each optimization runs through multiple episodes controlled by GPT-4o:
1. **Coordinator Decision**: GPT-4o analyzes Pareto frontier and history, decides next action
2. **Quantization**: Apply selected compression method
3. **Evaluation**: Run benchmarks, measure metrics
4. **State Update**: Update Pareto frontier and history
5. **Loop**: Return to coordinator until termination condition met

### Termination Conditions
- Max episodes reached
- Time budget exceeded
- Convergence (5 consecutive episodes with no Pareto improvement)

## Data Models (src/common/schemas.py)

Key enums and models:
- `CompressionMethod`: AUTOROUND, GPTQ, INT8, AWQ, PRUNING, DISTILLATION, LORA, QLORA
- `Benchmark`: GSM8K, COMMONSENSEQA, TRUTHFULQA, HUMANEVAL, BIGBENCH_HARD
- `ModelSpec`: Inferred model specifications
- `CompressionStrategy`: Strategy for each episode
- `EvaluationResult`: Results from compression evaluation

## Experiment Output Structure

```
data/experiments/{experiment_name}/
├── model_spec.json           # Inferred model specification
├── config.json               # Configuration used for this experiment
├── pareto_frontier.json      # Pareto optimal solutions
├── final_results.json        # Summary and best solutions
├── pareto_visualization.html # Interactive chart (open in browser)
└── episode_XXX/
    ├── strategy.json         # Compression strategy tried
    └── results.json          # Evaluation results
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...  # Required for GPT-4o coordinator
HF_TOKEN=hf_...        # Optional: for gated models (Llama, etc.)

# Advanced config via environment (prefix: GREENAI_)
# Format: GREENAI_{SECTION}__{FIELD} (double underscore)
GREENAI_COORDINATOR__LLM_MODEL=gpt-4-turbo
GREENAI_COORDINATOR__LLM_TEMPERATURE=0.5
GREENAI_QUANTIZATION__DEFAULT_BIT_WIDTH=8
GREENAI_EVALUATION__USE_PROXY=false
```

## Advanced Configuration

The framework supports a unified configuration system with YAML/JSON files, environment variables, and CLI overrides.

### Configuration Priority (highest to lowest)
1. CLI arguments (`--llm-model`, `--temperature`, etc.)
2. Environment variables (`GREENAI_COORDINATOR__LLM_MODEL`, etc.)
3. Configuration file (`-c config/default.yaml`)
4. Presets (`--preset balanced`)
5. Default values

### Quick Start

```bash
# Use default configuration
python scripts/run_pipeline.py -m gpt2 -d gsm8k

# Use a preset
python scripts/run_pipeline.py -m gpt2 -d gsm8k --preset latency_focused

# Use config file with CLI overrides
python scripts/run_pipeline.py -m gpt2 -d gsm8k -c config/default.yaml --llm-model gpt-4-turbo

# Environment variable override
GREENAI_COORDINATOR__LLM_MODEL=gpt-4-turbo python scripts/run_pipeline.py -m gpt2 -d gsm8k
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `accuracy_focused` | High accuracy (8-bit, 95% min accuracy, LoRA rank 32) |
| `latency_focused` | Fast inference (4-bit GPTQ, 85% min accuracy) |
| `balanced` | Default balance (4-bit GPTQ, 90% min accuracy) |
| `memory_constrained` | Minimize VRAM (4-bit AWQ, LoRA rank 8) |

### New CLI Options

```bash
--preset, -p           # Preset configuration (accuracy_focused, latency_focused, balanced, memory_constrained)
--llm-model            # LLM model for coordinator (gpt-4o, gpt-4-turbo, etc.)
--temperature          # LLM temperature (0.0-2.0)
--bit-width            # Default quantization bit width (2, 3, 4, 8)
--no-proxy             # Disable proxy evaluation (use full dataset)
```

### Configuration File Structure

See `config/default.yaml` for a fully documented example. Key sections:

```yaml
coordinator:
  llm_model: gpt-4o           # LLM for decisions
  llm_temperature: 0.7        # Decision temperature

quantization:
  default_bit_width: 4        # 2, 3, 4, or 8
  default_method: gptq        # autoround, gptq, awq, int8

evaluation:
  use_proxy: true             # Fast proxy evaluation
  proxy_samples: 200          # Samples for proxy
  measure_carbon: true        # Track CO2 emissions

search:
  method: bandit              # random, bayesian, evolutionary, bandit

reward:
  accuracy_weight: 1.0        # Multi-objective weights
  latency_weight: 0.3
  min_accuracy: 0.9           # Hard constraint

finetuning:
  lora_rank: 16               # LoRA adapter rank

termination:
  max_episodes: 10            # Episode limit
  budget_hours: 2.0           # Time budget
```

### Config Module Location

- **Module**: `src/common/config.py`
- **Default Template**: `config/default.yaml`

### Programmatic Usage

```python
from src.common.config import AdvancedConfig, load_config

# Load from file
config = AdvancedConfig.from_yaml("config/default.yaml")

# Load preset
config = AdvancedConfig.from_preset("latency_focused")

# Load with priority merging
config = load_config(
    config_path="config/custom.yaml",
    preset="balanced",
    cli_overrides={"coordinator": {"llm_model": "gpt-4-turbo"}},
)

# Use with coordinator
from src.coordinator.langgraph_coordinator import LangGraphCoordinator
coordinator = LangGraphCoordinator(
    model_name="gpt2",
    dataset="gsm8k",
    config=config,
)
```

## Development Notes

### Adding New Compression Methods

1. Add wrapper in `src/tools/quantization_wrapper.py`
2. Update tool in `src/agents/quantization_agent.py`
3. Add to `CompressionMethod` enum in `src/common/schemas.py`

### Adding New Benchmarks

1. Create evaluator in `src/evaluation/evaluators/new_evaluator.py`
2. Add to `Benchmark` enum in `src/common/schemas.py`
3. Register in `src/evaluation/benchmark_runner.py`

### Adding New Model Support

Add model info to `MODEL_SIZE_DATABASE` in `src/common/model_utils.py`

### Implementation Status

Current status of real vs mock implementations:

| Component | Status | Notes |
|-----------|--------|-------|
| Coordinator (GPT-4o) | Real | LangGraph + OpenAI |
| Pareto Frontier | Real | Multi-objective optimization |
| Quantization (AutoRound/GPTQ/AWQ/INT8) | Real with fallback | Falls back to mock if libraries unavailable |
| LoRA/QLoRA Fine-tuning | Real with fallback | Uses PEFT library |
| lm-eval Integration | Real | Supports MMLU, HellaSwag, ARC, WinoGrande, etc. |
| BenchmarkRunner | Real | Uses HuggingFace datasets |
| LatencyEvaluator | Real | torch.cuda timing |
| Energy Tracking | Real with fallback | Uses codecarbon, falls back to mock |
| VRAM Estimation | Real with fallback | Uses HF Hub API / model config |
| HumanEval Code Execution | Real | Multiprocessing with timeout |
| Spec Inference | Rule-based | Static model database + regex |
| Search Algorithms | Real | Bayesian, Evolutionary, Multi-Armed Bandit |
| Pruning | Partial | Basic structure, full implementation pending |
| Distillation | Planned | Not yet implemented |

See `TODO.md` for remaining items.

### Git Workflow

- Main branch: `main`
- Development: `andrew_dev`
- Note: Documentation in `SPEC.md` and `README.md` is in Chinese

### Debugging & Verification

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Verify quantization libraries
python -c "from src.tools.quantization_wrapper import get_quantizer; print('Available: autoround, gptq, awq, int8, lora, qlora')"

# Quick smoke test with mock model
pytest tests/test_basic.py -v

# Check OpenAI API connectivity
python -c "from langchain_openai import ChatOpenAI; m = ChatOpenAI(); print('OK')"
```
