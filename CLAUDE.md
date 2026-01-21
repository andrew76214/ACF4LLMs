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
python run_manual_eval.py --model gpt2 --device cpu --proxy

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

Add model info to `MODEL_SIZE_DATABASE` in `src/coordinator/spec_inference.py`

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
python -c "from src.tools.quantization_wrapper import QuantizationWrapper; print(QuantizationWrapper.get_available_methods())"

# Quick smoke test with mock model
pytest tests/test_basic.py -v

# Check OpenAI API connectivity
python -c "from langchain_openai import ChatOpenAI; m = ChatOpenAI(); print('OK')"
```
