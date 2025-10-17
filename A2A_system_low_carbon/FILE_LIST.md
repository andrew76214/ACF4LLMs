# A2A Low Carbon System - Complete File List

**Last Updated**: 2025-10-16
**System Version**: 1.0

This document provides a comprehensive list of all files needed for the A2A (Agent-to-Agent) Low Carbon System, organized by category and purpose.

---

## Table of Contents
1. [Overview](#overview)
2. [Core System Files](#core-system-files)
3. [Evaluation Scripts](#evaluation-scripts)
4. [Test Files](#test-files)
5. [Configuration Files](#configuration-files)
6. [Documentation Files](#documentation-files)
7. [Data Files](#data-files)
8. [Minimal Required Files](#minimal-required-files)
9. [Installation Requirements](#installation-requirements)

---

## Overview

**Total Files**: 57 (excluding venv and cache)
- **Core Source Files**: 20 Python files
- **Evaluation Scripts**: 15 Python scripts
- **Test Files**: 4 Python files
- **Documentation**: 6 Markdown files
- **Configuration**: 3 files
- **Data**: 1 JSON file

---

## Core System Files

### 1. Agent Layer (`src/agents/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/agents/__init__.py` | Agent module initialization | ✅ Yes | ~10 |
| `src/agents/router.py` | Task routing and classification | ✅ Yes | ~350 |
| `src/agents/solver.py` | Answer generation with LLM | ✅ Yes | ~600 |
| `src/agents/judge.py` | Answer evaluation and metrics | ✅ Yes | ~510 |
| `src/agents/search.py` | Web search and retrieval | ⚠️ Optional* | ~540 |

**Note**: `search.py` is optional if you're only doing math tasks without web search.

### 2. Model Layer (`src/models/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/models/__init__.py` | Model module initialization | ✅ Yes | ~10 |
| `src/models/model_manager.py` | Direct vLLM model loading | ✅ Yes | ~400 |
| `src/models/vllm_client.py` | OpenAI-compatible API client | ✅ Yes | ~250 |

### 3. Configuration Layer (`src/configs/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/configs/__init__.py` | Config module initialization | ✅ Yes | ~5 |
| `src/configs/config.py` | All system configurations | ✅ Yes | ~220 |

### 4. Evaluation Layer (`src/evaluation/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/evaluation/__init__.py` | Evaluation module initialization | ✅ Yes | ~5 |
| `src/evaluation/evaluator.py` | Main evaluation orchestrator | ✅ Yes | ~450 |
| `src/evaluation/sequential_evaluator.py` | Sequential evaluation variant | ⚠️ Optional | ~350 |

### 5. Utility Layer (`src/utils/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/utils/__init__.py` | Utils module initialization | ✅ Yes | ~5 |
| `src/utils/math_verify.py` | Symbolic math equivalence | ✅ Yes | ~300 |
| `src/utils/metrics.py` | Evaluation metrics (F1, EM) | ✅ Yes | ~320 |
| `src/utils/text_processing.py` | Text utilities | ✅ Yes | ~200 |
| `src/utils/data_processing.py` | Dataset loading | ✅ Yes | ~250 |
| `src/utils/logging.py` | Logging utilities | ✅ Yes | ~150 |

### 6. Pipeline Layer (`src/`)

| File | Purpose | Essential | Lines |
|------|---------|-----------|-------|
| `src/__init__.py` | Source module initialization | ✅ Yes | ~5 |
| `src/pipeline.py` | High-level pipeline interface | ✅ Yes | ~335 |
| `src/sequential_pipeline.py` | Sequential pipeline variant | ⚠️ Optional | ~300 |

---

## Evaluation Scripts

### Main Evaluation Scripts (Root Directory)

| File | Purpose | Key Features |
|------|---------|--------------|
| **`run_gsm8k_evaluation_lazy.py`** | GSM8K with lazy loading | Memory-efficient, on-demand model load/unload |
| **`run_gsm8k_evaluation.py`** | Standard GSM8K evaluation | Self-consistency, Math-Verify |
| **`run_commonsenseqa_fast.py`** | CommonsenseQA evaluation | Multiple choice, fast inference |
| **`run_hotpotqa_evaluation.py`** | HotpotQA evaluation | Multi-hop search, web retrieval |
| `run_complete_benchmark_evaluation.py` | All benchmarks | Comprehensive suite |
| `run_dataset_evaluation.py` | Generic dataset evaluation | Flexible dataset support |
| `run_commonsenseqa_evaluation.py` | CommonsenseQA (alternative) | Detailed metrics |
| `run_full_gsm8k_evaluation.py` | Full GSM8K suite | Complete test set |
| `run_full_gsm8k_evaluation_with_carbon.py` | GSM8K + carbon tracking | CodeCarbon integration |
| `run_full_gsm8k_with_verified_carbon.py` | GSM8K + verified carbon | Validated emissions |
| `run_hotpotqa_100_simple.py` | HotpotQA sample | 100 questions, simplified |
| `run_hotpotqa_sequential.py` | HotpotQA sequential | Sequential processing |
| `run_quick_evaluation.py` | Quick test | Fast sanity check |
| `run_real_dataset_evaluation.py` | Real dataset test | Production data |
| `run_real_tests.py` | Real system tests | Integration tests |
| `run_sequential_evaluation.py` | Sequential evaluation | Single-threaded |

**Recommended Scripts**:
- **For GSM8K**: `run_gsm8k_evaluation_lazy.py` (best VRAM efficiency)
- **For CommonsenseQA**: `run_commonsenseqa_fast.py`
- **For HotpotQA**: `run_hotpotqa_evaluation.py`
- **For Full Suite**: `run_complete_benchmark_evaluation.py`

---

## Test Files

### Test Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `tests/test_pipeline.py` | Pipeline unit tests | `pytest tests/test_pipeline.py` |
| `test_qwen_model.py` | Model loading test | `python test_qwen_model.py` |
| `test_carbon_tracking.py` | Carbon tracking test | `python test_carbon_tracking.py` |
| `test_tavily_search.py` | Search API test | `python test_tavily_search.py` |

### Utility Scripts

| File | Purpose |
|------|---------|
| `explore_commonsenseqa.py` | Dataset exploration script |
| `setup.py` | Package installation setup |

---

## Configuration Files

### System Configuration

| File | Purpose | Required |
|------|---------|----------|
| **`requirements.txt`** | Python dependencies | ✅ Yes |
| **`.gitignore`** | Git ignore rules | ✅ Yes |
| `.env` (user-created) | Environment variables (API keys) | ✅ Yes |

### Environment Variables (`.env`)

Create this file in the root directory:

```bash
# .env
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here  # Optional
```

---

## Documentation Files

### User Documentation

| File | Purpose | Pages |
|------|---------|-------|
| **`README.md`** | Main documentation | ~500 lines |
| **`CODE_TRACING.md`** | Complete code tracing guide | ~2900 lines |
| **`QUICKSTART.md`** | Quick start guide | ~200 lines |
| `IMPLEMENTATION_REPORT.md` | Implementation details | ~400 lines |
| `COMPREHENSIVE_EVALUATION_REPORT.md` | Evaluation results | ~600 lines |
| `PROJECT_COMPLETE.md` | Project completion report | ~300 lines |

**Essential Documentation**:
- **README.md**: Start here for system overview
- **QUICKSTART.md**: For quick setup and first run
- **CODE_TRACING.md**: For developers who want to understand/extend the code

---

## Data Files

### Dataset Files

| File | Purpose | Size | Source |
|------|---------|------|--------|
| `data/gsm8k_test.json` | GSM8K test set sample | ~1 MB | HuggingFace Datasets |

**Note**: Most datasets are loaded automatically from HuggingFace Datasets. The `data/` directory is for cached or custom datasets.

---

## Minimal Required Files

To run the A2A system with **minimal functionality** (GSM8K evaluation only), you need:

### Absolutely Essential (21 files)

```
A2A_System_Package/
├── requirements.txt                      # Dependencies
├── .env                                  # API keys (user-created)
├── .gitignore                           # Git rules
├── README.md                            # Documentation
│
├── run_gsm8k_evaluation_lazy.py         # Evaluation script
│
└── src/
    ├── __init__.py
    ├── pipeline.py                      # Pipeline
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── router.py                    # Routing
    │   ├── solver.py                    # Solving
    │   └── judge.py                     # Judging
    │
    ├── models/
    │   ├── __init__.py
    │   ├── model_manager.py             # Model loading
    │   └── vllm_client.py               # vLLM client
    │
    ├── configs/
    │   ├── __init__.py
    │   └── config.py                    # Configuration
    │
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluator.py                 # Evaluator
    │
    └── utils/
        ├── __init__.py
        ├── math_verify.py               # Math verification
        ├── metrics.py                   # Metrics
        ├── text_processing.py           # Text utils
        ├── data_processing.py           # Data loading
        └── logging.py                   # Logging
```

**Total**: 21 files (excluding `.env`)

### For Full Functionality (Add These)

```
src/agents/search.py                     # Web search (for HotpotQA)
run_commonsenseqa_fast.py               # CommonsenseQA
run_hotpotqa_evaluation.py              # HotpotQA
```

**Total**: 24 files for full functionality

---

## Installation Requirements

### 1. Python Dependencies (`requirements.txt`)

```txt
# Core
torch>=2.0.0
vllm>=0.3.0
transformers>=4.35.0
datasets>=2.14.0

# Math & Scientific
sympy>=1.12
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Search & Retrieval
tavily-python>=0.3.0
sentence-transformers>=2.2.2
FlagEmbedding>=1.0.0

# Utilities
aiohttp>=3.9.0
tqdm>=4.66.0
python-dotenv>=1.0.0

# Carbon Tracking (Optional)
codecarbon>=2.3.0

# Testing (Optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### 2. System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **GPU**: NVIDIA GPU with 13+ GB VRAM (RTX 3090, RTX 4090, A100, etc.)
- **CUDA**: 11.8 or 12.1
- **RAM**: 32 GB+ recommended
- **Storage**: 20 GB for model + data

### 3. External Dependencies

- **Tavily API**: For web search (HotpotQA)
  - Sign up at https://tavily.com
  - Free tier: 1000 requests/month
  - Set `TAVILY_API_KEY` in `.env`

- **HuggingFace Token** (Optional):
  - For private models or rate limits
  - Create at https://huggingface.co/settings/tokens
  - Set `HUGGINGFACE_TOKEN` in `.env`

---

## File Size Summary

### By Category

| Category | Files | Total Size |
|----------|-------|------------|
| **Core Source** | 20 | ~4.5 MB |
| **Evaluation Scripts** | 15 | ~1.5 MB |
| **Test Files** | 4 | ~0.3 MB |
| **Documentation** | 6 | ~5 MB |
| **Configuration** | 3 | ~50 KB |
| **Data** | 1 | ~1 MB |
| **Total** | 49 | **~12.4 MB** |

**Note**: Model files (~4.19 GB) are downloaded separately from HuggingFace.

---

## Directory Structure

```
A2A_System_Package/
│
├── 📄 Configuration & Setup
│   ├── requirements.txt           # Python dependencies
│   ├── setup.py                   # Package setup
│   ├── .gitignore                # Git ignore rules
│   └── .env                      # Environment variables (create this)
│
├── 📚 Documentation
│   ├── README.md                 # Main documentation
│   ├── CODE_TRACING.md          # Code tracing guide
│   ├── QUICKSTART.md            # Quick start
│   ├── FILE_LIST.md             # This file
│   ├── IMPLEMENTATION_REPORT.md
│   ├── COMPREHENSIVE_EVALUATION_REPORT.md
│   └── PROJECT_COMPLETE.md
│
├── 🚀 Evaluation Scripts
│   ├── run_gsm8k_evaluation_lazy.py         # ⭐ GSM8K (recommended)
│   ├── run_gsm8k_evaluation.py
│   ├── run_commonsenseqa_fast.py            # ⭐ CommonsenseQA
│   ├── run_hotpotqa_evaluation.py           # ⭐ HotpotQA
│   ├── run_complete_benchmark_evaluation.py # ⭐ Full suite
│   ├── run_dataset_evaluation.py
│   ├── run_commonsenseqa_evaluation.py
│   ├── run_full_gsm8k_evaluation.py
│   ├── run_full_gsm8k_evaluation_with_carbon.py
│   ├── run_full_gsm8k_with_verified_carbon.py
│   ├── run_hotpotqa_100_simple.py
│   ├── run_hotpotqa_sequential.py
│   ├── run_quick_evaluation.py
│   ├── run_real_dataset_evaluation.py
│   ├── run_real_tests.py
│   └── run_sequential_evaluation.py
│
├── 🧪 Test Files
│   ├── tests/
│   │   └── test_pipeline.py
│   ├── test_qwen_model.py
│   ├── test_carbon_tracking.py
│   └── test_tavily_search.py
│
├── 📦 Data
│   └── data/
│       └── gsm8k_test.json
│
└── 🎯 Source Code (src/)
    ├── __init__.py
    ├── pipeline.py                      # ⭐ Main pipeline
    ├── sequential_pipeline.py
    │
    ├── agents/                          # ⭐ Agent layer
    │   ├── __init__.py
    │   ├── router.py                    # Task routing
    │   ├── solver.py                    # Answer generation
    │   ├── judge.py                     # Evaluation
    │   └── search.py                    # Web search
    │
    ├── models/                          # ⭐ Model layer
    │   ├── __init__.py
    │   ├── model_manager.py             # Model loading
    │   └── vllm_client.py               # vLLM client
    │
    ├── configs/                         # ⭐ Configuration
    │   ├── __init__.py
    │   └── config.py                    # All configs
    │
    ├── evaluation/                      # ⭐ Evaluation
    │   ├── __init__.py
    │   ├── evaluator.py                 # Main evaluator
    │   └── sequential_evaluator.py
    │
    └── utils/                           # ⭐ Utilities
        ├── __init__.py
        ├── math_verify.py               # Math verification
        ├── metrics.py                   # Metrics
        ├── text_processing.py           # Text utils
        ├── data_processing.py           # Data loading
        └── logging.py                   # Logging

⭐ = Essential files
```

---

## Quick Setup Guide

### 1. Clone/Download Repository

```bash
git clone <your-repo-url>
cd A2A_System_Package
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Create .env file
echo "TAVILY_API_KEY=your_key_here" > .env
```

### 5. Test Installation

```bash
# Quick test
python run_quick_evaluation.py

# Full GSM8K evaluation (5 questions)
PYTHONPATH="." python run_gsm8k_evaluation_lazy.py --limit 5
```

---

## File Dependencies

### Core Dependencies

```
pipeline.py
├── agents/router.py
├── agents/solver.py
│   └── models/model_manager.py
│       └── models/vllm_client.py
├── agents/judge.py
│   └── utils/math_verify.py
├── agents/search.py (optional)
└── evaluation/evaluator.py
    ├── utils/metrics.py
    ├── utils/text_processing.py
    ├── utils/data_processing.py
    └── utils/logging.py

configs/config.py (used by all)
```

### Evaluation Script Dependencies

```
run_gsm8k_evaluation_lazy.py
├── src/pipeline.py
├── src/agents/*
├── src/models/*
├── src/evaluation/*
├── src/utils/*
└── src/configs/config.py
```

---

## Files You Can Safely Remove

If you want to minimize the installation:

### Optional Files (Can Remove)

```
# Alternative evaluation scripts (keep your preferred ones)
run_commonsenseqa_evaluation.py       # Alternative to fast version
run_full_gsm8k_evaluation.py         # Alternative to lazy version
run_full_gsm8k_evaluation_with_carbon.py
run_full_gsm8k_with_verified_carbon.py
run_hotpotqa_100_simple.py           # Sample version
run_hotpotqa_sequential.py           # Alternative version
run_sequential_evaluation.py         # Alternative evaluator

# Optional pipeline
src/sequential_pipeline.py
src/evaluation/sequential_evaluator.py

# Test files (unless you're developing)
tests/test_pipeline.py
test_qwen_model.py
test_carbon_tracking.py
test_tavily_search.py

# Exploration scripts
explore_commonsenseqa.py

# Extra documentation
IMPLEMENTATION_REPORT.md
COMPREHENSIVE_EVALUATION_REPORT.md
PROJECT_COMPLETE.md
```

**Minimal System**: Keep only 21 core files (listed in "Minimal Required Files" section)

---

## Troubleshooting

### Missing Files Error

```bash
ModuleNotFoundError: No module named 'src'
```

**Solution**: Run with `PYTHONPATH`:
```bash
PYTHONPATH="." python run_gsm8k_evaluation_lazy.py
```

### Import Error

```bash
ImportError: cannot import name 'RouterAgent' from 'src.agents'
```

**Solution**: Check that `src/agents/__init__.py` exists (even if empty).

### Configuration Error

```bash
FileNotFoundError: .env file not found
```

**Solution**: Create `.env` file:
```bash
echo "TAVILY_API_KEY=your_key_here" > .env
```

---

## File Checklist

Use this checklist to verify your installation:

### Essential Files (21)

- [ ] `requirements.txt`
- [ ] `.env` (user-created)
- [ ] `README.md`
- [ ] `run_gsm8k_evaluation_lazy.py`
- [ ] `src/__init__.py`
- [ ] `src/pipeline.py`
- [ ] `src/agents/__init__.py`
- [ ] `src/agents/router.py`
- [ ] `src/agents/solver.py`
- [ ] `src/agents/judge.py`
- [ ] `src/models/__init__.py`
- [ ] `src/models/model_manager.py`
- [ ] `src/models/vllm_client.py`
- [ ] `src/configs/__init__.py`
- [ ] `src/configs/config.py`
- [ ] `src/evaluation/__init__.py`
- [ ] `src/evaluation/evaluator.py`
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/math_verify.py`
- [ ] `src/utils/metrics.py`
- [ ] `src/utils/text_processing.py`
- [ ] `src/utils/data_processing.py`
- [ ] `src/utils/logging.py`

### Optional but Recommended

- [ ] `src/agents/search.py` (for HotpotQA)
- [ ] `run_commonsenseqa_fast.py` (for CommonsenseQA)
- [ ] `run_hotpotqa_evaluation.py` (for multi-hop QA)
- [ ] `CODE_TRACING.md` (for developers)
- [ ] `.gitignore` (for version control)

---

## Summary

### What You Need to Run the System

**Minimum**: 21 files + dependencies (~12 MB)
**Recommended**: 26 files (~13 MB)
**Full System**: 49 files (~12.4 MB)

**Plus**:
- Model: Qwen3-4B-Thinking (~4.19 GB, auto-downloaded)
- Python packages: ~2-3 GB (vLLM + PyTorch + dependencies)
- **Total Disk Space**: ~7-8 GB

### Quick Reference

- **Start here**: `README.md`
- **Run GSM8K**: `run_gsm8k_evaluation_lazy.py`
- **Run CommonsenseQA**: `run_commonsenseqa_fast.py`
- **Run HotpotQA**: `run_hotpotqa_evaluation.py`
- **Extend system**: See `CODE_TRACING.md` (Extending the System section)
- **Troubleshoot**: Check `.env`, `PYTHONPATH`, and file paths

---

**Last Updated**: 2025-10-16

For questions or issues, see README.md or check the troubleshooting section above.
