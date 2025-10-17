# A2A Pipeline - Quick Start Guide

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- CUDA-compatible GPU with 24GB VRAM
- ~50GB free disk space for models and datasets

### 2. Installation
```bash
# Clone and navigate to project
cd a2a_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages (if needed)
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

Required API keys:
- `TAVILY_API_KEY`: Get from https://tavily.com (free tier available)
- `HF_TOKEN`: Optional, for Hugging Face model downloads

### 4. Validate Setup
```bash
# Check configuration
python src/pipeline.py --config-check

# Run basic tests (after installing full dependencies)
python tests/test_pipeline.py
```

## Usage Examples

### Interactive Mode
```bash
python src/pipeline.py --mode interactive
```
```
[auto] Question: What is 2+2?
Solving...

Answer: 4
Task Type: math
Used Search: No
Confidence: 0.950
Time: 1.23s
```

### Single Question Solving
```bash
# Math problem
python src/pipeline.py --mode solve --question "Find the value of 2^5 + 3^3"

# QA with dataset context
python src/pipeline.py --mode solve --dataset hotpotqa --question "Which university did the author of 'To Kill a Mockingbird' attend?"
```

### Dataset Evaluation
```bash
# Small sample for testing
python src/pipeline.py --mode evaluate --dataset gsm8k --sample-size 10

# Full evaluation with multiple trials (for variance analysis)
python src/pipeline.py --mode evaluate --dataset aime --num-trials 5

# Comprehensive evaluation on all datasets
python src/pipeline.py --mode evaluate --comprehensive
```

## Expected Results

### Math Tasks (with self-consistency)
- **GSM8K**: ~80% accuracy
- **AIME**: ~70% ± 5% (high variance)
- **MATH**: ~30-40% (very challenging)

### QA Tasks
- **SQuAD v1**: ~90% F1
- **HotpotQA**: ~65% Joint EM
- **Natural Questions**: ~35% F1

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **CUDA out of memory**:
   - Reduce batch size in config
   - Use 4-bit quantization instead of FP8
   - Enable CPU offloading

3. **vLLM server fails to start**:
   ```bash
   # Try direct model loading instead
   python src/pipeline.py --use-direct-model
   ```

4. **Math-Verify not working**:
   ```bash
   pip install math-verify
   # Or it will fall back to string matching
   ```

### Performance Tuning

For 24GB VRAM optimization:
```python
# In configs/config.py
config.model.gpu_memory_utilization = 0.9
config.model.quantization = "fp8"  # or "4bit" if needed
config.search.embedding_batch_size = 16  # reduce if needed
```

### API Rate Limits

If hitting Tavily API limits:
- Upgrade to paid tier
- Implement result caching
- Reduce search frequency for batch evaluation

## Development

### Adding New Datasets
1. Add loader in `src/utils/data_processing.py`
2. Add routing rules in `src/agents/router.py`
3. Add evaluation logic in `src/agents/judge.py`
4. Update configuration in `src/configs/config.py`

### Custom Prompts
Edit templates in `src/agents/solver.py`:
```python
class PromptTemplates:
    @staticmethod
    def get_custom_prompt() -> str:
        return """Your custom prompt here..."""
```

### Logging and Debugging
```python
# Enable debug logging
from utils.logging import setup_logger
setup_logger(level="DEBUG")

# View detailed logs
tail -f logs/a2a_pipeline.log
```

## Production Deployment

### Recommended Setup
1. **Docker containerization** for consistent environments
2. **API wrapper** for serving multiple requests
3. **Result caching** to avoid redundant computations
4. **Load balancing** for high-throughput scenarios

### Monitoring
- Track accuracy metrics across datasets
- Monitor GPU memory usage and inference latency
- Log routing decisions and search effectiveness
- Set up alerts for API failures or performance degradation

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run health check: `python src/pipeline.py --config-check`
3. Validate components: `python tests/test_pipeline.py`
4. Review implementation details in `IMPLEMENTATION_REPORT.md`

---

**Ready to solve complex math and QA problems with state-of-the-art reasoning!** 🚀