"""
Configuration file for A2A QA/Math Pipeline
Contains all hyperparameters, model settings, and API configurations
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for dir_path in [DATA_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for the primary solver model"""
    model_name: str = "unsloth/Qwen3-4B-Thinking-2507"
    quantization: str = "fp8"  # Options: fp8, 4bit, 8bit, none
    max_model_len: int = 32768
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.5  # Reduced for RTX 4090 with limited free VRAM
    enable_reasoning: bool = True
    reasoning_parser: str = "openai"
    vllm_port: int = 8000

    # Decoding parameters
    temperature_math: float = 0.6
    temperature_qa: float = 0.0
    top_p_math: float = 0.95
    top_p_qa: float = 0.9
    max_tokens_math: int = 32768  # Use model's full capacity for math reasoning
    max_tokens_qa: int = 32768    # Use model's full capacity for QA

    # Self-consistency parameters
    num_samples_math: int = 10
    num_samples_qa: int = 1

@dataclass
class SearchConfig:
    """Configuration for web search and retrieval"""
    # Tavily API
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY", "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")
    max_search_results: int = 20
    search_depth: str = "advanced"  # basic, advanced

    # BGE Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # BGE Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_batch_size: int = 16
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # Hybrid retrieval
    bm25_weight: float = 0.3
    dense_weight: float = 0.7

    # Context limits
    max_context_tokens: int = 8000
    max_passage_tokens: int = 200

@dataclass
class RouterConfig:
    """Configuration for task routing"""
    # Dataset mappings
    math_datasets: List[str] = None
    qa_datasets: List[str] = None
    multihop_datasets: List[str] = None
    multiple_choice_datasets: List[str] = None

    # Keyword-based routing
    math_keywords: List[str] = None
    qa_keywords: List[str] = None

    def __post_init__(self):
        if self.math_datasets is None:
            self.math_datasets = ["gsm8k", "math", "aime", "mathqa"]
        if self.qa_datasets is None:
            self.qa_datasets = ["squad", "nq", "triviaqa"]
        if self.multihop_datasets is None:
            self.multihop_datasets = ["hotpotqa", "2wikimultihopqa"]
        if self.multiple_choice_datasets is None:
            self.multiple_choice_datasets = ["commonsenseqa", "arc", "hellaswag", "piqa"]
        if self.math_keywords is None:
            self.math_keywords = ["calculate", "solve", "equation", "number", "find the value"]
        if self.qa_keywords is None:
            self.qa_keywords = ["who", "what", "where", "when", "which", "how"]

@dataclass
class EvaluationConfig:
    """Configuration for evaluation and judging"""
    # Math evaluation
    use_math_verify: bool = True
    math_verify_timeout: int = 10

    # Official evaluation scripts
    eval_scripts: Dict[str, str] = None

    # LLM Judge (optional)
    judge_model: Optional[str] = None
    judge_temperature: float = 0.0

    # Metrics tracking
    compute_detailed_metrics: bool = True
    save_predictions: bool = True

    def __post_init__(self):
        if self.eval_scripts is None:
            self.eval_scripts = {
                "hotpotqa": "hotpot_evaluate_v1.py",
                "squad": "squad_evaluate.py",
                "nq": "nq_evaluate.py"
            }

@dataclass
class DatasetConfig:
    """Configuration for specific datasets"""
    # Paths to dataset files
    dataset_paths: Dict[str, str] = None

    # Sample sizes for testing (None = full dataset)
    sample_sizes: Dict[str, Optional[int]] = None

    # Answer formats
    answer_formats: Dict[str, str] = None

    def __post_init__(self):
        if self.dataset_paths is None:
            self.dataset_paths = {
                "gsm8k": "datasets/gsm8k_platinum",
                "aime": "datasets/aime_2024",
                "math": "datasets/math_500",
                "hotpotqa": "datasets/hotpotqa_dev",
                "squad_v1": "datasets/squad_v1_dev",
                "squad_v2": "datasets/squad_v2_dev",
                "nq": "datasets/nq_dev"
            }
        if self.sample_sizes is None:
            self.sample_sizes = {
                "gsm8k": None,  # Full dataset
                "aime": None,   # Full dataset (15-25 questions)
                "math": 100,    # Sample for testing
                "hotpotqa": 100,
                "squad_v1": 100,
                "squad_v2": 100,
                "nq": 50
            }
        if self.answer_formats is None:
            self.answer_formats = {
                "math": r"\\boxed\{([^}]+)\}",
                "aime": r"\\boxed\{(\d{1,3})\}",
                "json": r"\{.*\}"
            }

@dataclass
class ExperimentConfig:
    """Configuration for experimental settings"""
    # Reproducibility
    random_seeds: List[int] = None
    num_trials_math: int = 5  # Multiple trials for math tasks

    # Logging
    log_level: str = "INFO"
    log_reasoning: bool = True
    log_search_results: bool = True

    # Performance
    batch_size: int = 4
    num_workers: int = 2

    # Output
    save_intermediate_results: bool = True
    generate_report: bool = True

    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456, 789, 999]

# Main configuration class that combines all configs
@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    search: SearchConfig = None
    router: RouterConfig = None
    evaluation: EvaluationConfig = None
    datasets: DatasetConfig = None
    experiment: ExperimentConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.router is None:
            self.router = RouterConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.datasets is None:
            self.datasets = DatasetConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()

# Global configuration instance
config = Config()

# Validation functions
def validate_config() -> bool:
    """Validate the configuration"""
    errors = []

    # Check API keys
    if config.search.tavily_api_key is None:
        errors.append("TAVILY_API_KEY environment variable not set")

    # Check model settings
    if config.model.quantization not in ["fp8", "4bit", "8bit", "none"]:
        errors.append(f"Invalid quantization: {config.model.quantization}")

    # Check GPU memory
    if config.model.gpu_memory_utilization > 1.0:
        errors.append("GPU memory utilization cannot exceed 1.0")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

def print_config():
    """Print current configuration"""
    print("=== A2A Pipeline Configuration ===")
    print(f"Model: {config.model.model_name}")
    print(f"Quantization: {config.model.quantization}")
    print(f"Search API: {'Tavily' if config.search.tavily_api_key else 'None'}")
    print(f"Math samples: {config.model.num_samples_math}")
    print(f"Random seeds: {config.experiment.random_seeds}")
    print("==================================")