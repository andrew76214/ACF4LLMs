"""Data schemas for the Agentic Compression Framework."""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class CompressionMethod(str, Enum):
    """Available compression methods."""
    AUTOROUND = "autoround"
    GPTQ = "gptq"
    INT8 = "int8"
    AWQ = "awq"
    ASVD = "asvd"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    LORA = "lora"
    QLORA = "qlora"


class Benchmark(str, Enum):
    """Available benchmark datasets.

    Benchmarks are evaluated using EleutherAI's lm-eval harness where possible,
    with custom evaluators for specialized tasks (e.g., HumanEval code execution).
    """

    # Core benchmarks (lm-eval)
    GSM8K = "gsm8k"
    COMMONSENSE_QA = "commonsenseqa"
    TRUTHFUL_QA = "truthfulqa"
    HUMANEVAL = "humaneval"  # Custom evaluator (code execution)
    BIGBENCH_HARD = "bigbench_hard"

    # Additional lm-eval benchmarks
    MMLU = "mmlu"
    HELLASWAG = "hellaswag"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    WINOGRANDE = "winogrande"


class ModelSpec(BaseModel):
    """Model specification inferred from model name."""
    model_name: str = Field(..., description="Original model name")
    model_size_gb: float = Field(..., description="Model size in GB")
    model_family: str = Field(..., description="Model family (llama, mistral, etc.)")
    parameter_count: Optional[str] = Field(None, description="Parameter count (e.g., '8B', '70B')")
    preferred_methods: List[CompressionMethod] = Field(
        default_factory=list,
        description="Recommended compression methods for this model"
    )
    calibration_samples: int = Field(512, description="Number of calibration samples")
    max_sequence_length: int = Field(4096, description="Maximum sequence length")

    # Hardware constraints
    min_vram_gb: float = Field(..., description="Minimum VRAM required")
    recommended_vram_gb: float = Field(..., description="Recommended VRAM")

    # Budget and objectives
    primary_objective: str = Field("accuracy", description="Primary optimization objective")
    accuracy_threshold: float = Field(0.95, description="Minimum acceptable accuracy")
    target_speedup: float = Field(2.0, description="Target speedup factor")
    target_memory_reduction: float = Field(0.5, description="Target memory reduction factor")


class CompressionStrategy(BaseModel):
    """A compression strategy to be executed."""
    episode_id: int = Field(..., description="Episode number")
    strategy_id: str = Field(..., description="Unique strategy identifier")
    methods: List[CompressionMethod] = Field(..., description="Compression methods to apply")

    # Method-specific parameters
    quantization_bits: Optional[int] = Field(None, description="Bit width for quantization")
    quantization_method: Optional[str] = Field(None, description="Specific quantization method")
    pruning_ratio: Optional[float] = Field(None, description="Pruning sparsity ratio")
    pruning_method: Optional[str] = Field(None, description="Pruning method: magnitude or structured")
    pruning_granularity: Optional[str] = Field(None, description="Pruning granularity: weight, channel, or head")
    distillation_teacher: Optional[str] = Field(None, description="Teacher model for distillation")
    lora_rank: Optional[int] = Field(None, description="LoRA rank for fine-tuning")
    asvd_rank_ratio: Optional[float] = Field(None, description="ASVD rank ratio for SVD compression (0.0-1.0)")

    # Execution parameters
    do_finetune: bool = Field(False, description="Whether to fine-tune after compression")
    finetune_steps: Optional[int] = Field(None, description="Number of fine-tuning steps")
    calibration_dataset: Optional[str] = Field(None, description="Dataset for calibration")

    created_at: datetime = Field(default_factory=datetime.now)


class EvaluationResult(BaseModel):
    """Results from evaluating a compressed model."""
    strategy_id: str = Field(..., description="Strategy that produced this result")
    checkpoint_path: str = Field(..., description="Path to compressed model checkpoint")

    # Model metrics
    model_size_gb: float = Field(..., description="Compressed model size in GB")
    compression_ratio: float = Field(..., description="Size reduction ratio")

    # Performance metrics (weighted average across benchmarks)
    accuracy: float = Field(..., description="Average accuracy across benchmarks")
    latency_ms: float = Field(..., description="Average inference latency in ms")
    throughput_tokens_per_sec: float = Field(..., description="Inference throughput")
    memory_gb: float = Field(..., description="Peak memory usage during inference")

    # Energy/Carbon metrics
    energy_joules: Optional[float] = Field(None, description="Energy consumption per inference in Joules")
    energy_kwh: Optional[float] = Field(None, description="Total energy consumed in kWh")
    co2_grams: Optional[float] = Field(None, description="CO2 emissions in grams")
    co2_kg: Optional[float] = Field(None, description="CO2 emissions in kg")
    num_inferences_measured: Optional[int] = Field(None, description="Number of inferences for energy measurement")
    is_carbon_mock: bool = Field(False, description="Whether carbon measurement is mocked")

    # Per-benchmark results
    benchmark_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy scores per benchmark"
    )

    # Metadata
    is_pareto_optimal: bool = Field(False, description="Whether this solution is on Pareto frontier")
    evaluation_time_sec: float = Field(..., description="Time taken for evaluation")
    evaluated_at: datetime = Field(default_factory=datetime.now)


class ParetoSolution(BaseModel):
    """A solution on the Pareto frontier."""
    strategy: CompressionStrategy
    result: EvaluationResult
    dominates: List[str] = Field(
        default_factory=list,
        description="List of strategy IDs this solution dominates"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "strategy": self.strategy.model_dump(),
            "result": self.result.model_dump(),
            "dominates": self.dominates
        }


class Episode(BaseModel):
    """An episode in the compression optimization process."""
    episode_id: int
    strategy: CompressionStrategy
    result: Optional[EvaluationResult] = None
    status: Literal["planning", "executing", "evaluating", "completed", "failed"] = "planning"
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def duration_seconds(self) -> Optional[float]:
        """Calculate episode duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class RewardConfig(BaseModel):
    """Configuration for the multi-objective reward function."""
    accuracy_weight: float = Field(1.0, description="Weight for accuracy")
    latency_weight: float = Field(0.3, description="Weight for latency (lower is better)")
    memory_weight: float = Field(0.3, description="Weight for memory usage (lower is better)")
    energy_weight: float = Field(0.1, description="Weight for energy consumption (lower is better)")

    # Hard constraints
    min_accuracy: float = Field(0.9, description="Minimum acceptable accuracy")
    max_latency_ms: Optional[float] = Field(None, description="Maximum acceptable latency")
    max_memory_gb: Optional[float] = Field(None, description="Maximum memory usage")


class ProxyEvaluationConfig(BaseModel):
    """Configuration for proxy evaluation."""
    enabled: bool = Field(True, description="Whether to use proxy evaluation")
    sample_ratio: float = Field(0.1, description="Ratio of dataset to use for proxy")
    max_samples: int = Field(1000, description="Maximum samples for proxy evaluation")
    confidence_threshold: float = Field(0.95, description="Confidence threshold for proxy results")


class SearchConfig(BaseModel):
    """Configuration for strategy search."""
    method: Literal["random", "bayesian", "evolutionary", "bandit"] = "random"
    exploration_ratio: float = Field(0.2, description="Exploration vs exploitation ratio")
    mutation_rate: float = Field(0.1, description="Mutation rate for evolutionary search")
    population_size: int = Field(10, description="Population size for evolutionary search")