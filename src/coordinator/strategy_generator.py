"""Strategy generator for creating compression strategies based on history and objectives."""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import uuid4
import logging

from src.common.schemas import CompressionStrategy, CompressionMethod, EvaluationResult
from src.reward.reward_function import MultiObjectiveReward, RewardConfig

logger = logging.getLogger(__name__)


def _estimate_model_size_gb(model_name: str) -> float:
    """Estimate model size in GB from model name."""
    import re

    MODEL_SIZE_DATABASE = {
        "gpt2": 0.5, "gpt2-medium": 1.5, "gpt2-large": 3.0, "gpt2-xl": 6.0,
        "facebook/opt-125m": 0.25, "facebook/opt-350m": 0.7, "facebook/opt-1.3b": 2.6,
        "facebook/opt-2.7b": 5.4, "facebook/opt-6.7b": 13.4, "facebook/opt-13b": 26.0,
        "meta-llama/Meta-Llama-3-8B": 16.0, "meta-llama/Llama-2-7b-hf": 13.5,
        "mistralai/Mistral-7B-v0.1": 13.5, "Qwen/Qwen-7B": 14.0,
    }

    if model_name in MODEL_SIZE_DATABASE:
        return MODEL_SIZE_DATABASE[model_name]

    model_lower = model_name.lower()
    for key, size in MODEL_SIZE_DATABASE.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return size

    patterns = [
        (r'(\d+\.?\d*)B', lambda x: float(x) * 2),
        (r'(\d+\.?\d*)b', lambda x: float(x) * 2),
        (r'(\d+)M', lambda x: float(x) / 1000 * 2),
        (r'(\d+)m', lambda x: float(x) / 1000 * 2),
    ]

    for pattern, converter in patterns:
        match = re.search(pattern, model_name)
        if match:
            return converter(match.group(1))

    return 1.0


class StrategyGenerator:
    """Generate compression strategies based on objectives and history."""

    def __init__(
        self,
        model_spec: Dict[str, Any],
        reward_function: Optional[MultiObjectiveReward] = None,
    ):
        """Initialize strategy generator.

        Args:
            model_spec: Model specification from spec inference
            reward_function: Reward function for scoring strategies
        """
        self.model_spec = model_spec
        self.reward_function = reward_function or MultiObjectiveReward()
        self.history = []
        self.episode_count = 0

    def generate_next_strategy(
        self,
        episode_id: int,
        previous_results: Optional[List[EvaluationResult]] = None,
        exploration_rate: float = 0.2,
    ) -> CompressionStrategy:
        """Generate the next compression strategy.

        Args:
            episode_id: Current episode number
            previous_results: Results from previous episodes
            exploration_rate: Probability of exploration vs exploitation

        Returns:
            Next compression strategy to try
        """
        self.episode_count = episode_id

        # Determine strategy based on episode phase
        if episode_id <= 3:
            # Early exploration phase
            strategy = self._generate_exploration_strategy(episode_id)
        elif episode_id <= 7:
            # Refinement phase
            strategy = self._generate_refinement_strategy(previous_results)
        else:
            # Optimization phase
            strategy = self._generate_optimization_strategy(previous_results)

        # Add exploration with probability
        if random.random() < exploration_rate:
            strategy = self._add_exploration_noise(strategy)

        self.history.append(strategy)
        return strategy

    def _generate_exploration_strategy(self, episode_id: int) -> CompressionStrategy:
        """Generate exploratory strategies for early episodes."""
        strategies = [
            # Episode 1: Conservative quantization
            {
                "methods": [CompressionMethod.INT8],
                "quantization_bits": 8,
                "quantization_method": "int8",
                "rationale": "Conservative start with INT8 quantization",
            },
            # Episode 2: Moderate quantization
            {
                "methods": [CompressionMethod.GPTQ],
                "quantization_bits": 4,
                "quantization_method": "gptq",
                "rationale": "Test 4-bit quantization with GPTQ",
            },
            # Episode 3: Alternative method
            {
                "methods": [CompressionMethod.AUTOROUND],
                "quantization_bits": 4,
                "quantization_method": "autoround",
                "rationale": "Try AutoRound for better accuracy retention",
            },
        ]

        if episode_id <= len(strategies):
            strategy_params = strategies[episode_id - 1]
        else:
            # Random exploration
            strategy_params = self._generate_random_strategy()

        return CompressionStrategy(
            episode_id=episode_id,
            strategy_id=str(uuid4())[:8],
            **strategy_params
        )

    def _generate_refinement_strategy(
        self,
        previous_results: Optional[List[EvaluationResult]],
    ) -> CompressionStrategy:
        """Generate refinement strategies based on previous results."""
        if not previous_results:
            return self._generate_random_strategy()

        # Find best performing method so far
        best_result = max(previous_results, key=lambda r: r.accuracy if r else 0)

        # Refine the best method
        if best_result.accuracy < self.model_spec.get("accuracy_threshold", 0.9):
            # Accuracy too low, try less aggressive compression
            if "4bit" in best_result.checkpoint_path:
                # Try 8-bit instead
                strategy_params = {
                    "methods": [CompressionMethod.GPTQ],
                    "quantization_bits": 8,
                    "quantization_method": "gptq",
                    "do_finetune": True,  # Add fine-tuning
                    "finetune_steps": 1000,
                    "rationale": "Less aggressive quantization with fine-tuning",
                }
            else:
                # Try pruning with fine-tuning
                strategy_params = {
                    "methods": [CompressionMethod.PRUNING],
                    "pruning_ratio": 0.3,
                    "do_finetune": True,
                    "finetune_steps": 1500,
                    "rationale": "Structured pruning with fine-tuning",
                }
        else:
            # Accuracy acceptable, optimize for other metrics
            if best_result.latency_ms > 100:
                # Try to improve latency
                strategy_params = {
                    "methods": [CompressionMethod.AWQ],
                    "quantization_bits": 4,
                    "quantization_method": "awq",
                    "rationale": "AWQ for better inference speed",
                }
            else:
                # Try combined approach
                strategy_params = {
                    "methods": [CompressionMethod.PRUNING, CompressionMethod.QUANTIZATION],
                    "pruning_ratio": 0.2,
                    "quantization_bits": 8,
                    "quantization_method": "int8",
                    "rationale": "Combined pruning and quantization",
                }

        return CompressionStrategy(
            episode_id=self.episode_count,
            strategy_id=str(uuid4())[:8],
            **strategy_params
        )

    def _generate_optimization_strategy(
        self,
        previous_results: Optional[List[EvaluationResult]],
    ) -> CompressionStrategy:
        """Generate optimized strategies for late episodes."""
        if not previous_results:
            return self._generate_random_strategy()

        # Analyze Pareto frontier
        pareto_strategies = [r for r in previous_results if r and r.is_pareto_optimal]

        if pareto_strategies:
            # Find gaps in Pareto frontier
            gap_strategy = self._find_pareto_gap_strategy(pareto_strategies)
            if gap_strategy:
                return gap_strategy

        # Fine-tune best strategy
        best_result = max(previous_results, key=lambda r: self._score_result(r) if r else 0)

        # Make small adjustments
        strategy_params = self._mutate_strategy(best_result)

        return CompressionStrategy(
            episode_id=self.episode_count,
            strategy_id=str(uuid4())[:8],
            **strategy_params
        )

    def _generate_random_strategy(self) -> Dict[str, Any]:
        """Generate a random compression strategy."""
        methods = random.choice([
            [CompressionMethod.AUTOROUND],
            [CompressionMethod.GPTQ],
            [CompressionMethod.AWQ],
            [CompressionMethod.PRUNING],
            [CompressionMethod.INT8],
        ])

        strategy = {
            "methods": methods,
            "rationale": "Random exploration",
        }

        if CompressionMethod.AUTOROUND in methods or CompressionMethod.GPTQ in methods:
            strategy["quantization_bits"] = random.choice([4, 8])
            strategy["quantization_method"] = methods[0].value
        elif CompressionMethod.AWQ in methods:
            strategy["quantization_bits"] = 4
            strategy["quantization_method"] = "awq"
        elif CompressionMethod.INT8 in methods:
            strategy["quantization_bits"] = 8
            strategy["quantization_method"] = "int8"
        elif CompressionMethod.PRUNING in methods:
            strategy["pruning_ratio"] = random.uniform(0.2, 0.6)

        # Randomly add fine-tuning
        if random.random() < 0.3:
            strategy["do_finetune"] = True
            strategy["finetune_steps"] = random.choice([500, 1000, 2000])

        return strategy

    def _add_exploration_noise(self, strategy: CompressionStrategy) -> CompressionStrategy:
        """Add exploration noise to a strategy."""
        # Randomly modify some parameters
        if hasattr(strategy, "quantization_bits") and random.random() < 0.5:
            strategy.quantization_bits = 8 if strategy.quantization_bits == 4 else 4

        if hasattr(strategy, "pruning_ratio") and random.random() < 0.5:
            strategy.pruning_ratio = min(0.8, max(0.1, strategy.pruning_ratio + random.uniform(-0.1, 0.1)))

        if random.random() < 0.2:
            strategy.do_finetune = not strategy.do_finetune

        return strategy

    def _score_result(self, result: EvaluationResult) -> float:
        """Score a result using the reward function."""
        if not result:
            return 0.0

        reward, _ = self.reward_function.calculate_reward(
            accuracy=result.accuracy,
            latency_ms=result.latency_ms,
            memory_gb=result.memory_gb,
            size_gb=result.model_size_gb,
            energy_j=result.energy_joules,
        )
        return reward

    def _find_pareto_gap_strategy(
        self,
        pareto_strategies: List[EvaluationResult],
    ) -> Optional[CompressionStrategy]:
        """Find strategies that could fill gaps in Pareto frontier."""
        if len(pareto_strategies) < 2:
            return None

        # Sort by accuracy
        sorted_strategies = sorted(pareto_strategies, key=lambda s: s.accuracy)

        # Find largest accuracy gap
        max_gap = 0
        gap_index = 0
        for i in range(len(sorted_strategies) - 1):
            gap = sorted_strategies[i + 1].accuracy - sorted_strategies[i].accuracy
            if gap > max_gap:
                max_gap = gap
                gap_index = i

        if max_gap > 0.05:  # Significant gap
            # Try to fill the gap
            lower_strat = sorted_strategies[gap_index]
            upper_strat = sorted_strategies[gap_index + 1]

            # Interpolate between strategies
            if lower_strat.model_size_gb < upper_strat.model_size_gb:
                # Lower accuracy has smaller size, try intermediate compression
                target_bits = 6  # Between 4 and 8
                strategy_params = {
                    "methods": [CompressionMethod.GPTQ],
                    "quantization_bits": target_bits,
                    "quantization_method": "gptq",
                    "do_finetune": True,
                    "finetune_steps": 1000,
                    "rationale": f"Fill Pareto gap between {lower_strat.accuracy:.2f} and {upper_strat.accuracy:.2f}",
                }

                return CompressionStrategy(
                    episode_id=self.episode_count,
                    strategy_id=str(uuid4())[:8],
                    **strategy_params
                )

        return None

    def _mutate_strategy(self, result: EvaluationResult) -> Dict[str, Any]:
        """Create a mutated version of a successful strategy."""
        strategy_params = {
            "methods": [],
            "rationale": "Fine-tuning successful strategy",
        }

        # Infer original strategy from checkpoint path
        checkpoint = result.checkpoint_path.lower()

        if "autoround" in checkpoint:
            strategy_params["methods"] = [CompressionMethod.AUTOROUND]
            strategy_params["quantization_method"] = "autoround"
        elif "gptq" in checkpoint:
            strategy_params["methods"] = [CompressionMethod.GPTQ]
            strategy_params["quantization_method"] = "gptq"
        elif "awq" in checkpoint:
            strategy_params["methods"] = [CompressionMethod.AWQ]
            strategy_params["quantization_method"] = "awq"
        elif "int8" in checkpoint:
            strategy_params["methods"] = [CompressionMethod.INT8]
            strategy_params["quantization_method"] = "int8"
        elif "pruned" in checkpoint:
            strategy_params["methods"] = [CompressionMethod.PRUNING]

        # Extract bit width if present
        if "4bit" in checkpoint:
            strategy_params["quantization_bits"] = 4
        elif "8bit" in checkpoint:
            strategy_params["quantization_bits"] = 8

        # Make small mutations
        if "quantization_bits" in strategy_params:
            if result.accuracy < 0.9:
                # Try higher bits
                strategy_params["quantization_bits"] = min(8, strategy_params["quantization_bits"] + 2)
            # Add fine-tuning if not present
            if "lora" not in checkpoint and "finetune" not in checkpoint:
                strategy_params["do_finetune"] = True
                strategy_params["finetune_steps"] = 1000

        if "pruning" in checkpoint:
            # Adjust pruning ratio
            current_ratio = 0.3  # Default assumption
            if result.accuracy < 0.9:
                strategy_params["pruning_ratio"] = max(0.1, current_ratio - 0.1)
            else:
                strategy_params["pruning_ratio"] = min(0.6, current_ratio + 0.1)

        return strategy_params


class AdaptiveStrategyGenerator(StrategyGenerator):
    """Adaptive strategy generator that learns from feedback."""

    def __init__(
        self,
        model_spec: Dict[str, Any],
        reward_function: Optional[MultiObjectiveReward] = None,
    ):
        super().__init__(model_spec, reward_function)
        self.successful_strategies = []
        self.failed_strategies = []
        self.method_performance = {}

    def record_result(
        self,
        strategy: CompressionStrategy,
        result: EvaluationResult,
    ):
        """Record strategy result for learning.

        Args:
            strategy: Executed strategy
            result: Evaluation result
        """
        # Categorize as success or failure
        score = self._score_result(result)

        if score > 0 and result.accuracy >= self.model_spec.get("accuracy_threshold", 0.85):
            self.successful_strategies.append((strategy, result))
        else:
            self.failed_strategies.append((strategy, result))

        # Track method performance
        for method in strategy.methods:
            if method.value not in self.method_performance:
                self.method_performance[method.value] = {
                    "count": 0,
                    "total_score": 0,
                    "avg_accuracy": 0,
                    "avg_compression": 0,
                }

            stats = self.method_performance[method.value]
            stats["count"] += 1
            stats["total_score"] += score
            stats["avg_accuracy"] = (
                stats["avg_accuracy"] * (stats["count"] - 1) + result.accuracy
            ) / stats["count"]
            stats["avg_compression"] = (
                stats["avg_compression"] * (stats["count"] - 1) + result.compression_ratio
            ) / stats["count"]

    def get_best_methods(self, top_k: int = 3) -> List[str]:
        """Get best performing compression methods.

        Args:
            top_k: Number of top methods to return

        Returns:
            List of best method names
        """
        if not self.method_performance:
            return []

        # Sort by average score
        sorted_methods = sorted(
            self.method_performance.items(),
            key=lambda x: x[1]["total_score"] / max(1, x[1]["count"]),
            reverse=True,
        )

        return [method for method, _ in sorted_methods[:top_k]]

    def generate_informed_strategy(self) -> CompressionStrategy:
        """Generate strategy based on learned performance."""
        best_methods = self.get_best_methods(top_k=2)

        if not best_methods:
            return self._generate_random_strategy()

        # Use best performing method
        method_name = best_methods[0]
        method_enum = CompressionMethod(method_name)

        strategy_params = {
            "methods": [method_enum],
            "rationale": f"Using best performing method: {method_name}",
        }

        # Set method-specific parameters based on history
        stats = self.method_performance[method_name]

        if method_enum in [CompressionMethod.AUTOROUND, CompressionMethod.GPTQ, CompressionMethod.AWQ]:
            # Choose bit width based on accuracy requirements
            if stats["avg_accuracy"] < self.model_spec.get("accuracy_threshold", 0.9):
                strategy_params["quantization_bits"] = 8
            else:
                strategy_params["quantization_bits"] = 4
            strategy_params["quantization_method"] = method_name.lower()

        elif method_enum == CompressionMethod.PRUNING:
            # Set pruning ratio based on performance
            if stats["avg_accuracy"] < 0.9:
                strategy_params["pruning_ratio"] = 0.2
            else:
                strategy_params["pruning_ratio"] = 0.4

        # Add fine-tuning if accuracy is below threshold
        if stats["avg_accuracy"] < self.model_spec.get("accuracy_threshold", 0.9):
            strategy_params["do_finetune"] = True
            strategy_params["finetune_steps"] = 1500

        return CompressionStrategy(
            episode_id=self.episode_count,
            strategy_id=str(uuid4())[:8],
            **strategy_params
        )

    def suggest_early_stopping(self) -> bool:
        """Suggest whether to stop early based on convergence.

        Returns:
            True if convergence detected
        """
        if len(self.successful_strategies) < 5:
            return False

        # Check if recent strategies aren't improving
        recent_scores = [
            self._score_result(result)
            for _, result in self.successful_strategies[-5:]
        ]

        # Check variance in recent scores
        score_variance = np.var(recent_scores)
        score_improvement = recent_scores[-1] - recent_scores[0]

        # Suggest stopping if low variance and no improvement
        return score_variance < 0.01 and score_improvement < 0.05


def create_strategy_generator(
    model_spec: Dict[str, Any],
    objective: str = "balanced",
) -> StrategyGenerator:
    """Create a strategy generator with appropriate configuration.

    Args:
        model_spec: Model specification
        objective: Optimization objective

    Returns:
        Configured strategy generator
    """
    # Create appropriate reward function
    if objective == "accuracy":
        from src.reward.reward_function import create_accuracy_focused_reward

        reward_config = create_accuracy_focused_reward()
    elif objective == "latency":
        from src.reward.reward_function import create_latency_focused_reward

        reward_config = create_latency_focused_reward()
    elif objective == "memory":
        from src.reward.reward_function import create_memory_constrained_reward

        model_name = model_spec.get("model_name", "")
        default_memory = _estimate_model_size_gb(model_name) if model_name else 8.0
        max_memory = model_spec.get("target_memory_gb", default_memory)
        reward_config = create_memory_constrained_reward(max_memory)
    else:
        from src.reward.reward_function import create_balanced_reward

        reward_config = create_balanced_reward()

    reward_function = MultiObjectiveReward(reward_config)

    # Use adaptive generator for better learning
    return AdaptiveStrategyGenerator(model_spec, reward_function)


__all__ = [
    "StrategyGenerator",
    "AdaptiveStrategyGenerator",
    "create_strategy_generator",
]