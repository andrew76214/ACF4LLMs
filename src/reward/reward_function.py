"""Multi-objective reward function for compression optimization."""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the multi-objective reward function."""
    # Objective weights (higher = more important)
    accuracy_weight: float = 1.0
    latency_weight: float = 0.3
    memory_weight: float = 0.3
    size_weight: float = 0.2
    energy_weight: float = 0.1

    # Hard constraints (violations result in negative reward)
    min_accuracy: float = 0.85
    max_latency_ms: Optional[float] = None
    max_memory_gb: Optional[float] = None
    max_size_gb: Optional[float] = None

    # Normalization targets (for scaling)
    target_accuracy: float = 0.95
    target_latency_ms: float = 100.0
    target_memory_gb: float = 8.0
    target_size_gb: float = 4.0
    target_energy_j: float = 1.0

    # Penalty multipliers
    constraint_violation_penalty: float = 10.0

    # Reward shaping
    use_exponential_accuracy: bool = True
    use_log_latency: bool = True


class MultiObjectiveReward:
    """Calculate multi-objective rewards for compression strategies."""

    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize reward calculator with configuration."""
        self.config = config or RewardConfig()

    def calculate_reward(
        self,
        accuracy: float,
        latency_ms: float,
        memory_gb: float,
        size_gb: float,
        energy_j: Optional[float] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate multi-objective reward.

        Args:
            accuracy: Model accuracy (0-1)
            latency_ms: Inference latency in milliseconds
            memory_gb: Memory usage in GB
            size_gb: Model size in GB
            energy_j: Energy consumption in joules (optional)
            baseline_metrics: Baseline metrics for relative scoring

        Returns:
            Tuple of (total_reward, component_scores)
        """
        component_scores = {}

        # 1. Accuracy component (higher is better)
        if self.config.use_exponential_accuracy:
            # Exponential reward for accuracy (strongly penalize low accuracy)
            accuracy_score = np.exp(3 * (accuracy - self.config.target_accuracy))
        else:
            # Linear normalization
            accuracy_score = accuracy / self.config.target_accuracy

        component_scores["accuracy"] = accuracy_score * self.config.accuracy_weight

        # 2. Latency component (lower is better)
        if self.config.use_log_latency:
            # Log scale for latency (diminishing returns for very low latency)
            latency_score = 1.0 - np.log(latency_ms / self.config.target_latency_ms + 1) / np.log(10)
        else:
            # Linear normalization
            latency_score = self.config.target_latency_ms / latency_ms

        component_scores["latency"] = latency_score * self.config.latency_weight

        # 3. Memory component (lower is better)
        memory_score = self.config.target_memory_gb / memory_gb
        component_scores["memory"] = memory_score * self.config.memory_weight

        # 4. Size component (lower is better)
        size_score = self.config.target_size_gb / size_gb
        component_scores["size"] = size_score * self.config.size_weight

        # 5. Energy component (lower is better, optional)
        if energy_j is not None:
            energy_score = self.config.target_energy_j / energy_j
            component_scores["energy"] = energy_score * self.config.energy_weight

        # Calculate base reward
        total_reward = sum(component_scores.values())

        # Apply constraints (hard penalties)
        constraint_violations = []

        if accuracy < self.config.min_accuracy:
            violation = self.config.min_accuracy - accuracy
            penalty = violation * self.config.constraint_violation_penalty
            total_reward -= penalty
            constraint_violations.append(f"accuracy < {self.config.min_accuracy}")

        if self.config.max_latency_ms and latency_ms > self.config.max_latency_ms:
            violation = (latency_ms - self.config.max_latency_ms) / self.config.max_latency_ms
            penalty = violation * self.config.constraint_violation_penalty
            total_reward -= penalty
            constraint_violations.append(f"latency > {self.config.max_latency_ms}ms")

        if self.config.max_memory_gb and memory_gb > self.config.max_memory_gb:
            violation = (memory_gb - self.config.max_memory_gb) / self.config.max_memory_gb
            penalty = violation * self.config.constraint_violation_penalty
            total_reward -= penalty
            constraint_violations.append(f"memory > {self.config.max_memory_gb}GB")

        if self.config.max_size_gb and size_gb > self.config.max_size_gb:
            violation = (size_gb - self.config.max_size_gb) / self.config.max_size_gb
            penalty = violation * self.config.constraint_violation_penalty
            total_reward -= penalty
            constraint_violations.append(f"size > {self.config.max_size_gb}GB")

        # Add relative improvement bonus if baseline provided
        if baseline_metrics:
            improvement_bonus = self._calculate_improvement_bonus(
                accuracy, latency_ms, memory_gb, size_gb, baseline_metrics
            )
            component_scores["improvement_bonus"] = improvement_bonus
            total_reward += improvement_bonus

        # Store constraint violations in scores
        if constraint_violations:
            component_scores["constraint_violations"] = constraint_violations

        return total_reward, component_scores

    def _calculate_improvement_bonus(
        self,
        accuracy: float,
        latency_ms: float,
        memory_gb: float,
        size_gb: float,
        baseline_metrics: Dict[str, float],
    ) -> float:
        """Calculate bonus reward for improvements over baseline."""
        bonus = 0.0

        # Accuracy retention bonus
        if "accuracy" in baseline_metrics:
            accuracy_retention = accuracy / baseline_metrics["accuracy"]
            if accuracy_retention > 0.95:  # Maintained 95%+ of original accuracy
                bonus += 0.2
            elif accuracy_retention > 0.90:  # Maintained 90%+ of original accuracy
                bonus += 0.1

        # Speedup bonus
        if "latency_ms" in baseline_metrics:
            speedup = baseline_metrics["latency_ms"] / latency_ms
            if speedup > 2.0:  # 2x+ speedup
                bonus += 0.3
            elif speedup > 1.5:  # 1.5x+ speedup
                bonus += 0.15

        # Compression bonus
        if "size_gb" in baseline_metrics:
            compression_ratio = baseline_metrics["size_gb"] / size_gb
            if compression_ratio > 4.0:  # 4x+ compression
                bonus += 0.3
            elif compression_ratio > 2.0:  # 2x+ compression
                bonus += 0.15

        return bonus

    def rank_strategies(
        self,
        strategies: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rank multiple strategies by reward.

        Args:
            strategies: List of strategy results with metrics

        Returns:
            List of (strategy, reward) tuples sorted by reward
        """
        strategy_rewards = []

        for strategy in strategies:
            reward, scores = self.calculate_reward(
                accuracy=strategy.get("accuracy", 0),
                latency_ms=strategy.get("latency_ms", 1000),
                memory_gb=strategy.get("memory_gb", 16),
                size_gb=strategy.get("size_gb", 16),
                energy_j=strategy.get("energy_j"),
            )

            strategy_rewards.append((strategy, reward))

        # Sort by reward (descending)
        strategy_rewards.sort(key=lambda x: x[1], reverse=True)

        return strategy_rewards

    def update_weights(
        self,
        feedback: Dict[str, float],
    ):
        """Update reward weights based on user feedback.

        Args:
            feedback: Dictionary with weight adjustments
        """
        if "accuracy_weight" in feedback:
            self.config.accuracy_weight = feedback["accuracy_weight"]
        if "latency_weight" in feedback:
            self.config.latency_weight = feedback["latency_weight"]
        if "memory_weight" in feedback:
            self.config.memory_weight = feedback["memory_weight"]
        if "size_weight" in feedback:
            self.config.size_weight = feedback["size_weight"]
        if "energy_weight" in feedback:
            self.config.energy_weight = feedback["energy_weight"]

        logger.info(f"Updated reward weights: {self.get_weights()}")

    def get_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        return {
            "accuracy": self.config.accuracy_weight,
            "latency": self.config.latency_weight,
            "memory": self.config.memory_weight,
            "size": self.config.size_weight,
            "energy": self.config.energy_weight,
        }

    def calculate_pareto_score(
        self,
        metrics: Dict[str, float],
        pareto_solutions: List[Dict[str, float]],
    ) -> float:
        """Calculate how close a solution is to the Pareto frontier.

        Args:
            metrics: Current solution metrics
            pareto_solutions: List of Pareto optimal solutions

        Returns:
            Distance score (lower is better)
        """
        if not pareto_solutions:
            return 0.0

        # Calculate minimum distance to Pareto frontier
        min_distance = float("inf")

        for pareto_solution in pareto_solutions:
            # Euclidean distance in normalized space
            distance = 0.0

            # Normalize and calculate distance
            if "accuracy" in metrics and "accuracy" in pareto_solution:
                acc_diff = (metrics["accuracy"] - pareto_solution["accuracy"]) / self.config.target_accuracy
                distance += acc_diff ** 2

            if "latency_ms" in metrics and "latency_ms" in pareto_solution:
                lat_diff = (metrics["latency_ms"] - pareto_solution["latency_ms"]) / self.config.target_latency_ms
                distance += lat_diff ** 2

            if "memory_gb" in metrics and "memory_gb" in pareto_solution:
                mem_diff = (metrics["memory_gb"] - pareto_solution["memory_gb"]) / self.config.target_memory_gb
                distance += mem_diff ** 2

            if "size_gb" in metrics and "size_gb" in pareto_solution:
                size_diff = (metrics["size_gb"] - pareto_solution["size_gb"]) / self.config.target_size_gb
                distance += size_diff ** 2

            distance = np.sqrt(distance)
            min_distance = min(min_distance, distance)

        return min_distance


class AdaptiveRewardFunction(MultiObjectiveReward):
    """Adaptive reward function that learns from history."""

    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__(config)
        self.history = []
        self.weight_adaptation_rate = 0.1

    def add_to_history(
        self,
        metrics: Dict[str, float],
        reward: float,
        user_feedback: Optional[float] = None,
    ):
        """Add evaluation to history for learning.

        Args:
            metrics: Strategy metrics
            reward: Calculated reward
            user_feedback: Optional user rating (-1 to 1)
        """
        self.history.append({
            "metrics": metrics,
            "reward": reward,
            "user_feedback": user_feedback,
            "weights": self.get_weights().copy(),
        })

        # Adapt weights if user feedback provided
        if user_feedback is not None:
            self._adapt_weights(metrics, reward, user_feedback)

    def _adapt_weights(
        self,
        metrics: Dict[str, float],
        calculated_reward: float,
        user_feedback: float,
    ):
        """Adapt weights based on user feedback.

        Args:
            metrics: Strategy metrics
            calculated_reward: Our calculated reward
            user_feedback: User's assessment (-1 to 1)
        """
        # If user disagrees with our reward, adjust weights
        reward_error = user_feedback - np.tanh(calculated_reward)

        if abs(reward_error) > 0.2:  # Significant disagreement
            # Analyze which component might be misweighted
            _, component_scores = self.calculate_reward(**metrics)

            # Find component with highest contribution
            max_component = max(component_scores, key=component_scores.get)

            if reward_error > 0:  # User thinks it's better than we do
                # Increase weight of best component
                if max_component == "accuracy":
                    self.config.accuracy_weight *= (1 + self.weight_adaptation_rate)
                elif max_component == "latency":
                    self.config.latency_weight *= (1 + self.weight_adaptation_rate)
            else:  # User thinks it's worse than we do
                # Decrease weight of best component
                if max_component == "accuracy":
                    self.config.accuracy_weight *= (1 - self.weight_adaptation_rate)
                elif max_component == "latency":
                    self.config.latency_weight *= (1 - self.weight_adaptation_rate)

            # Normalize weights
            total_weight = (
                self.config.accuracy_weight +
                self.config.latency_weight +
                self.config.memory_weight +
                self.config.size_weight +
                self.config.energy_weight
            )

            self.config.accuracy_weight /= total_weight
            self.config.latency_weight /= total_weight
            self.config.memory_weight /= total_weight
            self.config.size_weight /= total_weight
            self.config.energy_weight /= total_weight

            logger.info(f"Adapted weights based on user feedback: {self.get_weights()}")

    def predict_user_preference(
        self,
        strategy_a: Dict[str, float],
        strategy_b: Dict[str, float],
    ) -> str:
        """Predict which strategy user would prefer based on history.

        Args:
            strategy_a: First strategy metrics
            strategy_b: Second strategy metrics

        Returns:
            "a" or "b" indicating predicted preference
        """
        reward_a, _ = self.calculate_reward(**strategy_a)
        reward_b, _ = self.calculate_reward(**strategy_b)

        # Use learned weights to predict
        return "a" if reward_a > reward_b else "b"


# Helper functions for common reward configurations

def create_accuracy_focused_reward() -> RewardConfig:
    """Create reward config focused on accuracy preservation."""
    return RewardConfig(
        accuracy_weight=2.0,
        latency_weight=0.2,
        memory_weight=0.2,
        size_weight=0.1,
        energy_weight=0.05,
        min_accuracy=0.95,
        use_exponential_accuracy=True,
    )


def create_latency_focused_reward() -> RewardConfig:
    """Create reward config focused on latency reduction."""
    return RewardConfig(
        accuracy_weight=0.5,
        latency_weight=2.0,
        memory_weight=0.3,
        size_weight=0.2,
        energy_weight=0.1,
        min_accuracy=0.85,
        max_latency_ms=50,
        use_log_latency=True,
    )


def create_balanced_reward() -> RewardConfig:
    """Create balanced reward configuration."""
    return RewardConfig(
        accuracy_weight=1.0,
        latency_weight=1.0,
        memory_weight=1.0,
        size_weight=1.0,
        energy_weight=0.5,
        min_accuracy=0.90,
    )


def create_memory_constrained_reward(max_memory_gb: float) -> RewardConfig:
    """Create reward config for memory-constrained environments."""
    return RewardConfig(
        accuracy_weight=1.0,
        latency_weight=0.3,
        memory_weight=2.0,
        size_weight=1.5,
        energy_weight=0.1,
        min_accuracy=0.85,
        max_memory_gb=max_memory_gb,
    )


__all__ = [
    "RewardConfig",
    "MultiObjectiveReward",
    "AdaptiveRewardFunction",
    "create_accuracy_focused_reward",
    "create_latency_focused_reward",
    "create_balanced_reward",
    "create_memory_constrained_reward",
]