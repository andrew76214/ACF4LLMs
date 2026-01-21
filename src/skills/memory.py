"""Skill Memory for learning from historical skill executions.

This module provides a persistent memory system that records skill execution
results and uses them to make recommendations for future decisions.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

from src.skills.schema import (
    SkillContext,
    SkillResult,
    SkillRecord,
    SkillStatistics,
    SkillRecommendation,
)

logger = logging.getLogger(__name__)

# Default storage path for global skill memory
DEFAULT_MEMORY_PATH = "data/skill_memory/memory.json"


class SkillMemory:
    """Skill execution memory for learning and recommendations.

    This class maintains a persistent record of skill executions and their
    outcomes, enabling data-driven recommendations for future optimization runs.

    The memory is global and shared across all experiments, allowing the system
    to learn from accumulated experience.

    Example:
        memory = SkillMemory()

        # Record an execution
        context = SkillContext(model_family="llama", model_size_gb=14.0, ...)
        result = SkillResult(skill_name="quantization_gptq", accuracy_delta=-0.02, ...)
        memory.record(context, result)

        # Get recommendations for a new context
        new_context = SkillContext(model_family="llama", model_size_gb=7.0, ...)
        recommendations = memory.query(new_context)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the skill memory.

        Args:
            storage_path: Path to the JSON storage file.
                         Uses default global path if not specified.
        """
        self.storage_path = Path(storage_path or DEFAULT_MEMORY_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._records: List[SkillRecord] = []
        self._statistics: Dict[str, SkillStatistics] = {}
        self._last_aggregation: Optional[datetime] = None

        # Load existing memory
        self._load()

    def record(
        self,
        context: SkillContext,
        result: SkillResult,
        experiment_id: Optional[str] = None,
    ) -> None:
        """Record a skill execution result.

        Args:
            context: The context in which the skill was executed
            result: The result of the skill execution
            experiment_id: Optional experiment identifier
        """
        record = SkillRecord(
            context=context,
            result=result,
            timestamp=datetime.now(),
            experiment_id=experiment_id,
        )

        self._records.append(record)
        self._save()

        # Update statistics if we have enough records
        if len(self._records) % 10 == 0:
            self._aggregate_statistics()

        logger.debug(f"Recorded skill execution: {result.skill_name}")

    def query(
        self,
        context: SkillContext,
        n_recommendations: int = 3,
        min_samples: int = 3,
    ) -> List[SkillRecommendation]:
        """Query for skill recommendations based on context.

        Args:
            context: The current context for which recommendations are needed
            n_recommendations: Maximum number of recommendations to return
            min_samples: Minimum number of samples required for a recommendation

        Returns:
            List of skill recommendations sorted by confidence
        """
        # Ensure statistics are up to date
        if self._last_aggregation is None or len(self._records) % 5 == 0:
            self._aggregate_statistics()

        recommendations = []

        # Find similar contexts in history
        similar_records = self._find_similar_contexts(context)

        # Group by skill and analyze
        skill_performance: Dict[str, List[SkillRecord]] = defaultdict(list)
        for record in similar_records:
            skill_performance[record.result.skill_name].append(record)

        # Generate recommendations for each skill with sufficient data
        for skill_name, records in skill_performance.items():
            if len(records) < min_samples:
                continue

            rec = self._generate_recommendation(skill_name, records, context)
            if rec is not None:
                recommendations.append(rec)

        # Sort by confidence and return top N
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations[:n_recommendations]

    def get_statistics(self, skill_name: str) -> Optional[SkillStatistics]:
        """Get aggregated statistics for a specific skill.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillStatistics if available, None otherwise
        """
        if self._last_aggregation is None:
            self._aggregate_statistics()

        return self._statistics.get(skill_name)

    def get_all_statistics(self) -> Dict[str, SkillStatistics]:
        """Get statistics for all skills.

        Returns:
            Dictionary mapping skill name to statistics
        """
        if self._last_aggregation is None:
            self._aggregate_statistics()

        return self._statistics.copy()

    def get_best_skill_for_objective(
        self,
        objective: str,
        model_family: Optional[str] = None,
    ) -> Optional[str]:
        """Get the best performing skill for a specific objective.

        Args:
            objective: Optimization objective ('accuracy', 'size', 'latency')
            model_family: Optional model family filter

        Returns:
            Name of the best skill, or None if insufficient data
        """
        if self._last_aggregation is None:
            self._aggregate_statistics()

        relevant_records = self._records
        if model_family:
            relevant_records = [
                r for r in relevant_records
                if r.context.model_family.lower() == model_family.lower()
            ]

        if not relevant_records:
            return None

        # Group by skill
        skill_scores: Dict[str, List[float]] = defaultdict(list)

        for record in relevant_records:
            if not record.result.success:
                continue

            skill = record.result.skill_name

            if objective == "accuracy":
                # Higher accuracy delta is better (less negative)
                skill_scores[skill].append(record.result.accuracy_delta)
            elif objective == "size":
                # Higher compression ratio is better
                skill_scores[skill].append(record.result.compression_ratio)
            elif objective == "latency":
                # More negative latency delta is better
                skill_scores[skill].append(-record.result.latency_delta)

        if not skill_scores:
            return None

        # Find skill with best average score
        best_skill = None
        best_avg = float('-inf')

        for skill, scores in skill_scores.items():
            if len(scores) >= 3:  # Minimum samples
                avg = statistics.mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_skill = skill

        return best_skill

    def import_from_experiments(self, experiments_dir: str) -> int:
        """Import historical data from experiment directories.

        Args:
            experiments_dir: Path to the experiments directory

        Returns:
            Number of records imported
        """
        experiments_path = Path(experiments_dir)
        if not experiments_path.exists():
            logger.warning(f"Experiments directory not found: {experiments_dir}")
            return 0

        imported = 0

        for exp_dir in experiments_path.iterdir():
            if not exp_dir.is_dir():
                continue

            # Try to load model spec for context
            spec_path = exp_dir / "model_spec.json"
            if not spec_path.exists():
                continue

            try:
                with open(spec_path) as f:
                    spec = json.load(f)

                # Process each episode
                for episode_dir in sorted(exp_dir.glob("episode_*")):
                    imported += self._import_episode(episode_dir, spec, exp_dir.name)

            except Exception as e:
                logger.warning(f"Failed to import from {exp_dir}: {e}")

        if imported > 0:
            self._save()
            self._aggregate_statistics()
            logger.info(f"Imported {imported} records from experiments")

        return imported

    def clear(self) -> None:
        """Clear all memory records."""
        self._records.clear()
        self._statistics.clear()
        self._last_aggregation = None
        self._save()
        logger.info("Cleared skill memory")

    def get_record_count(self) -> int:
        """Get the total number of records in memory."""
        return len(self._records)

    def generate_llm_summary(self, context: Optional[SkillContext] = None) -> str:
        """Generate a summary of skill performance for LLM consumption.

        Args:
            context: Optional context to tailor the summary

        Returns:
            Formatted summary string
        """
        if self._last_aggregation is None:
            self._aggregate_statistics()

        if not self._statistics:
            return "No skill execution history available yet."

        lines = ["Skill Performance Summary (based on historical executions):\n"]

        for skill_name, stats in sorted(self._statistics.items()):
            lines.append(f"\n{skill_name}:")
            lines.append(f"  Executions: {stats.total_executions}")
            lines.append(f"  Success rate: {stats.success_rate:.1%}")
            lines.append(f"  Avg accuracy delta: {stats.avg_accuracy_delta:+.3f}")
            lines.append(f"  Avg compression ratio: {stats.avg_compression_ratio:.2f}x")
            lines.append(f"  Pareto hit rate: {stats.pareto_hit_rate:.1%}")

            if stats.best_for_model_families:
                lines.append(f"  Best for models: {', '.join(stats.best_for_model_families)}")

        # Add recommendations if context provided
        if context:
            recommendations = self.query(context, n_recommendations=3)
            if recommendations:
                lines.append("\n\nRecommendations for current context:")
                for rec in recommendations:
                    lines.append(f"  - {rec.skill_name} (confidence: {rec.confidence:.1%})")
                    lines.append(f"    {rec.reasoning}")

        return "\n".join(lines)

    # Private methods

    def _load(self) -> None:
        """Load memory from storage file."""
        if not self.storage_path.exists():
            logger.info("No existing skill memory found, starting fresh")
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Load records
            self._records = [
                SkillRecord(
                    context=SkillContext(**r["context"]),
                    result=SkillResult(**r["result"]),
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    experiment_id=r.get("experiment_id"),
                )
                for r in data.get("records", [])
            ]

            # Load statistics
            for skill_name, stats_data in data.get("statistics", {}).items():
                self._statistics[skill_name] = SkillStatistics(**stats_data)

            if data.get("last_aggregation"):
                self._last_aggregation = datetime.fromisoformat(data["last_aggregation"])

            logger.info(f"Loaded {len(self._records)} skill memory records")

        except Exception as e:
            logger.error(f"Failed to load skill memory: {e}")
            self._records = []
            self._statistics = {}

    def _save(self) -> None:
        """Save memory to storage file."""
        try:
            data = {
                "records": [
                    {
                        "context": r.context.model_dump(),
                        "result": r.result.model_dump(),
                        "timestamp": r.timestamp.isoformat(),
                        "experiment_id": r.experiment_id,
                    }
                    for r in self._records
                ],
                "statistics": {
                    name: stats.model_dump()
                    for name, stats in self._statistics.items()
                },
                "last_aggregation": (
                    self._last_aggregation.isoformat()
                    if self._last_aggregation else None
                ),
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save skill memory: {e}")

    def _aggregate_statistics(self) -> None:
        """Aggregate statistics from all records."""
        if not self._records:
            return

        skill_records: Dict[str, List[SkillRecord]] = defaultdict(list)
        for record in self._records:
            skill_records[record.result.skill_name].append(record)

        for skill_name, records in skill_records.items():
            successful = [r for r in records if r.result.success]

            if not records:
                continue

            # Calculate statistics
            success_rate = len(successful) / len(records)

            accuracy_deltas = [r.result.accuracy_delta for r in successful]
            compression_ratios = [r.result.compression_ratio for r in successful]
            execution_times = [r.result.execution_time_sec for r in successful]
            pareto_hits = [r for r in successful if r.result.is_pareto_optimal]

            avg_accuracy_delta = statistics.mean(accuracy_deltas) if accuracy_deltas else 0.0
            avg_compression = statistics.mean(compression_ratios) if compression_ratios else 1.0
            avg_time = statistics.mean(execution_times) if execution_times else 0.0
            pareto_rate = len(pareto_hits) / len(records) if records else 0.0

            # Find best model families
            family_performance: Dict[str, List[float]] = defaultdict(list)
            for r in successful:
                family_performance[r.context.model_family].append(r.result.accuracy_delta)

            best_families = []
            if family_performance:
                family_avgs = {
                    f: statistics.mean(scores)
                    for f, scores in family_performance.items()
                    if len(scores) >= 2
                }
                # Families where this skill has above-average accuracy retention
                threshold = statistics.mean(accuracy_deltas) if accuracy_deltas else 0.0
                best_families = [
                    f for f, avg in family_avgs.items()
                    if avg >= threshold
                ]

            # Find common parameters
            common_params = self._find_common_params(successful)

            self._statistics[skill_name] = SkillStatistics(
                skill_name=skill_name,
                total_executions=len(records),
                success_rate=success_rate,
                avg_accuracy_delta=avg_accuracy_delta,
                avg_latency_delta=0.0,  # TODO: track latency
                avg_compression_ratio=avg_compression,
                avg_execution_time_sec=avg_time,
                pareto_hit_rate=pareto_rate,
                best_for_model_families=best_families,
                best_for_objectives=[],  # TODO: determine best objectives
                common_params=common_params,
            )

        self._last_aggregation = datetime.now()
        logger.debug(f"Aggregated statistics for {len(self._statistics)} skills")

    def _find_similar_contexts(
        self,
        context: SkillContext,
        max_records: int = 100,
    ) -> List[SkillRecord]:
        """Find records with similar contexts.

        Args:
            context: The context to match
            max_records: Maximum number of records to return

        Returns:
            List of similar records, sorted by similarity
        """
        scored_records: List[Tuple[float, SkillRecord]] = []

        for record in self._records:
            similarity = self._compute_context_similarity(context, record.context)
            scored_records.append((similarity, record))

        # Sort by similarity (descending) and return top N
        scored_records.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored_records[:max_records]]

    def _compute_context_similarity(
        self,
        context1: SkillContext,
        context2: SkillContext,
    ) -> float:
        """Compute similarity score between two contexts.

        Args:
            context1: First context
            context2: Second context

        Returns:
            Similarity score (0.0 to 1.0)
        """
        score = 0.0

        # Model family match (most important)
        if context1.model_family.lower() == context2.model_family.lower():
            score += 0.4
        elif self._families_similar(context1.model_family, context2.model_family):
            score += 0.2

        # Model size similarity
        if context1.model_size_gb > 0 and context2.model_size_gb > 0:
            size_ratio = min(context1.model_size_gb, context2.model_size_gb) / \
                        max(context1.model_size_gb, context2.model_size_gb)
            score += 0.3 * size_ratio

        # Objective match
        if context1.target_objective == context2.target_objective:
            score += 0.2

        # Pareto gap match
        if context1.current_pareto_gap == context2.current_pareto_gap:
            score += 0.1

        return score

    def _families_similar(self, family1: str, family2: str) -> bool:
        """Check if two model families are similar."""
        similar_groups = [
            {"llama", "llama2", "llama3", "codellama"},
            {"mistral", "mixtral"},
            {"gpt", "gpt2", "gpt-neo", "gpt-j"},
            {"falcon", "falcon2"},
            {"phi", "phi2", "phi3"},
        ]

        f1_lower = family1.lower()
        f2_lower = family2.lower()

        for group in similar_groups:
            if f1_lower in group and f2_lower in group:
                return True

        return False

    def _generate_recommendation(
        self,
        skill_name: str,
        records: List[SkillRecord],
        context: SkillContext,
    ) -> Optional[SkillRecommendation]:
        """Generate a recommendation for a skill based on historical records.

        Args:
            skill_name: Name of the skill
            records: Historical records for this skill
            context: Current context

        Returns:
            SkillRecommendation or None
        """
        successful = [r for r in records if r.result.success]
        if not successful:
            return None

        # Calculate metrics
        accuracy_deltas = [r.result.accuracy_delta for r in successful]
        compression_ratios = [r.result.compression_ratio for r in successful]
        pareto_hits = sum(1 for r in successful if r.result.is_pareto_optimal)

        avg_accuracy = statistics.mean(accuracy_deltas)
        avg_compression = statistics.mean(compression_ratios)

        # Calculate confidence based on sample size and consistency
        n_samples = len(successful)
        base_confidence = min(0.9, 0.5 + (n_samples / 20))

        if len(accuracy_deltas) >= 2:
            # Reduce confidence if results are inconsistent
            std_accuracy = statistics.stdev(accuracy_deltas)
            consistency_factor = max(0.5, 1.0 - std_accuracy)
            base_confidence *= consistency_factor

        # Boost confidence if skill works well for this model family
        family_records = [
            r for r in successful
            if r.context.model_family.lower() == context.model_family.lower()
        ]
        if len(family_records) >= 2:
            base_confidence = min(0.95, base_confidence * 1.1)

        # Generate reasoning
        reasoning_parts = []

        if avg_accuracy > -0.03:
            reasoning_parts.append(f"good accuracy retention ({avg_accuracy:+.3f})")
        elif avg_accuracy > -0.05:
            reasoning_parts.append(f"moderate accuracy loss ({avg_accuracy:+.3f})")
        else:
            reasoning_parts.append(f"notable accuracy loss ({avg_accuracy:+.3f})")

        if avg_compression > 3.0:
            reasoning_parts.append(f"high compression ({avg_compression:.1f}x)")
        elif avg_compression > 1.5:
            reasoning_parts.append(f"good compression ({avg_compression:.1f}x)")

        pareto_rate = pareto_hits / len(successful) if successful else 0
        if pareto_rate > 0.5:
            reasoning_parts.append(f"often Pareto-optimal ({pareto_rate:.0%})")

        reasoning = "Based on history: " + ", ".join(reasoning_parts) + "."

        # Suggest common successful parameters
        suggested_params = self._find_common_params(
            [r for r in successful if r.result.is_pareto_optimal] or successful[:5]
        )

        return SkillRecommendation(
            skill_name=skill_name,
            confidence=base_confidence,
            expected_accuracy_delta=avg_accuracy,
            expected_compression_ratio=avg_compression,
            reasoning=reasoning,
            suggested_params=suggested_params,
        )

    def _find_common_params(self, records: List[SkillRecord]) -> Dict[str, Any]:
        """Find common parameters from successful records.

        Args:
            records: List of records to analyze

        Returns:
            Dictionary of common parameters
        """
        if not records:
            return {}

        param_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

        for record in records:
            for param, value in record.result.params.items():
                # Convert lists to tuples for hashability
                if isinstance(value, list):
                    value = tuple(value)
                param_counts[param][value] += 1

        # Find most common value for each parameter
        common_params = {}
        for param, value_counts in param_counts.items():
            if value_counts:
                most_common = max(value_counts.items(), key=lambda x: x[1])
                # Only include if it appears in at least 50% of records
                if most_common[1] >= len(records) / 2:
                    value = most_common[0]
                    # Convert tuples back to lists
                    if isinstance(value, tuple):
                        value = list(value)
                    common_params[param] = value

        return common_params

    def _import_episode(
        self,
        episode_dir: Path,
        spec: Dict[str, Any],
        experiment_id: str,
    ) -> int:
        """Import records from a single episode directory.

        Args:
            episode_dir: Path to the episode directory
            spec: Model specification
            experiment_id: Experiment identifier

        Returns:
            Number of records imported
        """
        strategy_path = episode_dir / "strategy.json"
        results_path = episode_dir / "results.json"

        if not strategy_path.exists() or not results_path.exists():
            return 0

        try:
            with open(strategy_path) as f:
                strategy = json.load(f)
            with open(results_path) as f:
                results = json.load(f)

            # Build context
            context = SkillContext(
                model_family=spec.get("model_family", "unknown"),
                model_size_gb=spec.get("model_size_gb", 1.0),
                target_objective=spec.get("primary_objective", "balanced"),
                hardware_profile="imported",
            )

            # Determine skill name from strategy
            methods = strategy.get("methods", [])
            if not methods:
                return 0

            method = methods[0]
            if isinstance(method, dict):
                method = method.get("value", method.get("name", "unknown"))

            # Map method to skill name
            skill_name = self._map_method_to_skill(method, strategy)

            # Build result
            result = SkillResult(
                skill_name=skill_name,
                params=self._extract_params(strategy),
                accuracy_delta=0.0,  # Would need baseline to compute
                compression_ratio=results.get("compression_ratio", 1.0),
                is_pareto_optimal=results.get("is_pareto_optimal", False),
                execution_time_sec=results.get("evaluation_time_sec", 0.0),
                success=True,
            )

            # Check if already imported
            for existing in self._records:
                if (existing.experiment_id == experiment_id and
                    existing.result.skill_name == skill_name):
                    return 0

            # Record it
            self._records.append(SkillRecord(
                context=context,
                result=result,
                experiment_id=experiment_id,
            ))

            return 1

        except Exception as e:
            logger.debug(f"Failed to import episode {episode_dir}: {e}")
            return 0

    def _map_method_to_skill(self, method: str, strategy: Dict[str, Any]) -> str:
        """Map a compression method to a skill name."""
        method_lower = method.lower()

        if method_lower in ["gptq", "autoround", "int8", "awq"]:
            return f"quantization_{method_lower}"
        elif method_lower == "pruning":
            pruning_method = strategy.get("pruning_method", "magnitude")
            return f"pruning_{pruning_method}"
        elif method_lower == "lora":
            return "lora_finetune"
        elif method_lower == "qlora":
            return "qlora_finetune"

        return f"unknown_{method_lower}"

    def _extract_params(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant parameters from a strategy."""
        params = {}

        if strategy.get("quantization_bits"):
            params["bit_width"] = strategy["quantization_bits"]
        if strategy.get("pruning_ratio"):
            params["sparsity"] = strategy["pruning_ratio"]
        if strategy.get("lora_rank"):
            params["lora_rank"] = strategy["lora_rank"]

        return params


# Global memory instance
_global_memory: Optional[SkillMemory] = None


def get_global_memory() -> SkillMemory:
    """Get the global skill memory instance.

    Returns:
        Global SkillMemory instance
    """
    global _global_memory
    if _global_memory is None:
        _global_memory = SkillMemory()
    return _global_memory


def reset_global_memory() -> None:
    """Reset the global skill memory instance."""
    global _global_memory
    _global_memory = None


__all__ = [
    "SkillMemory",
    "get_global_memory",
    "reset_global_memory",
    "DEFAULT_MEMORY_PATH",
]
