"""Pareto frontier tracking for multi-objective optimization."""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from src.common.schemas import EvaluationResult, CompressionStrategy, ParetoSolution


class ParetoFrontier:
    """Manages the Pareto frontier of non-dominated solutions."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize Pareto frontier tracker.

        Args:
            storage_path: Path to store/load frontier data. If None, uses in-memory only.
        """
        self.storage_path = storage_path
        self.solutions: List[ParetoSolution] = []
        self.history: List[Dict[str, Any]] = []  # Track all evaluated solutions

        # Load existing frontier if storage path exists
        if storage_path and os.path.exists(storage_path):
            self.load_from_file()

    def is_dominated(self, solution_a: EvaluationResult, solution_b: EvaluationResult) -> bool:
        """Check if solution_a is dominated by solution_b.

        Solution A is dominated by B if:
        - B is at least as good as A in all objectives
        - B is strictly better than A in at least one objective

        We optimize:
        - Maximize: accuracy
        - Minimize: latency_ms, memory_gb, energy_joules, model_size_gb

        Args:
            solution_a: First solution to compare
            solution_b: Second solution to compare

        Returns:
            True if solution_a is dominated by solution_b
        """
        # Track if B is at least as good in all objectives
        at_least_as_good = True
        # Track if B is strictly better in at least one objective
        strictly_better = False

        # Compare accuracy (higher is better)
        if solution_a.accuracy > solution_b.accuracy:
            at_least_as_good = False
        elif solution_a.accuracy < solution_b.accuracy:
            strictly_better = True

        # Compare metrics where lower is better
        minimize_metrics = ['latency_ms', 'memory_gb', 'model_size_gb']

        for metric in minimize_metrics:
            value_a = getattr(solution_a, metric)
            value_b = getattr(solution_b, metric)

            if value_a < value_b:
                at_least_as_good = False
            elif value_a > value_b:
                strictly_better = True

        # Include energy if available
        if solution_a.energy_joules is not None and solution_b.energy_joules is not None:
            if solution_a.energy_joules < solution_b.energy_joules:
                at_least_as_good = False
            elif solution_a.energy_joules > solution_b.energy_joules:
                strictly_better = True

        # Include CO2 emissions if available (5th optimization objective)
        if solution_a.co2_grams is not None and solution_b.co2_grams is not None:
            if solution_a.co2_grams < solution_b.co2_grams:
                at_least_as_good = False
            elif solution_a.co2_grams > solution_b.co2_grams:
                strictly_better = True

        return at_least_as_good and strictly_better

    def add_solution(
        self,
        strategy: CompressionStrategy,
        result: EvaluationResult
    ) -> Tuple[bool, List[str]]:
        """Add a new solution to the frontier if it's non-dominated.

        Args:
            strategy: The compression strategy used
            result: The evaluation results

        Returns:
            Tuple of (is_pareto_optimal, dominated_solutions)
                - is_pareto_optimal: True if the solution is on the Pareto frontier
                - dominated_solutions: List of strategy IDs that this solution dominates
        """
        # Track in history regardless of Pareto optimality
        self.history.append({
            "strategy_id": strategy.strategy_id,
            "timestamp": datetime.now().isoformat(),
            "result": result.model_dump()
        })

        # Check if new solution is dominated by any existing Pareto solution
        for existing in self.solutions:
            if self.is_dominated(result, existing.result):
                # New solution is dominated, not Pareto optimal
                result.is_pareto_optimal = False
                self._save_if_configured()
                return False, []

        # Find solutions that are dominated by the new solution
        dominated = []
        non_dominated = []

        for existing in self.solutions:
            if self.is_dominated(existing.result, result):
                dominated.append(existing.strategy.strategy_id)
            else:
                non_dominated.append(existing)

        # Update frontier with non-dominated solutions
        self.solutions = non_dominated

        # Add new solution to frontier
        result.is_pareto_optimal = True
        new_solution = ParetoSolution(
            strategy=strategy,
            result=result,
            dominates=dominated
        )
        self.solutions.append(new_solution)

        # Save updated frontier
        self._save_if_configured()

        return True, dominated

    def get_frontier(self) -> List[ParetoSolution]:
        """Get current Pareto frontier solutions."""
        return self.solutions

    def get_best_by_objective(self, objective: str) -> Optional[ParetoSolution]:
        """Get the best solution for a specific objective.

        Args:
            objective: The objective to optimize
                - "accuracy": Highest accuracy
                - "latency": Lowest latency
                - "memory": Lowest memory usage
                - "size": Smallest model size
                - "energy": Lowest energy consumption

        Returns:
            Best solution for the objective, or None if frontier is empty
        """
        if not self.solutions:
            return None

        if objective == "accuracy":
            return max(self.solutions, key=lambda s: s.result.accuracy)
        elif objective == "latency":
            return min(self.solutions, key=lambda s: s.result.latency_ms)
        elif objective == "memory":
            return min(self.solutions, key=lambda s: s.result.memory_gb)
        elif objective == "size":
            return min(self.solutions, key=lambda s: s.result.model_size_gb)
        elif objective == "energy":
            # Filter solutions with energy data
            with_energy = [s for s in self.solutions if s.result.energy_joules is not None]
            if with_energy:
                return min(with_energy, key=lambda s: s.result.energy_joules)
        elif objective == "carbon":
            # Filter solutions with CO2 data
            with_co2 = [s for s in self.solutions if s.result.co2_grams is not None]
            if with_co2:
                return min(with_co2, key=lambda s: s.result.co2_grams)

        return None

    def get_balanced_solution(self) -> Optional[ParetoSolution]:
        """Get a balanced solution using normalized scores.

        Returns:
            A solution that balances all objectives including CO2 emissions
        """
        if not self.solutions:
            return None

        # Normalize all metrics to 0-1 range
        # Collect min/max for each metric
        acc_values = [s.result.accuracy for s in self.solutions]
        lat_values = [s.result.latency_ms for s in self.solutions]
        mem_values = [s.result.memory_gb for s in self.solutions]
        size_values = [s.result.model_size_gb for s in self.solutions]

        # Collect CO2 values (may be None for some solutions)
        co2_values = [s.result.co2_grams for s in self.solutions if s.result.co2_grams is not None]
        has_co2_data = len(co2_values) > 0

        acc_range = (min(acc_values), max(acc_values))
        lat_range = (min(lat_values), max(lat_values))
        mem_range = (min(mem_values), max(mem_values))
        size_range = (min(size_values), max(size_values))
        co2_range = (min(co2_values), max(co2_values)) if has_co2_data else (0, 1)

        best_solution = None
        best_score = float('-inf')

        for solution in self.solutions:
            # Normalize (0 = worst, 1 = best)
            norm_acc = self._normalize(solution.result.accuracy, acc_range, higher_better=True)
            norm_lat = self._normalize(solution.result.latency_ms, lat_range, higher_better=False)
            norm_mem = self._normalize(solution.result.memory_gb, mem_range, higher_better=False)
            norm_size = self._normalize(solution.result.model_size_gb, size_range, higher_better=False)

            # Weighted score (adjusted for 5D optimization with CO2)
            if has_co2_data and solution.result.co2_grams is not None:
                norm_co2 = self._normalize(solution.result.co2_grams, co2_range, higher_better=False)
                score = (
                    0.35 * norm_acc +   # Accuracy is most important
                    0.20 * norm_lat +   # Latency
                    0.15 * norm_mem +   # Memory
                    0.15 * norm_size +  # Model size
                    0.15 * norm_co2     # CO2 emissions
                )
            else:
                # Fallback to 4D optimization if no CO2 data
                score = (
                    0.4 * norm_acc +
                    0.2 * norm_lat +
                    0.2 * norm_mem +
                    0.2 * norm_size
                )

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def _normalize(self, value: float, range_tuple: Tuple[float, float], higher_better: bool) -> float:
        """Normalize a value to 0-1 range."""
        min_val, max_val = range_tuple
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        if higher_better:
            return normalized
        else:
            return 1 - normalized  # Invert for "lower is better" metrics

    def save_to_file(self, path: Optional[str] = None):
        """Save frontier to JSON file.

        Raises:
            ValueError: If no storage path specified
            IOError: If file cannot be written
        """
        save_path = path or self.storage_path
        if not save_path:
            raise ValueError("No storage path specified")

        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "frontier": [sol.to_dict() for sol in self.solutions],
                "history": self.history,
                "metadata": {
                    "num_solutions": len(self.solutions),
                    "num_evaluated": len(self.history),
                    "last_updated": datetime.now().isoformat()
                }
            }

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except (IOError, OSError) as e:
            print(f"Warning: Failed to save Pareto frontier to {save_path}: {e}")
            raise IOError(f"Failed to save Pareto frontier: {e}") from e

    def load_from_file(self, path: Optional[str] = None):
        """Load frontier from JSON file.

        Silently returns if file doesn't exist. Logs warning on parse errors.
        """
        load_path = path or self.storage_path
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path, 'r') as f:
                data = json.load(f)

            self.solutions = []
            for sol_dict in data.get("frontier", []):
                strategy = CompressionStrategy(**sol_dict["strategy"])
                result = EvaluationResult(**sol_dict["result"])
                solution = ParetoSolution(
                    strategy=strategy,
                    result=result,
                    dominates=sol_dict.get("dominates", [])
                )
                self.solutions.append(solution)

            self.history = data.get("history", [])
        except (IOError, OSError) as e:
            print(f"Warning: Failed to read Pareto frontier from {load_path}: {e}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to parse Pareto frontier from {load_path}: {e}")

    def _save_if_configured(self):
        """Save to file if storage path is configured."""
        if self.storage_path:
            self.save_to_file()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the Pareto frontier."""
        if not self.solutions:
            return {
                "num_solutions": 0,
                "num_evaluated": len(self.history),
                "best_accuracy": None,
                "best_latency": None,
                "best_memory": None,
                "best_size": None,
                "best_co2_grams": None,
            }

        # Collect CO2 values (may be None for some solutions)
        co2_values = [s.result.co2_grams for s in self.solutions if s.result.co2_grams is not None]

        summary = {
            "num_solutions": len(self.solutions),
            "num_evaluated": len(self.history),
            "best_accuracy": max(s.result.accuracy for s in self.solutions),
            "best_latency": min(s.result.latency_ms for s in self.solutions),
            "best_memory": min(s.result.memory_gb for s in self.solutions),
            "best_size": min(s.result.model_size_gb for s in self.solutions),
            "best_co2_grams": min(co2_values) if co2_values else None,
            "accuracy_range": (
                min(s.result.accuracy for s in self.solutions),
                max(s.result.accuracy for s in self.solutions)
            ),
            "latency_range": (
                min(s.result.latency_ms for s in self.solutions),
                max(s.result.latency_ms for s in self.solutions)
            ),
            "memory_range": (
                min(s.result.memory_gb for s in self.solutions),
                max(s.result.memory_gb for s in self.solutions)
            ),
            "size_range": (
                min(s.result.model_size_gb for s in self.solutions),
                max(s.result.model_size_gb for s in self.solutions)
            ),
        }

        # Add CO2 range if data available
        if co2_values:
            summary["co2_range"] = (min(co2_values), max(co2_values))

        return summary

    def visualize_frontier(self, save_path: Optional[str] = None) -> Optional[str]:
        """Create a visualization of the Pareto frontier.

        Args:
            save_path: Path to save the visualization

        Returns:
            Path to the saved visualization or None if no solutions
        """
        if not self.solutions:
            return None

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Check if CO2 data is available
            co2_values = [s.result.co2_grams for s in self.solutions if s.result.co2_grams is not None]
            has_co2_data = len(co2_values) > 0

            # Create subplots for different objective pairs (3 rows if CO2 data available)
            if has_co2_data:
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        "Accuracy vs Latency",
                        "Accuracy vs Memory",
                        "Accuracy vs Model Size",
                        "Accuracy vs CO2",
                        "Latency vs Memory",
                        "CO2 vs Model Size"
                    )
                )
            else:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Accuracy vs Latency",
                        "Accuracy vs Memory",
                        "Accuracy vs Model Size",
                        "Latency vs Memory"
                    )
                )

            # Extract data
            accuracies = [s.result.accuracy for s in self.solutions]
            latencies = [s.result.latency_ms for s in self.solutions]
            memories = [s.result.memory_gb for s in self.solutions]
            sizes = [s.result.model_size_gb for s in self.solutions]
            labels = [s.strategy.strategy_id for s in self.solutions]
            co2_data = [s.result.co2_grams if s.result.co2_grams is not None else 0 for s in self.solutions]

            # Accuracy vs Latency
            fig.add_trace(
                go.Scatter(
                    x=latencies,
                    y=accuracies,
                    mode='markers+text',
                    text=labels,
                    textposition="top center",
                    marker=dict(size=10, color='blue'),
                    name="Pareto Solutions"
                ),
                row=1, col=1
            )

            # Accuracy vs Memory
            fig.add_trace(
                go.Scatter(
                    x=memories,
                    y=accuracies,
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    showlegend=False
                ),
                row=1, col=2
            )

            # Accuracy vs Model Size
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=accuracies,
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    showlegend=False
                ),
                row=2, col=1
            )

            if has_co2_data:
                # Accuracy vs CO2
                fig.add_trace(
                    go.Scatter(
                        x=co2_data,
                        y=accuracies,
                        mode='markers',
                        marker=dict(size=10, color='orange'),
                        name="CO2 Emissions",
                        showlegend=False
                    ),
                    row=2, col=2
                )

                # Latency vs Memory
                fig.add_trace(
                    go.Scatter(
                        x=memories,
                        y=latencies,
                        mode='markers',
                        marker=dict(size=10, color='purple'),
                        showlegend=False
                    ),
                    row=3, col=1
                )

                # CO2 vs Model Size
                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=co2_data,
                        mode='markers',
                        marker=dict(size=10, color='teal'),
                        showlegend=False
                    ),
                    row=3, col=2
                )

                # Update axes labels for 3x2 layout
                fig.update_xaxes(title_text="Latency (ms)", row=1, col=1)
                fig.update_xaxes(title_text="Memory (GB)", row=1, col=2)
                fig.update_xaxes(title_text="Model Size (GB)", row=2, col=1)
                fig.update_xaxes(title_text="CO2 (grams)", row=2, col=2)
                fig.update_xaxes(title_text="Memory (GB)", row=3, col=1)
                fig.update_xaxes(title_text="Model Size (GB)", row=3, col=2)

                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy", row=2, col=1)
                fig.update_yaxes(title_text="Accuracy", row=2, col=2)
                fig.update_yaxes(title_text="Latency (ms)", row=3, col=1)
                fig.update_yaxes(title_text="CO2 (grams)", row=3, col=2)

                # Update layout for 3x2
                fig.update_layout(
                    title_text="Pareto Frontier Visualization (5D: Accuracy, Latency, Memory, Size, CO2)",
                    height=1000,
                    showlegend=True
                )
            else:
                # Latency vs Memory (original 2x2 layout)
                fig.add_trace(
                    go.Scatter(
                        x=memories,
                        y=latencies,
                        mode='markers',
                        marker=dict(size=10, color='purple'),
                        showlegend=False
                    ),
                    row=2, col=2
                )

                # Update axes labels for 2x2 layout
                fig.update_xaxes(title_text="Latency (ms)", row=1, col=1)
                fig.update_xaxes(title_text="Memory (GB)", row=1, col=2)
                fig.update_xaxes(title_text="Model Size (GB)", row=2, col=1)
                fig.update_xaxes(title_text="Memory (GB)", row=2, col=2)

                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy", row=2, col=1)
                fig.update_yaxes(title_text="Latency (ms)", row=2, col=2)

                # Update layout for 2x2
                fig.update_layout(
                    title_text="Pareto Frontier Visualization",
                    height=800,
                    showlegend=True
                )

            # Save if path provided
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                return save_path
            else:
                # Return as HTML string
                return fig.to_html()

        except ImportError:
            print("Warning: Plotly not installed. Skipping visualization.")
            return None


# Example usage
if __name__ == "__main__":
    # Create a frontier
    frontier = ParetoFrontier(storage_path="data/experiments/pareto_frontier.json")

    # Mock some solutions
    from uuid import uuid4

    for i in range(5):
        strategy = CompressionStrategy(
            episode_id=i,
            strategy_id=str(uuid4())[:8],
            methods=[],
            quantization_bits=4 if i % 2 == 0 else 8
        )

        result = EvaluationResult(
            strategy_id=strategy.strategy_id,
            checkpoint_path=f"/checkpoints/model_{i}",
            model_size_gb=10.0 - i * 1.5,
            compression_ratio=1.5 + i * 0.5,
            accuracy=0.95 - i * 0.02,
            latency_ms=100 + i * 20,
            throughput_tokens_per_sec=100 - i * 10,
            memory_gb=8.0 - i * 0.5,
            evaluation_time_sec=60.0
        )

        is_pareto, dominated = frontier.add_solution(strategy, result)
        print(f"Solution {i}: Pareto optimal = {is_pareto}, Dominates = {dominated}")

    # Print summary
    print("\nFrontier Summary:")
    print(json.dumps(frontier.get_summary(), indent=2))

    # Get best solutions
    best_acc = frontier.get_best_by_objective('accuracy')
    best_lat = frontier.get_best_by_objective('latency')
    best_mem = frontier.get_best_by_objective('memory')

    if best_acc:
        print(f"\nBest accuracy: {best_acc.result.accuracy:.3f}")
    if best_lat:
        print(f"Best latency: {best_lat.result.latency_ms:.1f} ms")
    if best_mem:
        print(f"Best memory: {best_mem.result.memory_gb:.1f} GB")

    # Save visualization
    frontier.visualize_frontier("data/experiments/pareto_visualization.html")