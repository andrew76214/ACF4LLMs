"""LangGraph state schema for the compression optimization workflow."""

from typing import Dict, List, Optional, Any, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from src.common.schemas import ModelSpec, CompressionStrategy, EvaluationResult


def merge_history(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Reducer for history: append new entries."""
    return left + right


def replace_value(left: Any, right: Any) -> Any:
    """Reducer that replaces with new value."""
    return right


class CompressionState(TypedDict):
    """State schema for the LangGraph compression optimization workflow.

    This state is shared across all nodes in the graph and tracks the
    entire optimization process.
    """
    # Basic information (set once at initialization)
    model_name: str
    dataset: str
    model_spec: Dict[str, Any]  # ModelSpec as dict for JSON serialization

    # Episode tracking
    current_episode: int
    max_episodes: int
    budget_hours: float
    start_time: str  # ISO format datetime string

    # History and Pareto frontier
    history: Annotated[List[Dict], merge_history]  # [{strategy, result}, ...]
    pareto_frontier: List[Dict]  # Current Pareto-optimal solutions

    # Current episode state
    current_strategy: Optional[Dict[str, Any]]  # CompressionStrategy as dict
    current_result: Optional[Dict[str, Any]]  # EvaluationResult as dict
    compressed_model_path: Optional[str]
    compression_ratio: Optional[float]  # Compression ratio from quantization/pruning
    compressed_model_size_gb: Optional[float]  # Compressed model size from quantization/pruning

    # Agent messages for LLM communication
    messages: Annotated[List[BaseMessage], add_messages]

    # Coordinator decision
    next_action: Optional[str]  # "quantization", "evaluation", "search", "end"
    action_params: Optional[Dict[str, Any]]  # Parameters for the action

    # Termination tracking
    should_terminate: bool
    termination_reason: Optional[str]
    consecutive_no_improvement: int  # Track convergence

    # Experiment metadata
    experiment_dir: str


def create_initial_state(
    model_name: str,
    dataset: str,
    model_spec: ModelSpec,
    max_episodes: int = 10,
    budget_hours: float = 2.0,
    experiment_dir: str = "data/experiments/default",
) -> CompressionState:
    """Create the initial state for a new compression optimization run.

    Args:
        model_name: HuggingFace model name
        dataset: Target benchmark dataset
        model_spec: Inferred model specification
        max_episodes: Maximum number of compression episodes
        budget_hours: Time budget in hours
        experiment_dir: Directory to save experiment results

    Returns:
        Initial CompressionState
    """
    return CompressionState(
        # Basic info
        model_name=model_name,
        dataset=dataset,
        model_spec=model_spec.model_dump(),

        # Episode tracking
        current_episode=0,
        max_episodes=max_episodes,
        budget_hours=budget_hours,
        start_time=datetime.now().isoformat(),

        # History
        history=[],
        pareto_frontier=[],

        # Current state
        current_strategy=None,
        current_result=None,
        compressed_model_path=None,
        compression_ratio=None,
        compressed_model_size_gb=None,

        # Messages
        messages=[],

        # Coordinator decision
        next_action=None,
        action_params=None,

        # Termination
        should_terminate=False,
        termination_reason=None,
        consecutive_no_improvement=0,

        # Metadata
        experiment_dir=experiment_dir,
    )


def get_pareto_summary(state: CompressionState) -> str:
    """Generate a human-readable summary of the current Pareto frontier.

    Args:
        state: Current compression state

    Returns:
        Summary string for LLM consumption
    """
    frontier = state.get("pareto_frontier", [])
    if not frontier:
        return "Pareto frontier is empty. No solutions have been found yet."

    summary_lines = [f"Pareto frontier has {len(frontier)} solutions:"]

    for i, sol in enumerate(frontier, 1):
        result = sol.get("result", {})
        strategy = sol.get("strategy", {})

        acc = result.get("accuracy", 0)
        latency = result.get("latency_ms", 0)
        size = result.get("model_size_gb", 0)
        method = strategy.get("quantization_method", "unknown")
        bits = strategy.get("quantization_bits", "?")

        summary_lines.append(
            f"  {i}. {method} {bits}-bit: "
            f"acc={acc:.3f}, latency={latency:.1f}ms, size={size:.2f}GB"
        )

    # Add gap analysis
    if len(frontier) >= 2:
        accuracies = [s["result"].get("accuracy", 0) for s in frontier]
        latencies = [s["result"].get("latency_ms", 0) for s in frontier]

        acc_range = max(accuracies) - min(accuracies)
        latency_range = max(latencies) - min(latencies)

        summary_lines.append(f"\nCoverage: accuracy range={acc_range:.3f}, latency range={latency_range:.1f}ms")

    return "\n".join(summary_lines)


def get_recent_history(state: CompressionState, n: int = 5) -> str:
    """Get the most recent N episodes as a summary string.

    Args:
        state: Current compression state
        n: Number of recent episodes to include

    Returns:
        Summary string for LLM consumption
    """
    history = state.get("history", [])
    if not history:
        return "No episodes have been executed yet."

    recent = history[-n:]
    lines = [f"Recent {len(recent)} episodes:"]

    for entry in recent:
        strategy = entry.get("strategy", {})
        result = entry.get("result", {})

        method = strategy.get("quantization_method", "unknown")
        bits = strategy.get("quantization_bits", "?")
        acc = result.get("accuracy", 0)
        is_pareto = result.get("is_pareto_optimal", False)

        pareto_mark = " [PARETO]" if is_pareto else ""
        lines.append(f"  - {method} {bits}-bit: acc={acc:.3f}{pareto_mark}")

    return "\n".join(lines)


def check_termination(state: CompressionState) -> tuple[bool, Optional[str]]:
    """Check if the optimization should terminate.

    Args:
        state: Current compression state

    Returns:
        Tuple of (should_terminate, reason)
    """
    # Check episode limit
    if state["current_episode"] >= state["max_episodes"]:
        return True, f"Reached maximum episodes ({state['max_episodes']})"

    # Check time budget
    start = datetime.fromisoformat(state["start_time"])
    elapsed_hours = (datetime.now() - start).total_seconds() / 3600
    if elapsed_hours >= state["budget_hours"]:
        return True, f"Exceeded time budget ({state['budget_hours']}h)"

    # Check convergence (no improvement for N consecutive episodes)
    if state["consecutive_no_improvement"] >= 5:
        return True, "Converged (no improvement for 5 consecutive episodes)"

    return False, None
