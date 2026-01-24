"""Coordinator module for the Agentic Compression Framework.

This module provides the main LangGraph-based orchestration for compression optimization.
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "LangGraphCoordinator":
        from src.coordinator.langgraph_coordinator import LangGraphCoordinator
        return LangGraphCoordinator
    elif name == "create_coordinator":
        from src.coordinator.langgraph_coordinator import create_coordinator
        return create_coordinator
    elif name == "infer_spec":
        from src.coordinator.spec_inference import infer_spec
        return infer_spec
    elif name == "format_spec_summary":
        from src.coordinator.spec_inference import format_spec_summary
        return format_spec_summary
    elif name == "ParetoFrontier":
        from src.coordinator.pareto import ParetoFrontier
        return ParetoFrontier
    elif name == "CompressionState":
        from src.coordinator.state import CompressionState
        return CompressionState
    elif name == "create_initial_state":
        from src.coordinator.state import create_initial_state
        return create_initial_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LangGraphCoordinator",
    "create_coordinator",
    "infer_spec",
    "format_spec_summary",
    "ParetoFrontier",
    "CompressionState",
    "create_initial_state",
]
