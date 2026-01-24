"""Agents module for the Agentic Compression Framework.

This module provides specialized agents for various compression tasks.
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "get_quantization_subagent":
        from src.agents.quantization_agent import get_quantization_subagent
        return get_quantization_subagent
    elif name == "get_evaluation_subagent":
        from src.agents.evaluation_agent import get_evaluation_subagent
        return get_evaluation_subagent
    elif name == "get_pruning_subagent":
        from src.agents.pruning_agent import get_pruning_subagent
        return get_pruning_subagent
    elif name == "get_search_subagent":
        from src.agents.search_agent import get_search_subagent
        return get_search_subagent
    elif name == "get_distillation_subagent":
        from src.agents.distillation_agent import get_distillation_subagent
        return get_distillation_subagent
    elif name == "get_finetune_subagent":
        from src.agents.finetune_agent import get_finetune_subagent
        return get_finetune_subagent
    elif name == "get_resource_monitor_subagent":
        from src.agents.resource_monitor_agent import get_resource_monitor_subagent
        return get_resource_monitor_subagent
    elif name == "get_data_manager_subagent":
        from src.agents.data_manager_agent import get_data_manager_subagent
        return get_data_manager_subagent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_quantization_subagent",
    "get_evaluation_subagent",
    "get_pruning_subagent",
    "get_search_subagent",
    "get_distillation_subagent",
    "get_finetune_subagent",
    "get_resource_monitor_subagent",
    "get_data_manager_subagent",
]
