"""Tools module for the Agentic Compression Framework.

This module provides quantization and pruning wrappers.
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name in ("get_quantizer", "BaseQuantizer", "AutoRoundQuantizer",
                "GPTQQuantizer", "AWQQuantizer", "INT8Quantizer",
                "LoRATrainer", "QLoRATrainer"):
        from src.tools import quantization_wrapper
        return getattr(quantization_wrapper, name)
    elif name in ("get_pruner", "estimate_pruning_benefits", "BasePruner",
                  "MagnitudePruner", "StructuredPruner"):
        from src.tools import pruning_wrapper
        return getattr(pruning_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Quantization
    "get_quantizer",
    "BaseQuantizer",
    "AutoRoundQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "INT8Quantizer",
    "LoRATrainer",
    "QLoRATrainer",
    # Pruning
    "get_pruner",
    "estimate_pruning_benefits",
    "BasePruner",
    "MagnitudePruner",
    "StructuredPruner",
]
