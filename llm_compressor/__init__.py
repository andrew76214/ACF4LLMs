"""
LLM Compressor: Multi-agent system for LLM compression and optimization.
Supports quantization, pruning, distillation, KV optimization with Pareto frontier analysis.
"""

__version__ = "0.1.0"
__author__ = "MLOps/LLM Compression Team"

from .core.orchestrator import Orchestrator
from .core.registry import Registry
from .core.pareto import ParetoFrontier
from .core.metrics import MetricsCollector

__all__ = [
    "Orchestrator",
    "Registry", 
    "ParetoFrontier",
    "MetricsCollector"
]