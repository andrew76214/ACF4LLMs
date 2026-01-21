"""Shared model utilities for the Agentic Compression Framework.

This module provides a centralized MODEL_SIZE_DATABASE and model size estimation
function to avoid code duplication across agents and tools.
"""

import re
from typing import Optional, Tuple


# Comprehensive model size database (in GB) - single source of truth
MODEL_SIZE_DATABASE = {
    # Llama 3 family
    "meta-llama/Meta-Llama-3-8B": 16.0,
    "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,
    "meta-llama/Meta-Llama-3-70B": 140.0,
    "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,

    # Llama 2 family
    "meta-llama/Llama-2-7b-hf": 13.5,
    "meta-llama/Llama-2-7b-chat-hf": 13.5,
    "meta-llama/Llama-2-13b-hf": 26.0,
    "meta-llama/Llama-2-13b-chat-hf": 26.0,
    "meta-llama/Llama-2-70b-hf": 140.0,
    "meta-llama/Llama-2-70b-chat-hf": 140.0,

    # Mistral family
    "mistralai/Mistral-7B-v0.1": 13.5,
    "mistralai/Mistral-7B-Instruct-v0.1": 13.5,
    "mistralai/Mixtral-8x7B-v0.1": 87.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 87.0,

    # Qwen family
    "Qwen/Qwen-7B": 14.0,
    "Qwen/Qwen-14B": 28.0,
    "Qwen/Qwen-72B": 144.0,
    "Qwen/Qwen1.5-7B": 14.0,
    "Qwen/Qwen1.5-14B": 28.0,
    "Qwen/Qwen1.5-72B": 144.0,
    "Qwen/Qwen2-7B": 14.0,
    "Qwen/Qwen2-72B": 144.0,

    # GPT-2 family (for testing)
    "gpt2": 0.5,
    "gpt2-medium": 1.5,
    "gpt2-large": 3.0,
    "gpt2-xl": 6.0,

    # OPT family
    "facebook/opt-125m": 0.25,
    "facebook/opt-350m": 0.7,
    "facebook/opt-1.3b": 2.6,
    "facebook/opt-2.7b": 5.4,
    "facebook/opt-6.7b": 13.4,
    "facebook/opt-13b": 26.0,
    "facebook/opt-30b": 60.0,
    "facebook/opt-66b": 132.0,

    # Falcon family
    "tiiuae/falcon-7b": 14.0,
    "tiiuae/falcon-7b-instruct": 14.0,
    "tiiuae/falcon-40b": 80.0,
    "tiiuae/falcon-40b-instruct": 80.0,
}


def estimate_model_size_gb(model_name: str) -> float:
    """Estimate model size in GB from model name.

    Uses the known model database and pattern matching to estimate the original
    model size. Falls back to 1.0 GB if unknown.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Estimated model size in GB (FP16 weights)
    """
    # Check exact match first
    if model_name in MODEL_SIZE_DATABASE:
        return MODEL_SIZE_DATABASE[model_name]

    # Check partial match (e.g., "gpt2" in "openai/gpt2")
    model_lower = model_name.lower()
    for key, size in MODEL_SIZE_DATABASE.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return size

    # Pattern-based extraction (e.g., "7B" -> ~14 GB in FP16)
    patterns = [
        (r'(\d+\.?\d*)B', lambda x: float(x) * 2),  # B params -> ~2GB per billion in FP16
        (r'(\d+\.?\d*)b', lambda x: float(x) * 2),
        (r'(\d+)M', lambda x: float(x) / 1000 * 2),  # M params
        (r'(\d+)m', lambda x: float(x) / 1000 * 2),
    ]

    for pattern, converter in patterns:
        match = re.search(pattern, model_name)
        if match:
            return converter(match.group(1))

    # Default fallback
    return 1.0


def extract_model_size_from_name(model_name: str) -> Optional[Tuple[float, str]]:
    """Extract model size from model name using patterns.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tuple of (size_in_gb, parameter_string) or None if not found
    """
    patterns = [
        (r'(\d+)B', 1e9),       # 7B, 13B, 70B
        (r'(\d+\.\d+)B', 1e9),  # 1.3B, 6.7B
        (r'(\d+)b', 1e9),       # 7b, 13b (lowercase)
        (r'(\d+)M', 1e6),       # 125M, 350M
        (r'(\d+)m', 1e6),       # 125m (lowercase)
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, model_name)
        if match:
            size_value = float(match.group(1))
            # Estimate storage size (roughly 2 bytes per parameter for fp16)
            size_gb = (size_value * multiplier * 2) / 1e9
            return size_gb, match.group(0)

    return None


def estimate_params_from_size(size_gb: float) -> int:
    """Estimate number of parameters from model size in GB (assuming FP16).

    Args:
        size_gb: Model size in GB

    Returns:
        Estimated number of parameters
    """
    # FP16: 2 bytes per parameter, so params = size_gb * 1024^3 / 2
    return int(size_gb * (1024**3) / 2)


__all__ = [
    "MODEL_SIZE_DATABASE",
    "estimate_model_size_gb",
    "extract_model_size_from_name",
    "estimate_params_from_size",
]
