"""Shared model utilities for the Agentic Compression Framework.

This module provides a centralized MODEL_SIZE_DATABASE and model size estimation
function to avoid code duplication across agents and tools.

Enhanced with:
- Dynamic model info fetching from HuggingFace Hub API
- Caching for API results to avoid repeated calls
- Fallback chain: Hub API → regex pattern → database
"""

import re
import logging
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


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

    # Phi family (Microsoft)
    "microsoft/phi-1": 2.7,
    "microsoft/phi-1_5": 2.8,
    "microsoft/phi-2": 5.4,
    "microsoft/phi-3-mini-4k-instruct": 7.6,
    "microsoft/phi-3-mini-128k-instruct": 7.6,
    "microsoft/phi-3-small-8k-instruct": 14.0,
    "microsoft/phi-3-medium-4k-instruct": 28.0,

    # Gemma family (Google)
    "google/gemma-2b": 5.0,
    "google/gemma-2b-it": 5.0,
    "google/gemma-7b": 17.0,
    "google/gemma-7b-it": 17.0,
    "google/gemma-2-9b": 18.0,
    "google/gemma-2-9b-it": 18.0,
    "google/gemma-2-27b": 54.0,
    "google/gemma-2-27b-it": 54.0,

    # DeepSeek family
    "deepseek-ai/deepseek-llm-7b-base": 14.0,
    "deepseek-ai/deepseek-llm-7b-chat": 14.0,
    "deepseek-ai/deepseek-llm-67b-base": 134.0,
    "deepseek-ai/deepseek-coder-6.7b-base": 13.4,
    "deepseek-ai/deepseek-coder-33b-base": 66.0,

    # Yi family (01.AI)
    "01-ai/Yi-6B": 12.0,
    "01-ai/Yi-6B-Chat": 12.0,
    "01-ai/Yi-9B": 18.0,
    "01-ai/Yi-34B": 68.0,
    "01-ai/Yi-34B-Chat": 68.0,

    # StarCoder family
    "bigcode/starcoder": 30.0,
    "bigcode/starcoder2-3b": 6.0,
    "bigcode/starcoder2-7b": 14.0,
    "bigcode/starcoder2-15b": 30.0,

    # InternLM family
    "internlm/internlm2-7b": 14.0,
    "internlm/internlm2-20b": 40.0,
    "internlm/internlm2.5-7b": 14.0,
}

# Cache for Hub API results (in-memory)
_hub_cache: Dict[str, Dict[str, Any]] = {}


@lru_cache(maxsize=128)
def get_model_info_from_hub(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch model info from HuggingFace Hub API.

    This function queries the HuggingFace Hub API to get model metadata
    including size, architecture, and parameter count.

    Args:
        model_name: HuggingFace model name (e.g., 'meta-llama/Meta-Llama-3-8B')

    Returns:
        Dictionary with model info or None if unavailable:
        - model_size_gb: Estimated size in GB (FP16)
        - parameters: Number of parameters (if available)
        - architecture: Model architecture type
        - safetensors_size_bytes: Size of safetensors files
    """
    # Check in-memory cache first
    if model_name in _hub_cache:
        return _hub_cache[model_name]

    try:
        from huggingface_hub import model_info, HfApi

        info = model_info(model_name)

        result: Dict[str, Any] = {
            "model_id": info.id,
            "architecture": None,
            "parameters": None,
            "model_size_gb": None,
            "safetensors_size_bytes": None,
        }

        # Try to get size from safetensors first (more accurate)
        if info.safetensors:
            # safetensors has total size info
            total_bytes = info.safetensors.total
            if total_bytes:
                result["safetensors_size_bytes"] = total_bytes
                # Convert to GB
                result["model_size_gb"] = total_bytes / (1024 ** 3)
                logger.debug(f"Got size from safetensors: {result['model_size_gb']:.2f} GB")

        # Try to get size from siblings (model files)
        if result["model_size_gb"] is None and info.siblings:
            total_bytes = 0
            for sibling in info.siblings:
                filename = sibling.rfilename
                if filename.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                    if sibling.size:
                        total_bytes += sibling.size

            if total_bytes > 0:
                result["model_size_gb"] = total_bytes / (1024 ** 3)
                logger.debug(f"Got size from siblings: {result['model_size_gb']:.2f} GB")

        # Try to get architecture from config
        if info.config:
            architectures = info.config.get("architectures", [])
            if architectures:
                result["architecture"] = architectures[0]

            # Try to estimate parameters from config
            if "num_parameters" in info.config:
                result["parameters"] = info.config["num_parameters"]
            elif all(k in info.config for k in ["hidden_size", "num_hidden_layers", "vocab_size"]):
                # Rough estimate based on transformer architecture
                hidden = info.config["hidden_size"]
                layers = info.config["num_hidden_layers"]
                vocab = info.config["vocab_size"]
                # Approximate: embedding + layers * (attention + FFN)
                params = vocab * hidden + layers * (4 * hidden * hidden + 8 * hidden * hidden)
                result["parameters"] = params

        # Cache the result
        _hub_cache[model_name] = result

        logger.info(f"Fetched model info from Hub: {model_name} -> {result['model_size_gb']:.2f} GB")
        return result

    except ImportError:
        logger.warning("huggingface_hub not installed, skipping Hub API")
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch model info from Hub for {model_name}: {e}")
        return None


def get_model_architecture_from_hub(model_name: str) -> Optional[str]:
    """Get model architecture type from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name

    Returns:
        Architecture string (e.g., 'LlamaForCausalLM') or None
    """
    info = get_model_info_from_hub(model_name)
    if info:
        return info.get("architecture")
    return None


def get_model_config_from_hub(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model configuration from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name

    Returns:
        Model config dictionary or None
    """
    try:
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_files_only=False,
        )
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to get config from Hub: {e}")
        return None


def estimate_model_size_gb(model_name: str, use_hub_api: bool = True) -> float:
    """Estimate model size in GB from model name.

    Uses a fallback chain to estimate model size:
    1. HuggingFace Hub API (most accurate, if available)
    2. Known model database (fast, no network)
    3. Pattern matching from name (e.g., "7B" -> ~14 GB)
    4. Default fallback (1.0 GB)

    Args:
        model_name: HuggingFace model name or path
        use_hub_api: Whether to try Hub API first (default: True)

    Returns:
        Estimated model size in GB (FP16 weights)
    """
    # 1. Try Hub API first (most accurate)
    if use_hub_api:
        hub_info = get_model_info_from_hub(model_name)
        if hub_info and hub_info.get("model_size_gb"):
            return hub_info["model_size_gb"]

    # 2. Check exact match in database
    if model_name in MODEL_SIZE_DATABASE:
        return MODEL_SIZE_DATABASE[model_name]

    # 3. Check partial match (e.g., "gpt2" in "openai/gpt2")
    model_lower = model_name.lower()
    for key, size in MODEL_SIZE_DATABASE.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return size

    # 4. Pattern-based extraction (e.g., "7B" -> ~14 GB in FP16)
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

    # 5. Default fallback
    logger.warning(f"Could not determine model size for {model_name}, using 1.0 GB fallback")
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
    "get_model_info_from_hub",
    "get_model_architecture_from_hub",
    "get_model_config_from_hub",
]
