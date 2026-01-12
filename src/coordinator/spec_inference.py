"""Automatic spec inference from minimal input (model_name + dataset)."""

import re
from typing import Dict, List, Optional, Tuple
from src.common.schemas import ModelSpec, CompressionMethod, Benchmark


# Model size database (in GB) - can be extended with more models
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


def extract_model_size_from_name(model_name: str) -> Optional[Tuple[float, str]]:
    """Extract model size from model name using patterns.

    Returns:
        Tuple of (size_in_gb, parameter_string) or None
    """
    # Common patterns for model sizes
    patterns = [
        (r'(\d+)B', 1e9),    # 7B, 13B, 70B
        (r'(\d+\.\d+)B', 1e9),  # 1.3B, 6.7B
        (r'(\d+)b', 1e9),    # 7b, 13b (lowercase)
        (r'(\d+)M', 1e6),    # 125M, 350M
        (r'(\d+)m', 1e6),    # 125m (lowercase)
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, model_name)
        if match:
            size_value = float(match.group(1))
            # Estimate storage size (roughly 2 bytes per parameter for fp16)
            size_gb = (size_value * multiplier * 2) / 1e9
            return size_gb, match.group(0)

    return None


def get_model_family(model_name: str) -> str:
    """Determine model family from model name."""
    name_lower = model_name.lower()

    if 'llama' in name_lower:
        if 'llama-3' in name_lower or 'llama3' in name_lower:
            return 'llama3'
        return 'llama2'
    elif 'mistral' in name_lower:
        return 'mistral'
    elif 'mixtral' in name_lower:
        return 'mixtral'
    elif 'qwen' in name_lower:
        return 'qwen'
    elif 'gpt2' in name_lower or 'gpt-2' in name_lower:
        return 'gpt2'
    elif 'opt' in name_lower:
        return 'opt'
    elif 'falcon' in name_lower:
        return 'falcon'
    elif 'pythia' in name_lower:
        return 'pythia'
    elif 'bloom' in name_lower:
        return 'bloom'
    else:
        return 'unknown'


def get_recommended_methods(model_size_gb: float, model_family: str) -> List[CompressionMethod]:
    """Get recommended compression methods based on model size and family."""
    methods = []

    # Size-based recommendations
    if model_size_gb < 5:
        # Small models: aggressive quantization is fine
        methods.extend([
            CompressionMethod.INT8,
            CompressionMethod.GPTQ,
            CompressionMethod.AUTOROUND,
        ])
    elif model_size_gb < 30:
        # Medium models: balanced approach
        methods.extend([
            CompressionMethod.AUTOROUND,
            CompressionMethod.GPTQ,
            CompressionMethod.AWQ,
            CompressionMethod.INT8,
        ])
    else:
        # Large models: prefer methods that work well at scale
        methods.extend([
            CompressionMethod.AUTOROUND,
            CompressionMethod.AWQ,
            CompressionMethod.GPTQ,
            CompressionMethod.PRUNING,
        ])

    # Family-specific adjustments
    if model_family in ['llama3', 'llama2']:
        # Llama models work well with all methods
        if CompressionMethod.LORA not in methods:
            methods.append(CompressionMethod.LORA)
    elif model_family == 'mixtral':
        # MoE models benefit from specialized pruning
        if CompressionMethod.PRUNING not in methods:
            methods.insert(0, CompressionMethod.PRUNING)

    return methods


def get_dataset_requirements(dataset: str) -> Dict[str, any]:
    """Get specific requirements based on the target dataset."""
    dataset_lower = dataset.lower()

    requirements = {
        "primary_objective": "accuracy",
        "accuracy_threshold": 0.95,
        "calibration_samples": 512,
        "max_sequence_length": 2048,
    }

    if dataset_lower == "gsm8k":
        # Math reasoning requires high accuracy
        requirements.update({
            "primary_objective": "accuracy",
            "accuracy_threshold": 0.90,  # Math is sensitive
            "calibration_samples": 1024,  # More samples for math
            "max_sequence_length": 2048,
        })
    elif dataset_lower == "humaneval":
        # Code generation needs precision
        requirements.update({
            "primary_objective": "accuracy",
            "accuracy_threshold": 0.95,
            "calibration_samples": 512,
            "max_sequence_length": 4096,  # Code can be longer
        })
    elif dataset_lower == "commonsenseqa" or dataset_lower == "commonsense_qa":
        # Common sense is more robust
        requirements.update({
            "primary_objective": "balanced",  # Balance accuracy and speed
            "accuracy_threshold": 0.85,
            "calibration_samples": 512,
            "max_sequence_length": 512,  # Questions are short
        })
    elif dataset_lower == "truthfulqa" or dataset_lower == "truthful_qa":
        # Truthfulness is critical
        requirements.update({
            "primary_objective": "accuracy",
            "accuracy_threshold": 0.90,
            "calibration_samples": 512,
            "max_sequence_length": 1024,
        })
    elif dataset_lower == "bigbench" or "bigbench" in dataset_lower:
        # BIG-Bench Hard needs good accuracy
        requirements.update({
            "primary_objective": "accuracy",
            "accuracy_threshold": 0.85,
            "calibration_samples": 1024,
            "max_sequence_length": 2048,
        })

    return requirements


def calculate_hardware_requirements(model_size_gb: float) -> Dict[str, float]:
    """Calculate VRAM requirements based on model size."""
    # Rule of thumb:
    # - Inference needs ~1.2x model size
    # - Quantization/compression needs ~2-3x model size
    # - Fine-tuning needs ~4x model size

    return {
        "min_vram_gb": model_size_gb * 1.5,  # Minimum for inference
        "recommended_vram_gb": model_size_gb * 3.0,  # Comfortable for compression
    }


def infer_spec(model_name: str, dataset: str, **kwargs) -> ModelSpec:
    """Infer complete model specification from minimal input.

    Args:
        model_name: HuggingFace model name or path
        dataset: Target dataset/benchmark name
        **kwargs: Optional overrides for any spec field

    Returns:
        Complete ModelSpec with inferred values
    """
    # Try to get model size from database first
    model_size_gb = MODEL_SIZE_DATABASE.get(model_name)
    parameter_count = None

    # If not in database, try to extract from name
    if model_size_gb is None:
        size_info = extract_model_size_from_name(model_name)
        if size_info:
            model_size_gb, parameter_count = size_info
        else:
            # Default fallback for unknown models
            print(f"Warning: Unknown model size for {model_name}, using 1.0 GB fallback")
            model_size_gb = 1.0  # Conservative fallback
            parameter_count = "unknown"
    else:
        # Extract parameter count from name if available
        size_info = extract_model_size_from_name(model_name)
        if size_info:
            _, parameter_count = size_info

    # Get model family
    model_family = get_model_family(model_name)

    # Get recommended compression methods
    preferred_methods = get_recommended_methods(model_size_gb, model_family)

    # Get dataset-specific requirements
    dataset_reqs = get_dataset_requirements(dataset)

    # Calculate hardware requirements
    hardware_reqs = calculate_hardware_requirements(model_size_gb)

    # Build the spec
    spec = ModelSpec(
        model_name=model_name,
        model_size_gb=model_size_gb,
        model_family=model_family,
        parameter_count=parameter_count,
        preferred_methods=preferred_methods,
        calibration_samples=dataset_reqs["calibration_samples"],
        max_sequence_length=dataset_reqs["max_sequence_length"],
        min_vram_gb=hardware_reqs["min_vram_gb"],
        recommended_vram_gb=hardware_reqs["recommended_vram_gb"],
        primary_objective=dataset_reqs["primary_objective"],
        accuracy_threshold=dataset_reqs["accuracy_threshold"],
        target_speedup=2.0,  # Default target
        target_memory_reduction=0.5,  # Default target
    )

    # Apply any user overrides
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, value)

    return spec


def format_spec_summary(spec: ModelSpec) -> str:
    """Format a human-readable summary of the inferred spec."""
    lines = [
        f"Model Specification for {spec.model_name}",
        "=" * 50,
        f"Model Family: {spec.model_family}",
        f"Model Size: {spec.model_size_gb:.1f} GB",
    ]

    if spec.parameter_count:
        lines.append(f"Parameters: {spec.parameter_count}")

    lines.extend([
        f"\nHardware Requirements:",
        f"  Minimum VRAM: {spec.min_vram_gb:.1f} GB",
        f"  Recommended VRAM: {spec.recommended_vram_gb:.1f} GB",
        f"\nCompression Settings:",
        f"  Primary Objective: {spec.primary_objective}",
        f"  Accuracy Threshold: {spec.accuracy_threshold:.0%}",
        f"  Target Speedup: {spec.target_speedup}x",
        f"  Target Memory Reduction: {(1-spec.target_memory_reduction):.0%}",
        f"  Calibration Samples: {spec.calibration_samples}",
        f"  Max Sequence Length: {spec.max_sequence_length}",
        f"\nRecommended Methods:",
    ])

    for method in spec.preferred_methods:
        lines.append(f"  - {method.value}")

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test with various models
    test_cases = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", "gsm8k"),
        ("mistralai/Mistral-7B-v0.1", "humaneval"),
        ("gpt2", "commonsenseqa"),
        ("Qwen/Qwen2-72B", "bigbench_hard"),
    ]

    for model_name, dataset in test_cases:
        spec = infer_spec(model_name, dataset)
        print(format_spec_summary(spec))
        print("\n" + "=" * 50 + "\n")