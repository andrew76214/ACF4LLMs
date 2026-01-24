"""Automatic spec inference from minimal input (model_name + dataset).

Enhanced with:
- Dynamic model info from HuggingFace Hub
- Support for more model architectures (Phi, Gemma, DeepSeek, Yi, etc.)
- Auto-detection of architecture type using transformers config
- Architecture-specific recommendations
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.common.schemas import ModelSpec, CompressionMethod, Benchmark
from src.common.model_utils import (
    MODEL_SIZE_DATABASE,
    extract_model_size_from_name,
    estimate_model_size_gb,
    get_model_info_from_hub,
    get_model_architecture_from_hub,
)

logger = logging.getLogger(__name__)

# Architecture to model family mapping
ARCHITECTURE_FAMILY_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "Qwen2ForCausalLM": "qwen",
    "QWenLMHeadModel": "qwen",
    "GPT2LMHeadModel": "gpt2",
    "OPTForCausalLM": "opt",
    "FalconForCausalLM": "falcon",
    "GPTNeoXForCausalLM": "pythia",
    "BloomForCausalLM": "bloom",
    "PhiForCausalLM": "phi",
    "Phi3ForCausalLM": "phi",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
    "DeepseekForCausalLM": "deepseek",
    "YiForCausalLM": "yi",
    "StarCoderForCausalLM": "starcoder",
    "InternLM2ForCausalLM": "internlm",
    "InternLMForCausalLM": "internlm",
    "BaichuanForCausalLM": "baichuan",
    "ChatGLMForConditionalGeneration": "chatglm",
    "CohereForCausalLM": "cohere",
    "DbrxForCausalLM": "dbrx",
}


def get_model_family(model_name: str, use_auto_detect: bool = True) -> str:
    """Determine model family from model name.

    Uses a fallback chain:
    1. Auto-detect from HuggingFace Hub architecture (most accurate)
    2. Pattern matching from model name

    Args:
        model_name: HuggingFace model name
        use_auto_detect: Whether to try auto-detection from Hub (default: True)

    Returns:
        Model family string (e.g., 'llama3', 'mistral', 'phi')
    """
    # 1. Try auto-detection from Hub architecture
    if use_auto_detect:
        architecture = get_model_architecture_from_hub(model_name)
        if architecture:
            # Check architecture map
            for arch_pattern, family in ARCHITECTURE_FAMILY_MAP.items():
                if arch_pattern in architecture:
                    logger.debug(f"Auto-detected family '{family}' from architecture '{architecture}'")
                    # Special handling for Llama versions
                    if family == "llama":
                        name_lower = model_name.lower()
                        if 'llama-3' in name_lower or 'llama3' in name_lower:
                            return 'llama3'
                        return 'llama2'
                    return family

    # 2. Fall back to pattern matching
    name_lower = model_name.lower()

    # Llama family
    if 'llama' in name_lower:
        if 'llama-3' in name_lower or 'llama3' in name_lower:
            return 'llama3'
        return 'llama2'

    # Mistral/Mixtral family
    if 'mixtral' in name_lower:
        return 'mixtral'
    if 'mistral' in name_lower:
        return 'mistral'

    # Qwen family
    if 'qwen' in name_lower:
        return 'qwen'

    # Microsoft Phi family
    if 'phi-3' in name_lower or 'phi3' in name_lower:
        return 'phi3'
    if 'phi' in name_lower:
        return 'phi'

    # Google Gemma family
    if 'gemma-2' in name_lower or 'gemma2' in name_lower:
        return 'gemma2'
    if 'gemma' in name_lower:
        return 'gemma'

    # DeepSeek family
    if 'deepseek' in name_lower:
        if 'coder' in name_lower:
            return 'deepseek-coder'
        return 'deepseek'

    # Yi family (01.AI)
    if 'yi-' in name_lower or '/yi' in name_lower:
        return 'yi'

    # StarCoder family
    if 'starcoder' in name_lower:
        return 'starcoder'

    # InternLM family
    if 'internlm' in name_lower:
        return 'internlm'

    # Baichuan family
    if 'baichuan' in name_lower:
        return 'baichuan'

    # ChatGLM family
    if 'chatglm' in name_lower or 'glm' in name_lower:
        return 'chatglm'

    # GPT-2 family (for testing)
    if 'gpt2' in name_lower or 'gpt-2' in name_lower:
        return 'gpt2'

    # OPT family
    if 'opt-' in name_lower or '/opt' in name_lower:
        return 'opt'

    # Falcon family
    if 'falcon' in name_lower:
        return 'falcon'

    # Pythia family
    if 'pythia' in name_lower:
        return 'pythia'

    # BLOOM family
    if 'bloom' in name_lower:
        return 'bloom'

    # Cohere Command-R
    if 'command-r' in name_lower or 'cohere' in name_lower:
        return 'cohere'

    # DBRX
    if 'dbrx' in name_lower:
        return 'dbrx'

    return 'unknown'


def get_architecture_recommendations(model_family: str) -> Dict[str, Any]:
    """Get architecture-specific compression recommendations.

    Args:
        model_family: Model family string

    Returns:
        Dictionary with recommendations
    """
    # Default recommendations
    recommendations = {
        "preferred_quantization": ["autoround", "gptq", "awq"],
        "supports_lora": True,
        "supports_qlora": True,
        "calibration_samples": 512,
        "max_sequence_length": 2048,
        "notes": [],
    }

    # Family-specific adjustments
    if model_family in ["llama2", "llama3"]:
        recommendations["preferred_quantization"] = ["autoround", "awq", "gptq"]
        recommendations["notes"].append("Llama models work well with all quantization methods")
        if model_family == "llama3":
            recommendations["max_sequence_length"] = 8192

    elif model_family == "mistral":
        recommendations["preferred_quantization"] = ["awq", "autoround", "gptq"]
        recommendations["notes"].append("Mistral has good quantization robustness")

    elif model_family == "mixtral":
        recommendations["preferred_quantization"] = ["awq", "gptq"]
        recommendations["notes"].append("MoE models may need special handling for pruning")
        recommendations["max_sequence_length"] = 32768

    elif model_family in ["phi", "phi3"]:
        recommendations["preferred_quantization"] = ["awq", "gptq", "int8", "asvd"]
        recommendations["notes"].append("Phi models are sensitive to aggressive quantization")
        recommendations["notes"].append("ASVD can provide smoother accuracy trade-offs for Phi models")
        if model_family == "phi3":
            recommendations["max_sequence_length"] = 4096

    elif model_family in ["gemma", "gemma2"]:
        recommendations["preferred_quantization"] = ["awq", "gptq"]
        recommendations["notes"].append("Gemma models have good quantization support")
        recommendations["max_sequence_length"] = 8192

    elif model_family == "deepseek":
        recommendations["preferred_quantization"] = ["awq", "gptq", "autoround"]
        recommendations["notes"].append("DeepSeek models work well with standard methods")

    elif model_family == "deepseek-coder":
        recommendations["preferred_quantization"] = ["awq", "gptq", "asvd"]
        recommendations["notes"].append("Code models benefit from higher bit-widths for accuracy")
        recommendations["notes"].append("ASVD provides fine-grained compression control for code models")
        recommendations["calibration_samples"] = 1024

    elif model_family == "yi":
        recommendations["preferred_quantization"] = ["awq", "gptq", "autoround"]
        recommendations["notes"].append("Yi models have good quantization robustness")
        recommendations["max_sequence_length"] = 4096

    elif model_family == "starcoder":
        recommendations["preferred_quantization"] = ["gptq", "awq"]
        recommendations["notes"].append("Code models need careful calibration")
        recommendations["calibration_samples"] = 1024
        recommendations["max_sequence_length"] = 8192

    elif model_family == "qwen":
        recommendations["preferred_quantization"] = ["awq", "gptq", "autoround"]
        recommendations["max_sequence_length"] = 32768

    elif model_family == "gpt2":
        recommendations["preferred_quantization"] = ["int8", "gptq"]
        recommendations["notes"].append("Small model, aggressive quantization works well")
        recommendations["max_sequence_length"] = 1024

    return recommendations


def get_recommended_methods(model_size_gb: float, model_family: str) -> List[CompressionMethod]:
    """Get recommended compression methods based on model size and family.

    Uses architecture-specific recommendations combined with size-based heuristics.

    Args:
        model_size_gb: Model size in GB
        model_family: Model family string

    Returns:
        List of recommended CompressionMethod enums
    """
    methods = []

    # Get architecture-specific recommendations
    arch_recs = get_architecture_recommendations(model_family)
    preferred_quant = arch_recs.get("preferred_quantization", ["gptq", "autoround"])

    # Map string methods to CompressionMethod enums
    method_map = {
        "autoround": CompressionMethod.AUTOROUND,
        "gptq": CompressionMethod.GPTQ,
        "awq": CompressionMethod.AWQ,
        "int8": CompressionMethod.INT8,
        "asvd": CompressionMethod.ASVD,
    }

    # Size-based primary method selection
    if model_size_gb < 5:
        # Small models: can use any method, prefer simpler ones
        for m in preferred_quant:
            if m in method_map:
                methods.append(method_map[m])
        if CompressionMethod.INT8 not in methods:
            methods.append(CompressionMethod.INT8)

    elif model_size_gb < 30:
        # Medium models: use architecture-recommended methods
        for m in preferred_quant:
            if m in method_map:
                methods.append(method_map[m])
        # Add fallback methods
        for fallback in [CompressionMethod.AUTOROUND, CompressionMethod.GPTQ]:
            if fallback not in methods:
                methods.append(fallback)

    else:
        # Large models: prefer efficient methods for scale
        for m in preferred_quant:
            if m in method_map:
                methods.append(method_map[m])
        # Large models often benefit from pruning
        methods.append(CompressionMethod.PRUNING)

    # Family-specific adjustments
    if model_family in ['llama3', 'llama2', 'mistral', 'qwen']:
        # These families work well with LoRA for accuracy recovery
        if CompressionMethod.LORA not in methods:
            methods.append(CompressionMethod.LORA)

    elif model_family == 'mixtral':
        # MoE models benefit from specialized pruning
        if CompressionMethod.PRUNING not in methods:
            methods.insert(0, CompressionMethod.PRUNING)

    elif model_family in ['starcoder', 'deepseek-coder']:
        # Code models may need QLoRA for accuracy
        if CompressionMethod.QLORA not in methods:
            methods.append(CompressionMethod.QLORA)

    elif model_family in ['phi', 'phi3']:
        # Phi models are smaller but sensitive
        if CompressionMethod.INT8 not in methods:
            methods.insert(0, CompressionMethod.INT8)

    # Deduplicate while preserving order
    seen = set()
    unique_methods = []
    for m in methods:
        if m not in seen:
            seen.add(m)
            unique_methods.append(m)

    return unique_methods


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


def infer_spec(model_name: str, dataset: str, use_hub_api: bool = True, **kwargs) -> ModelSpec:
    """Infer complete model specification from minimal input.

    Uses a smart fallback chain to gather model information:
    1. HuggingFace Hub API (most accurate)
    2. Local database
    3. Pattern matching from name

    Args:
        model_name: HuggingFace model name or path
        dataset: Target dataset/benchmark name
        use_hub_api: Whether to try Hub API first (default: True)
        **kwargs: Optional overrides for any spec field

    Returns:
        Complete ModelSpec with inferred values
    """
    model_size_gb = None
    parameter_count = None

    # 1. Try Hub API first (most accurate)
    if use_hub_api:
        hub_info = get_model_info_from_hub(model_name)
        if hub_info:
            if hub_info.get("model_size_gb"):
                model_size_gb = hub_info["model_size_gb"]
                logger.info(f"Got model size from Hub API: {model_size_gb:.2f} GB")
            if hub_info.get("parameters"):
                parameter_count = str(hub_info["parameters"])

    # 2. Fall back to database if Hub didn't provide size
    if model_size_gb is None:
        model_size_gb = MODEL_SIZE_DATABASE.get(model_name)
        if model_size_gb:
            logger.debug(f"Got model size from database: {model_size_gb} GB")

    # 3. Fall back to pattern extraction from name
    if model_size_gb is None:
        size_info = extract_model_size_from_name(model_name)
        if size_info:
            model_size_gb, parameter_count = size_info
            logger.debug(f"Extracted model size from name: {model_size_gb} GB")
        else:
            # Default fallback for unknown models
            logger.warning(f"Unknown model size for {model_name}, using 1.0 GB fallback")
            model_size_gb = 1.0
            parameter_count = "unknown"

    # Extract parameter count from name if not already set
    if parameter_count is None:
        size_info = extract_model_size_from_name(model_name)
        if size_info:
            _, parameter_count = size_info

    # Get model family (with auto-detection)
    model_family = get_model_family(model_name, use_auto_detect=use_hub_api)

    # Get recommended compression methods
    preferred_methods = get_recommended_methods(model_size_gb, model_family)

    # Get dataset-specific requirements
    dataset_reqs = get_dataset_requirements(dataset)

    # Get architecture-specific recommendations
    arch_recs = get_architecture_recommendations(model_family)

    # Calculate hardware requirements
    hardware_reqs = calculate_hardware_requirements(model_size_gb)

    # Merge calibration samples and sequence length
    # Use the maximum of dataset and architecture recommendations
    calibration_samples = max(
        dataset_reqs["calibration_samples"],
        arch_recs.get("calibration_samples", 512)
    )
    max_sequence_length = min(
        dataset_reqs["max_sequence_length"],
        arch_recs.get("max_sequence_length", 2048)
    )

    # Build the spec
    spec = ModelSpec(
        model_name=model_name,
        model_size_gb=model_size_gb,
        model_family=model_family,
        parameter_count=parameter_count,
        preferred_methods=preferred_methods,
        calibration_samples=calibration_samples,
        max_sequence_length=max_sequence_length,
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