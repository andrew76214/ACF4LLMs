"""Quantization Agent for model compression using LangGraph tools."""

import json
import os
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool

from src.common.schemas import CompressionMethod
from src.common.model_utils import estimate_model_size_gb
from src.tools.quantization_wrapper import get_quantizer

logger = logging.getLogger(__name__)


@tool
def quantize_model(
    model_path: str,
    method: str,
    bit_width: int = 4,
    calibration_samples: int = 512,
    calibration_dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply quantization to a model checkpoint.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method ('autoround', 'gptq', 'int8', 'awq')
        bit_width: Number of bits for quantization (typically 4 or 8)
        calibration_samples: Number of calibration samples to use
        calibration_dataset: Dataset to use for calibration
        output_dir: Directory to save quantized model (auto-generated if None)

    Returns:
        Dictionary with quantization results including:
        - checkpoint_path: Path to quantized model
        - model_size_gb: Size of quantized model
        - compression_ratio: Compression ratio achieved
        - quantization_time_sec: Time taken for quantization
        - metadata: Additional quantization metadata
    """
    print(f"[Quantization] Starting {method.upper()} quantization with {bit_width} bits...")

    try:
        # Get the appropriate quantizer
        quantizer = get_quantizer(method)

        # Run quantization
        if method.lower() == "int8":
            # INT8 doesn't need bit_width parameter
            result = quantizer.quantize(
                model_name=model_path,
                output_dir=output_dir,
            )
        else:
            result = quantizer.quantize(
                model_name=model_path,
                bit_width=bit_width,
                calibration_samples=calibration_samples,
                calibration_dataset=calibration_dataset,
                output_dir=output_dir,
            )

        print(f"[Quantization] Completed! Saved to {result['checkpoint_path']}")
        print(f"[Quantization] Compression ratio: {result['compression_ratio']:.2f}x")
        print(f"[Quantization] Model size: {result['model_size_gb']:.1f} GB")

        return result

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


@tool
def estimate_quantization_vram(
    model_path: str,
    method: str,
    bit_width: int = 4,
) -> Dict[str, float]:
    """Estimate VRAM requirements for quantization.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method
        bit_width: Number of bits for quantization

    Returns:
        Dictionary with VRAM estimates:
        - required_vram_gb: Minimum VRAM needed
        - recommended_vram_gb: Recommended VRAM for smooth operation
        - model_size_gb: Estimated model size
    """
    model_size_gb = None

    # Method 1: Try to get model info from HuggingFace Hub
    try:
        from huggingface_hub import model_info

        info = model_info(model_path)
        # Calculate size from model files
        model_size_bytes = sum(
            s.size for s in info.siblings
            if s.rfilename.endswith(('.safetensors', '.bin', '.pt', '.pth'))
            and s.size is not None
        )
        if model_size_bytes > 0:
            model_size_gb = model_size_bytes / (1024**3)
            logger.info(f"Got model size from HuggingFace Hub: {model_size_gb:.2f} GB")
    except Exception as e:
        logger.debug(f"Could not get model info from Hub: {e}")

    # Method 2: Try to load config and estimate from parameters
    if model_size_gb is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Calculate approximate parameter count based on architecture
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 768))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
            vocab_size = getattr(config, 'vocab_size', 50257)
            intermediate_size = getattr(config, 'intermediate_size',
                                       getattr(config, 'n_inner', hidden_size * 4))
            num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))

            # Estimate parameter count for transformer architecture
            # Embeddings
            embed_params = vocab_size * hidden_size

            # Per layer: attention (Q, K, V, O) + FFN (up, down)
            attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
            ffn_params = 2 * hidden_size * intermediate_size  # up and down projections
            layer_norm_params = 4 * hidden_size  # 2 layer norms per layer

            per_layer_params = attention_params + ffn_params + layer_norm_params
            total_params = embed_params + (num_layers * per_layer_params) + hidden_size  # final layer norm

            # Size in GB (assuming FP16 = 2 bytes per param)
            model_size_gb = (total_params * 2) / (1024**3)
            logger.info(f"Estimated model size from config: {model_size_gb:.2f} GB ({total_params/1e9:.2f}B params)")

        except Exception as e:
            logger.debug(f"Could not estimate from config: {e}")

    # Method 3: Fallback to size estimation from name
    if model_size_gb is None:
        import re
        # Try to extract size from model name
        patterns = [
            (r'(\d+)B', 1e9),    # 7B, 13B, 70B
            (r'(\d+\.\d+)B', 1e9),  # 1.3B, 6.7B
            (r'(\d+)b', 1e9),    # 7b, 13b
            (r'(\d+)M', 1e6),    # 125M, 350M
            (r'(\d+)m', 1e6),    # 125m
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, model_path)
            if match:
                size_value = float(match.group(1))
                model_size_gb = (size_value * multiplier * 2) / 1e9  # 2 bytes per param (FP16)
                logger.info(f"Estimated model size from name: {model_size_gb:.2f} GB")
                break

    # Final fallback
    if model_size_gb is None:
        model_size_gb = estimate_model_size_gb(model_path)
        logger.warning(f"Could not determine model size for {model_path}, using estimated {model_size_gb:.2f}GB")

    # Different methods have different memory requirements
    method_multipliers = {
        "autoround": 2.5,  # Needs model + gradients + optimizer states
        "gptq": 2.0,       # Needs model + calibration data + intermediate states
        "int8": 1.5,       # Simpler method, less overhead
        "awq": 2.0,        # Similar to GPTQ
    }

    multiplier = method_multipliers.get(method.lower(), 2.0)

    required_vram = model_size_gb * multiplier
    recommended_vram = model_size_gb * (multiplier + 0.5)

    print(f"[VRAM] Model size: {model_size_gb:.1f} GB")
    print(f"[VRAM] Required for {method}: {required_vram:.1f} GB")
    print(f"[VRAM] Recommended: {recommended_vram:.1f} GB")

    return {
        "model_size_gb": model_size_gb,
        "required_vram_gb": required_vram,
        "recommended_vram_gb": recommended_vram,
        "method": method,
        "bit_width": bit_width,
        "multiplier": multiplier,
    }


@tool
def validate_quantization_compatibility(
    model_path: str,
    method: str,
    bit_width: int = 4,
) -> Dict[str, Any]:
    """Check if a model is compatible with the specified quantization method.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method to validate
        bit_width: Number of bits for quantization

    Returns:
        Dictionary with validation results:
        - is_compatible: Whether the method is compatible
        - warnings: List of warnings
        - recommendations: List of recommendations
    """
    warnings = []
    recommendations = []
    is_compatible = True

    # Mock validation logic
    # In production, this would check actual model architecture
    model_name_lower = model_path.lower()

    # Check method compatibility
    if method.lower() == "autoround":
        if "mistral" in model_name_lower or "mixtral" in model_name_lower:
            recommendations.append("AutoRound works particularly well with Mistral models")
    elif method.lower() == "gptq":
        if bit_width not in [4, 8]:
            warnings.append(f"GPTQ typically works best with 4 or 8 bits, got {bit_width}")
    elif method.lower() == "awq":
        if "llama" not in model_name_lower:
            warnings.append("AWQ is optimized for Llama models, results may vary")

    # Check bit width
    if bit_width < 4:
        warnings.append(f"Bit width {bit_width} is very aggressive, may cause significant accuracy loss")
        recommendations.append("Consider using 4 or 8 bits for better accuracy")
    elif bit_width > 8:
        warnings.append(f"Bit width {bit_width} provides limited compression benefit")
        recommendations.append("Consider using 4 or 8 bits for better compression")

    return {
        "is_compatible": is_compatible,
        "warnings": warnings,
        "recommendations": recommendations,
    }


@tool
def apply_lora_finetuning(
    model_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    training_steps: int = 100,
    learning_rate: float = 2e-4,
    calibration_dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply LoRA fine-tuning to adapt a model with low-rank adapters.

    LoRA (Low-Rank Adaptation) adds small trainable adapter layers to a frozen
    base model, enabling efficient fine-tuning without modifying original weights.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        lora_rank: Rank of LoRA decomposition (8, 16, 32, or 64). Higher = more capacity
        lora_alpha: LoRA scaling factor (typically 2x rank)
        training_steps: Number of training steps (default: 100)
        learning_rate: Learning rate for adapter training (default: 2e-4)
        calibration_dataset: Dataset name for fine-tuning (optional)
        output_dir: Directory to save adapter weights (auto-generated if None)

    Returns:
        Dictionary with fine-tuning results including:
        - checkpoint_path: Path to saved adapter
        - adapter_size_mb: Size of adapter weights in MB
        - training_loss: Final training loss
        - training_time_sec: Time taken for training
    """
    print(f"[LoRA] Starting LoRA fine-tuning with rank={lora_rank}, alpha={lora_alpha}...")

    try:
        trainer = get_quantizer("lora")
        result = trainer.finetune(
            model_name=model_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_steps=training_steps,
            learning_rate=learning_rate,
            calibration_dataset=calibration_dataset,
            output_dir=output_dir,
        )

        print(f"[LoRA] Completed! Adapter saved to {result['checkpoint_path']}")
        print(f"[LoRA] Adapter size: {result['adapter_size_mb']:.2f} MB")
        print(f"[LoRA] Training loss: {result['training_loss']:.4f}")

        return result

    except Exception as e:
        logger.error(f"LoRA fine-tuning failed: {e}")
        raise


@tool
def apply_qlora_finetuning(
    model_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    bits: int = 4,
    training_steps: int = 100,
    learning_rate: float = 2e-4,
    calibration_dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply QLoRA (4-bit quantization + LoRA) for memory-efficient fine-tuning.

    QLoRA loads the base model in 4-bit precision and trains LoRA adapters on top,
    enabling fine-tuning of large models on consumer GPUs with limited VRAM.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        lora_rank: Rank of LoRA decomposition (8, 16, 32, or 64)
        lora_alpha: LoRA scaling factor (typically 2x rank)
        bits: Quantization bits for base model (default: 4)
        training_steps: Number of training steps (default: 100)
        learning_rate: Learning rate for adapter training (default: 2e-4)
        calibration_dataset: Dataset name for fine-tuning (optional)
        output_dir: Directory to save adapter weights (auto-generated if None)

    Returns:
        Dictionary with fine-tuning results including:
        - checkpoint_path: Path to saved adapter
        - adapter_size_mb: Size of adapter weights in MB
        - training_loss: Final training loss
        - training_time_sec: Time taken for training
        - base_model_bits: Quantization bits for base model (4)
        - total_vram_gb: Peak VRAM usage during training
    """
    print(f"[QLoRA] Starting QLoRA fine-tuning with {bits}-bit base, rank={lora_rank}...")

    try:
        trainer = get_quantizer("qlora")
        result = trainer.finetune(
            model_name=model_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            bits=bits,
            training_steps=training_steps,
            learning_rate=learning_rate,
            calibration_dataset=calibration_dataset,
            output_dir=output_dir,
        )

        print(f"[QLoRA] Completed! Adapter saved to {result['checkpoint_path']}")
        print(f"[QLoRA] Adapter size: {result['adapter_size_mb']:.2f} MB")
        print(f"[QLoRA] Training loss: {result['training_loss']:.4f}")
        print(f"[QLoRA] Peak VRAM: {result['total_vram_gb']:.2f} GB")

        return result

    except Exception as e:
        logger.error(f"QLoRA fine-tuning failed: {e}")
        raise


@tool
def list_available_quantization_methods() -> List[Dict[str, Any]]:
    """List all available quantization methods with their characteristics.

    Returns:
        List of dictionaries describing each method
    """
    methods = [
        {
            "name": "autoround",
            "description": "Adaptive rounding for weight quantization",
            "supported_bits": [4, 8],
            "pros": ["Good accuracy retention", "Fast inference"],
            "cons": ["Longer quantization time"],
            "recommended_for": ["Large language models", "Production deployments"],
        },
        {
            "name": "gptq",
            "description": "Post-training quantization with layer-wise optimization",
            "supported_bits": [2, 3, 4, 8],
            "pros": ["Excellent compression", "Wide hardware support"],
            "cons": ["Requires calibration data"],
            "recommended_for": ["Memory-constrained environments", "Edge deployment"],
        },
        {
            "name": "int8",
            "description": "Simple 8-bit integer quantization",
            "supported_bits": [8],
            "pros": ["Fast quantization", "Good hardware support"],
            "cons": ["Limited compression ratio"],
            "recommended_for": ["Quick prototyping", "Balanced compression"],
        },
        {
            "name": "awq",
            "description": "Activation-aware weight quantization",
            "supported_bits": [4],
            "pros": ["Preserves important weights", "Good for LLMs"],
            "cons": ["Limited bit width options"],
            "recommended_for": ["Llama models", "Chat applications"],
        },
        {
            "name": "lora",
            "description": "Low-Rank Adaptation - adds small trainable adapters",
            "supported_bits": [16],
            "pros": ["Preserves base model", "Fast to train", "Small adapter files"],
            "cons": ["Requires training compute", "No direct compression"],
            "recommended_for": ["Recovery after quantization", "Task-specific adaptation"],
        },
        {
            "name": "qlora",
            "description": "Quantized LoRA - 4-bit base model with trainable adapters",
            "supported_bits": [4],
            "pros": ["Memory efficient", "Combines quantization + adaptation"],
            "cons": ["Requires GPU with 4-bit support"],
            "recommended_for": ["Fine-tuning large models on limited VRAM"],
        },
    ]

    return methods


def get_quantization_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the quantization subagent configuration for DeepAgents.

    Args:
        spec: Model specification with requirements and constraints

    Returns:
        Subagent configuration dictionary
    """
    # Extract relevant spec information
    model_name = spec.get("model_name", "unknown")
    preferred_methods = spec.get("preferred_methods", ["autoround", "gptq"])
    calibration_samples = spec.get("calibration_samples", 512)

    # Format preferred methods for prompt
    methods_str = ", ".join([m.value if hasattr(m, 'value') else str(m) for m in preferred_methods])

    prompt = f"""You are a Quantization Specialist Agent responsible for compressing neural networks.

Model: {model_name}
Preferred Methods: {methods_str}
Calibration Samples: {calibration_samples}

Your responsibilities:
1. Apply quantization methods to compress models
2. Validate compatibility before quantization
3. Estimate VRAM requirements
4. Track compression ratios and model sizes
5. Save quantized checkpoints with metadata

Available tools:
- quantize_model: Apply quantization to a model
- estimate_quantization_vram: Check VRAM requirements
- validate_quantization_compatibility: Validate method compatibility
- list_available_quantization_methods: Get information about methods

Guidelines:
- Always validate compatibility before quantizing
- Check VRAM requirements to prevent out-of-memory errors
- Start with conservative settings (8-bit) if unsure
- Use 4-bit quantization for aggressive compression
- Save detailed metadata with each quantized model
- Report compression ratio and final model size

When you receive a quantization request:
1. First, validate the method compatibility
2. Check VRAM requirements
3. Execute quantization with appropriate parameters
4. Return the checkpoint path and compression metrics
"""

    return {
        "name": "quantization",
        "description": "Applies various quantization methods to compress neural networks",
        "prompt": prompt,
        "tools": [
            quantize_model,
            estimate_quantization_vram,
            validate_quantization_compatibility,
            list_available_quantization_methods,
        ],
        "model": "anthropic:claude-3-haiku-20240307",  # Use cheaper model for specialized task
    }


# Export tools
__all__ = [
    "quantize_model",
    "estimate_quantization_vram",
    "validate_quantization_compatibility",
    "list_available_quantization_methods",
    "apply_lora_finetuning",
    "apply_qlora_finetuning",
    "get_quantization_subagent",
]
