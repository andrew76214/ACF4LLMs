"""Predefined pipeline templates for common compression scenarios.

This module provides ready-to-use pipeline configurations for various
compression goals and model types.
"""

from typing import Dict, Any, List, Optional

from src.skills.schema import PipelineStep, StepCondition
from src.skills.pipeline import CompressionPipeline, PipelineBuilder


# Pipeline template definitions as dictionaries
# These can be used directly or customized before execution

PIPELINE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "aggressive_compression": {
        "name": "aggressive_compression",
        "description": "Maximum compression with pruning + 4-bit quantization + optional LoRA recovery. "
                      "Targets ~10x compression ratio. Best for edge deployment.",
        "steps": [
            {
                "skill": "pruning_magnitude",
                "params": {"sparsity": 0.3},
            },
            {
                "skill": "quantization_gptq",
                "params": {"bit_width": 4},
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 16},
                "condition": {
                    "metric": "accuracy_delta",
                    "operator": "lt",
                    "threshold": -0.05,
                    "on_false": "skip",
                },
            },
        ],
    },

    "accuracy_recovery": {
        "name": "accuracy_recovery",
        "description": "4-bit quantization with LoRA fine-tuning for accuracy recovery. "
                      "Good balance of compression and accuracy retention.",
        "steps": [
            {
                "skill": "quantization_gptq",
                "params": {"bit_width": 4},
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 16, "training_steps": 100},
            },
        ],
    },

    "memory_efficient_large": {
        "name": "memory_efficient_large",
        "description": "Optimized for large models (>30B params) with limited VRAM. "
                      "Uses structured pruning + QLoRA for memory efficiency.",
        "steps": [
            {
                "skill": "pruning_structured",
                "params": {"sparsity": 0.2, "granularity": "head"},
            },
            {
                "skill": "qlora_finetune",
                "params": {"lora_rank": 32, "bits": 4},
            },
        ],
    },

    "adaptive_compression": {
        "name": "adaptive_compression",
        "description": "Intelligent compression that adapts based on intermediate results. "
                      "Applies additional compression only if initial results are insufficient.",
        "steps": [
            {
                "skill": "quantization_gptq",
                "params": {"bit_width": 4},
            },
            {
                "skill": "pruning_magnitude",
                "params": {"sparsity": 0.3},
                "condition": {
                    "metric": "compression_ratio",
                    "operator": "lt",
                    "threshold": 3.0,
                    "on_false": "skip",
                },
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 32},
                "condition": {
                    "metric": "accuracy",
                    "operator": "lt",
                    "threshold": 0.9,
                    "on_false": "skip",
                },
            },
        ],
    },

    "quick_compression": {
        "name": "quick_compression",
        "description": "Fast 8-bit quantization for minimal latency. "
                      "Best for quick prototyping and testing.",
        "steps": [
            {
                "skill": "quantization_int8",
                "params": {},
            },
        ],
    },

    "balanced_quality": {
        "name": "balanced_quality",
        "description": "Balanced approach with AWQ quantization and conditional LoRA. "
                      "Prioritizes quality retention while achieving good compression.",
        "steps": [
            {
                "skill": "quantization_awq",
                "params": {"bit_width": 4},
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 8},
                "condition": {
                    "metric": "accuracy_delta",
                    "operator": "lt",
                    "threshold": -0.03,
                    "on_false": "skip",
                },
            },
        ],
    },

    "llama_optimized": {
        "name": "llama_optimized",
        "description": "Optimized pipeline for Llama model family. "
                      "Uses AWQ (best for Llama) with structured head pruning.",
        "steps": [
            {
                "skill": "pruning_structured",
                "params": {"sparsity": 0.15, "granularity": "head"},
            },
            {
                "skill": "quantization_awq",
                "params": {"bit_width": 4},
            },
        ],
    },

    "production_deployment": {
        "name": "production_deployment",
        "description": "Production-ready pipeline with AutoRound (best accuracy) "
                      "and comprehensive quality checks.",
        "steps": [
            {
                "skill": "quantization_autoround",
                "params": {"bit_width": 4, "calibration_samples": 1024},
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 16, "training_steps": 200},
                "condition": {
                    "metric": "accuracy_delta",
                    "operator": "lt",
                    "threshold": -0.02,
                    "on_false": "skip",
                },
            },
        ],
    },

    "extreme_compression": {
        "name": "extreme_compression",
        "description": "Maximum compression for extremely constrained environments. "
                      "Aggressive pruning + 3-bit quantization. May impact accuracy significantly.",
        "steps": [
            {
                "skill": "pruning_magnitude",
                "params": {"sparsity": 0.5},
            },
            {
                "skill": "quantization_gptq",
                "params": {"bit_width": 3},
            },
            {
                "skill": "lora_finetune",
                "params": {"lora_rank": 32, "training_steps": 200},
            },
        ],
    },

    "pruning_focused": {
        "name": "pruning_focused",
        "description": "Pruning-centric pipeline for models where structured sparsity "
                      "can be leveraged for speedup.",
        "steps": [
            {
                "skill": "pruning_structured",
                "params": {"sparsity": 0.3, "granularity": "channel"},
            },
            {
                "skill": "pruning_magnitude",
                "params": {"sparsity": 0.2},
                "condition": {
                    "metric": "compression_ratio",
                    "operator": "lt",
                    "threshold": 1.5,
                    "on_false": "skip",
                },
            },
        ],
    },
}


def get_template(name: str) -> Optional[CompressionPipeline]:
    """Get a pipeline template by name.

    Args:
        name: Name of the template

    Returns:
        CompressionPipeline instance or None if not found
    """
    template = PIPELINE_TEMPLATES.get(name)
    if template is None:
        return None

    return CompressionPipeline.from_dict(template)


def get_template_names() -> List[str]:
    """Get list of available template names.

    Returns:
        List of template names
    """
    return list(PIPELINE_TEMPLATES.keys())


def get_template_descriptions() -> Dict[str, str]:
    """Get descriptions of all available templates.

    Returns:
        Dictionary mapping template name to description
    """
    return {
        name: template["description"]
        for name, template in PIPELINE_TEMPLATES.items()
    }


def get_templates_for_objective(objective: str) -> List[str]:
    """Get templates suitable for a specific objective.

    Args:
        objective: Optimization objective ('size', 'accuracy', 'latency', 'balanced')

    Returns:
        List of suitable template names
    """
    objective_mappings = {
        "size": [
            "aggressive_compression",
            "extreme_compression",
            "adaptive_compression",
        ],
        "accuracy": [
            "accuracy_recovery",
            "balanced_quality",
            "production_deployment",
        ],
        "latency": [
            "quick_compression",
            "pruning_focused",
            "llama_optimized",
        ],
        "balanced": [
            "balanced_quality",
            "adaptive_compression",
            "production_deployment",
        ],
        "memory": [
            "memory_efficient_large",
            "aggressive_compression",
        ],
    }

    return objective_mappings.get(objective.lower(), ["balanced_quality"])


def get_templates_for_model_family(model_family: str) -> List[str]:
    """Get templates optimized for a specific model family.

    Args:
        model_family: Model family (e.g., 'llama', 'mistral')

    Returns:
        List of suitable template names
    """
    family_mappings = {
        "llama": [
            "llama_optimized",
            "balanced_quality",
            "aggressive_compression",
        ],
        "mistral": [
            "production_deployment",
            "balanced_quality",
        ],
        "gpt": [
            "accuracy_recovery",
            "quick_compression",
        ],
        "falcon": [
            "balanced_quality",
            "adaptive_compression",
        ],
        "phi": [
            "quick_compression",
            "balanced_quality",
        ],
    }

    # Default templates for unknown families
    default = ["balanced_quality", "adaptive_compression", "quick_compression"]

    return family_mappings.get(model_family.lower(), default)


def get_templates_for_hardware(
    vram_gb: Optional[float] = None,
    model_size_gb: Optional[float] = None,
) -> List[str]:
    """Get templates suitable for available hardware.

    Args:
        vram_gb: Available VRAM in GB
        model_size_gb: Model size in GB

    Returns:
        List of suitable template names
    """
    templates = []

    # If VRAM is limited relative to model size
    if vram_gb is not None and model_size_gb is not None:
        ratio = vram_gb / model_size_gb

        if ratio < 1.0:
            # Very limited VRAM - need aggressive compression
            templates.extend([
                "memory_efficient_large",
                "extreme_compression",
            ])
        elif ratio < 2.0:
            # Moderate VRAM constraints
            templates.extend([
                "aggressive_compression",
                "quick_compression",
            ])
        else:
            # Comfortable VRAM
            templates.extend([
                "production_deployment",
                "balanced_quality",
            ])
    elif vram_gb is not None:
        if vram_gb < 8:
            templates.extend(["memory_efficient_large", "quick_compression"])
        elif vram_gb < 16:
            templates.extend(["adaptive_compression", "balanced_quality"])
        else:
            templates.extend(["production_deployment", "aggressive_compression"])
    else:
        # Unknown hardware - return safe defaults
        templates = ["balanced_quality", "quick_compression"]

    return templates


def generate_llm_template_prompt() -> str:
    """Generate a prompt describing available templates for LLM consumption.

    Returns:
        Formatted string describing available templates
    """
    lines = ["Available compression pipeline templates:\n"]

    for name, template in PIPELINE_TEMPLATES.items():
        steps_str = " -> ".join(s["skill"] for s in template["steps"])
        lines.append(f"\n{name}:")
        lines.append(f"  Description: {template['description']}")
        lines.append(f"  Steps: {steps_str}")

        # Note conditional steps
        conditional_steps = [
            s["skill"] for s in template["steps"]
            if "condition" in s
        ]
        if conditional_steps:
            lines.append(f"  Conditional steps: {', '.join(conditional_steps)}")

    return "\n".join(lines)


# Pre-built pipeline instances for common use cases

def create_aggressive_pipeline() -> CompressionPipeline:
    """Create the aggressive compression pipeline."""
    return get_template("aggressive_compression")


def create_balanced_pipeline() -> CompressionPipeline:
    """Create a balanced compression pipeline."""
    return get_template("balanced_quality")


def create_quick_pipeline() -> CompressionPipeline:
    """Create a quick compression pipeline."""
    return get_template("quick_compression")


def create_custom_pipeline(
    quantization_method: str = "gptq",
    quantization_bits: int = 4,
    pruning_sparsity: Optional[float] = None,
    lora_rank: Optional[int] = None,
    conditional_recovery: bool = True,
) -> CompressionPipeline:
    """Create a custom pipeline with specified parameters.

    Args:
        quantization_method: Quantization method to use
        quantization_bits: Bit width for quantization
        pruning_sparsity: Optional pruning sparsity (if None, skip pruning)
        lora_rank: Optional LoRA rank (if None, skip LoRA)
        conditional_recovery: Whether to make LoRA conditional on accuracy loss

    Returns:
        Configured CompressionPipeline
    """
    builder = PipelineBuilder("custom_pipeline")

    # Add pruning if specified
    if pruning_sparsity is not None:
        builder.add_pruning(method="magnitude", sparsity=pruning_sparsity)

    # Add quantization
    builder.add_quantization(method=quantization_method, bit_width=quantization_bits)

    # Add LoRA if specified
    if lora_rank is not None:
        if conditional_recovery:
            builder.add_conditional_recovery(
                accuracy_threshold=-0.05,
                lora_rank=lora_rank,
            )
        else:
            builder.add_finetuning(method="lora", lora_rank=lora_rank)

    return builder.build()


__all__ = [
    "PIPELINE_TEMPLATES",
    "get_template",
    "get_template_names",
    "get_template_descriptions",
    "get_templates_for_objective",
    "get_templates_for_model_family",
    "get_templates_for_hardware",
    "generate_llm_template_prompt",
    "create_aggressive_pipeline",
    "create_balanced_pipeline",
    "create_quick_pipeline",
    "create_custom_pipeline",
]
