"""Agent Skills system for dynamic skill discovery, composition, and learning.

This package provides three core capabilities:

1. **Dynamic Skill Discovery** (registry.py)
   - Automatically discover available compression methods at runtime
   - Check dependencies and hardware requirements
   - Generate LLM-friendly skill descriptions

2. **Skill Composition / Chaining** (pipeline.py, templates.py)
   - Chain multiple compression techniques (e.g., Pruning → Quantization → LoRA)
   - Support conditional execution based on intermediate results
   - Pre-defined pipeline templates for common scenarios

3. **Skill Learning** (memory.py)
   - Record skill execution results for learning
   - Query recommendations based on historical performance
   - Global shared memory across experiments

Example Usage:

    # Dynamic skill discovery
    from src.skills import SkillRegistry, get_available_skills

    available = get_available_skills()
    for skill in available:
        print(f"{skill.name}: {skill.description}")

    # Skill composition
    from src.skills import CompressionPipeline, get_template

    pipeline = get_template("aggressive_compression")
    result = pipeline.execute(model_path="meta-llama/Llama-2-7b-hf")

    # Skill learning
    from src.skills import get_global_memory, SkillContext, SkillResult

    memory = get_global_memory()
    recommendations = memory.query(SkillContext(
        model_family="llama",
        model_size_gb=14.0,
        target_objective="balanced",
    ))
"""

# Schema exports
from src.skills.schema import (
    # Enums
    SkillCategory,
    ParameterType,
    # Skill Definition
    ParameterSpec,
    HardwareSpec,
    SkillDefinition,
    # Pipeline
    StepCondition,
    PipelineStep,
    PipelineResult,
    # Learning
    SkillContext,
    SkillResult,
    SkillRecord,
    SkillStatistics,
    SkillRecommendation,
)

# Registry exports
from src.skills.registry import (
    SkillRegistry,
    register_skill,
    get_skill,
    get_available_skills,
    get_skills_prompt,
)

# Pipeline exports
from src.skills.pipeline import (
    CompressionPipeline,
    PipelineBuilder,
)

# Templates exports
from src.skills.templates import (
    PIPELINE_TEMPLATES,
    get_template,
    get_template_names,
    get_template_descriptions,
    get_templates_for_objective,
    get_templates_for_model_family,
    get_templates_for_hardware,
    generate_llm_template_prompt,
    create_aggressive_pipeline,
    create_balanced_pipeline,
    create_quick_pipeline,
    create_custom_pipeline,
)

# Memory exports
from src.skills.memory import (
    SkillMemory,
    get_global_memory,
    reset_global_memory,
    DEFAULT_MEMORY_PATH,
)


__all__ = [
    # Schema
    "SkillCategory",
    "ParameterType",
    "ParameterSpec",
    "HardwareSpec",
    "SkillDefinition",
    "StepCondition",
    "PipelineStep",
    "PipelineResult",
    "SkillContext",
    "SkillResult",
    "SkillRecord",
    "SkillStatistics",
    "SkillRecommendation",
    # Registry
    "SkillRegistry",
    "register_skill",
    "get_skill",
    "get_available_skills",
    "get_skills_prompt",
    # Pipeline
    "CompressionPipeline",
    "PipelineBuilder",
    # Templates
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
    # Memory
    "SkillMemory",
    "get_global_memory",
    "reset_global_memory",
    "DEFAULT_MEMORY_PATH",
]
