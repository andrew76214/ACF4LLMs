"""Dynamic Skill Registry for the Agent Skills system.

This module provides a central registry for discovering and managing
available compression skills at runtime.
"""

import importlib
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import lru_cache

import torch

from src.skills.schema import (
    SkillDefinition,
    SkillCategory,
    ParameterSpec,
    ParameterType,
    HardwareSpec,
)

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Dynamic skill registration center.

    The registry maintains a collection of available skills and provides
    methods to discover, register, and query skills based on various criteria.

    This class uses a singleton-like pattern with class-level storage to
    ensure skills are shared across the application.
    """

    _skills: Dict[str, SkillDefinition] = {}
    _tool_functions: Dict[str, Callable] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, skill: SkillDefinition, tool_function: Optional[Callable] = None) -> None:
        """Register a skill in the registry.

        Args:
            skill: The skill definition to register
            tool_function: Optional tool function reference (can be loaded lazily)
        """
        cls._skills[skill.name] = skill
        if tool_function is not None:
            cls._tool_functions[skill.name] = tool_function
        logger.debug(f"Registered skill: {skill.name}")

    @classmethod
    def unregister(cls, skill_name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            skill_name: Name of the skill to remove

        Returns:
            True if skill was removed, False if not found
        """
        if skill_name in cls._skills:
            del cls._skills[skill_name]
            if skill_name in cls._tool_functions:
                del cls._tool_functions[skill_name]
            logger.debug(f"Unregistered skill: {skill_name}")
            return True
        return False

    @classmethod
    def get(cls, skill_name: str) -> Optional[SkillDefinition]:
        """Get a skill definition by name.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillDefinition if found, None otherwise
        """
        cls._ensure_initialized()
        return cls._skills.get(skill_name)

    @classmethod
    def get_tool_function(cls, skill_name: str) -> Optional[Callable]:
        """Get the tool function for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Tool function if available, None otherwise
        """
        cls._ensure_initialized()

        # Return cached function if available
        if skill_name in cls._tool_functions:
            return cls._tool_functions[skill_name]

        # Try to load the function dynamically
        skill = cls._skills.get(skill_name)
        if skill is None:
            return None

        try:
            module = importlib.import_module(skill.tool_module)
            func = getattr(module, skill.tool_function_name, None)
            if func is not None:
                cls._tool_functions[skill_name] = func
            return func
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load tool function for {skill_name}: {e}")
            return None

    @classmethod
    def discover(cls) -> List[SkillDefinition]:
        """Discover all registered skills.

        Returns:
            List of all registered skill definitions
        """
        cls._ensure_initialized()
        return list(cls._skills.values())

    @classmethod
    def get_available(cls) -> List[SkillDefinition]:
        """Get skills that are actually available (dependencies installed).

        Returns:
            List of available skill definitions
        """
        cls._ensure_initialized()
        available = []

        for skill in cls._skills.values():
            if cls._check_dependencies(skill):
                available.append(skill)

        return available

    @classmethod
    def get_by_category(cls, category: SkillCategory) -> List[SkillDefinition]:
        """Get all skills in a specific category.

        Args:
            category: The skill category to filter by

        Returns:
            List of skills in the category
        """
        cls._ensure_initialized()
        return [
            skill for skill in cls._skills.values()
            if skill.category == category or skill.category == category.value
        ]

    @classmethod
    def get_compatible_skills(cls, skill_name: str) -> List[SkillDefinition]:
        """Get skills that can be combined with a given skill.

        Args:
            skill_name: Name of the skill to check compatibility for

        Returns:
            List of compatible skills
        """
        cls._ensure_initialized()
        skill = cls._skills.get(skill_name)
        if skill is None:
            return []

        compatible = []
        for other_skill in cls._skills.values():
            if other_skill.name == skill_name:
                continue

            # Check if explicitly compatible
            if other_skill.name in skill.compatible_with:
                compatible.append(other_skill)
            # Check if not explicitly incompatible
            elif (other_skill.name not in skill.incompatible_with and
                  skill_name not in other_skill.incompatible_with):
                compatible.append(other_skill)

        return compatible

    @classmethod
    def get_skills_for_model(cls, model_family: str) -> List[SkillDefinition]:
        """Get skills recommended for a specific model family.

        Args:
            model_family: Model family (e.g., 'llama', 'mistral')

        Returns:
            List of recommended skills
        """
        cls._ensure_initialized()
        available = cls.get_available()

        recommended = []
        for skill in available:
            # If no family restrictions, include the skill
            if not skill.compatible_model_families:
                recommended.append(skill)
            # If family is in compatible list, include it
            elif model_family.lower() in [f.lower() for f in skill.compatible_model_families]:
                recommended.append(skill)

        return recommended

    @classmethod
    def get_skills_for_hardware(
        cls,
        vram_gb: Optional[float] = None,
        has_cuda: bool = True,
    ) -> List[SkillDefinition]:
        """Get skills compatible with available hardware.

        Args:
            vram_gb: Available VRAM in GB (None = unknown)
            has_cuda: Whether CUDA is available

        Returns:
            List of compatible skills
        """
        cls._ensure_initialized()
        available = cls.get_available()

        compatible = []
        for skill in available:
            hw = skill.hardware_requirements

            # Check CUDA requirement
            if hw.requires_cuda and not has_cuda:
                continue

            # Check GPU requirement
            if hw.requires_gpu and not has_cuda:
                # Check if CPU is supported as fallback
                if not hw.supports_cpu:
                    continue

            # Check VRAM requirement
            if vram_gb is not None and hw.min_vram_gb is not None:
                if vram_gb < hw.min_vram_gb:
                    continue

            compatible.append(skill)

        return compatible

    @classmethod
    def generate_llm_prompt(cls, include_unavailable: bool = False) -> str:
        """Generate a prompt describing available skills for LLM consumption.

        Args:
            include_unavailable: Whether to include unavailable skills

        Returns:
            Formatted string describing available skills
        """
        cls._ensure_initialized()

        skills = cls.get_available() if not include_unavailable else cls.discover()

        if not skills:
            return "No skills are currently available."

        # Group by category
        by_category: Dict[str, List[SkillDefinition]] = {}
        for skill in skills:
            cat = skill.category if isinstance(skill.category, str) else skill.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(skill)

        lines = ["Available compression skills:\n"]

        for category, cat_skills in sorted(by_category.items()):
            lines.append(f"\n## {category.upper()}\n")
            for skill in cat_skills:
                lines.append(skill.get_llm_description())
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def _check_dependencies(cls, skill: SkillDefinition) -> bool:
        """Check if a skill's dependencies are installed.

        Args:
            skill: The skill to check

        Returns:
            True if all dependencies are available
        """
        for dep in skill.dependencies:
            if not cls._is_package_available(dep):
                logger.debug(f"Skill {skill.name} unavailable: missing {dep}")
                return False

        # Check hardware requirements
        hw = skill.hardware_requirements
        if hw.requires_cuda and not torch.cuda.is_available():
            logger.debug(f"Skill {skill.name} unavailable: requires CUDA")
            return False

        return True

    @classmethod
    @lru_cache(maxsize=128)
    def _is_package_available(cls, package_name: str) -> bool:
        """Check if a Python package is available.

        Args:
            package_name: Name of the package to check

        Returns:
            True if the package can be imported
        """
        # Handle package name variations
        name_mappings = {
            "auto_gptq": "auto_gptq",
            "auto-gptq": "auto_gptq",
            "auto_round": "auto_round",
            "autoround": "auto_round",
            "awq": "awq",
            "bitsandbytes": "bitsandbytes",
            "peft": "peft",
            "transformers": "transformers",
            "torch": "torch",
            "datasets": "datasets",
            "codecarbon": "codecarbon",
            "lm_eval": "lm_eval",
            "scipy": "scipy",
            "optuna": "optuna",
        }

        actual_name = name_mappings.get(package_name, package_name)

        try:
            importlib.import_module(actual_name)
            return True
        except ImportError:
            return False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure the registry is initialized with default skills."""
        if not cls._initialized:
            cls._initialize_default_skills()
            cls._initialized = True

    @classmethod
    def _initialize_default_skills(cls) -> None:
        """Initialize the registry with default skill definitions."""
        logger.info("Initializing default skill definitions...")

        # Register quantization skills
        cls._register_quantization_skills()

        # Register pruning skills
        cls._register_pruning_skills()

        # Register fine-tuning skills
        cls._register_finetuning_skills()

        # Register evaluation skills
        cls._register_evaluation_skills()

        # Register search skills
        cls._register_search_skills()

        logger.info(f"Registered {len(cls._skills)} skills")

    @classmethod
    def _register_quantization_skills(cls) -> None:
        """Register quantization-related skills."""

        # GPTQ Quantization
        cls.register(SkillDefinition(
            name="quantization_gptq",
            display_name="GPTQ Quantization",
            category=SkillCategory.QUANTIZATION,
            description="Post-training quantization with layer-wise optimization. "
                       "Provides excellent compression (4-bit) with good accuracy retention. "
                       "Best for memory-constrained deployments.",
            tool_function_name="quantize_model",
            tool_module="src.agents.quantization_agent",
            parameters=[
                ParameterSpec(
                    name="bit_width",
                    type=ParameterType.INT,
                    description="Quantization bit width",
                    required=False,
                    default=4,
                    min_value=2,
                    max_value=8,
                    choices=[2, 3, 4, 8],
                ),
                ParameterSpec(
                    name="calibration_samples",
                    type=ParameterType.INT,
                    description="Number of calibration samples",
                    required=False,
                    default=512,
                    min_value=128,
                    max_value=2048,
                ),
            ],
            dependencies=["auto_gptq", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=8.0,
                requires_cuda=True,
            ),
            compatible_with=["pruning_magnitude", "pruning_structured", "lora_finetune"],
            estimated_time_factor=1.5,
            compatible_model_families=["llama", "mistral", "gpt", "falcon", "phi"],
        ))

        # AutoRound Quantization
        cls.register(SkillDefinition(
            name="quantization_autoround",
            display_name="AutoRound Quantization",
            category=SkillCategory.QUANTIZATION,
            description="Adaptive rounding for weight quantization. "
                       "Better accuracy retention than GPTQ but slower. "
                       "Recommended for production deployments requiring high quality.",
            tool_function_name="quantize_model",
            tool_module="src.agents.quantization_agent",
            parameters=[
                ParameterSpec(
                    name="bit_width",
                    type=ParameterType.INT,
                    description="Quantization bit width",
                    required=False,
                    default=4,
                    choices=[4, 8],
                ),
                ParameterSpec(
                    name="calibration_samples",
                    type=ParameterType.INT,
                    description="Number of calibration samples",
                    required=False,
                    default=512,
                ),
            ],
            dependencies=["auto_round", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=12.0,
                requires_cuda=True,
            ),
            compatible_with=["pruning_magnitude", "lora_finetune"],
            estimated_time_factor=2.5,
            compatible_model_families=["llama", "mistral", "gpt"],
        ))

        # INT8 Quantization
        cls.register(SkillDefinition(
            name="quantization_int8",
            display_name="INT8 Quantization",
            category=SkillCategory.QUANTIZATION,
            description="Simple 8-bit integer quantization using BitsAndBytes. "
                       "Fast to apply with good hardware support. "
                       "Limited compression ratio but minimal accuracy loss.",
            tool_function_name="quantize_model",
            tool_module="src.agents.quantization_agent",
            parameters=[],  # INT8 doesn't need bit_width parameter
            dependencies=["bitsandbytes", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=4.0,
                requires_cuda=True,
            ),
            compatible_with=["pruning_magnitude", "pruning_structured", "lora_finetune", "qlora_finetune"],
            estimated_time_factor=0.5,
        ))

        # AWQ Quantization
        cls.register(SkillDefinition(
            name="quantization_awq",
            display_name="AWQ Quantization",
            category=SkillCategory.QUANTIZATION,
            description="Activation-aware weight quantization. "
                       "Preserves important weights based on activation patterns. "
                       "Best accuracy retention at 4-bit, optimized for Llama models.",
            tool_function_name="quantize_model",
            tool_module="src.agents.quantization_agent",
            parameters=[
                ParameterSpec(
                    name="bit_width",
                    type=ParameterType.INT,
                    description="Quantization bit width (AWQ only supports 4-bit)",
                    required=False,
                    default=4,
                    choices=[4],
                ),
            ],
            dependencies=["awq", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=8.0,
                requires_cuda=True,
            ),
            compatible_with=["lora_finetune"],
            estimated_time_factor=1.0,
            compatible_model_families=["llama"],
        ))

    @classmethod
    def _register_pruning_skills(cls) -> None:
        """Register pruning-related skills."""

        # Magnitude Pruning
        cls.register(SkillDefinition(
            name="pruning_magnitude",
            display_name="Magnitude Pruning",
            category=SkillCategory.PRUNING,
            description="Unstructured pruning based on weight magnitude. "
                       "Removes individual weights with smallest magnitudes. "
                       "Simple and works with any model, but requires sparse format for storage savings.",
            tool_function_name="prune_model",
            tool_module="src.agents.pruning_agent",
            parameters=[
                ParameterSpec(
                    name="sparsity",
                    type=ParameterType.FLOAT,
                    description="Target sparsity ratio (0.0 to 0.9)",
                    required=False,
                    default=0.3,
                    min_value=0.1,
                    max_value=0.9,
                ),
            ],
            dependencies=["torch", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=4.0,
                supports_cpu=True,
            ),
            compatible_with=[
                "quantization_gptq", "quantization_autoround", "quantization_int8",
                "lora_finetune", "qlora_finetune"
            ],
            estimated_time_factor=0.5,
        ))

        # Structured Pruning
        cls.register(SkillDefinition(
            name="pruning_structured",
            display_name="Structured Pruning",
            category=SkillCategory.PRUNING,
            description="Remove entire channels or attention heads. "
                       "Provides actual model size reduction and inference speedup. "
                       "More aggressive than magnitude pruning but with real benefits.",
            tool_function_name="prune_model",
            tool_module="src.agents.pruning_agent",
            parameters=[
                ParameterSpec(
                    name="sparsity",
                    type=ParameterType.FLOAT,
                    description="Target sparsity ratio",
                    required=False,
                    default=0.2,
                    min_value=0.1,
                    max_value=0.5,
                ),
                ParameterSpec(
                    name="granularity",
                    type=ParameterType.ENUM,
                    description="Pruning granularity",
                    required=False,
                    default="channel",
                    choices=["channel", "head"],
                ),
            ],
            dependencies=["torch", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=8.0,
                requires_cuda=True,
            ),
            compatible_with=["quantization_gptq", "quantization_int8", "lora_finetune"],
            estimated_time_factor=1.0,
            compatible_model_families=["llama", "mistral", "gpt", "bert", "falcon", "phi"],
        ))

    @classmethod
    def _register_finetuning_skills(cls) -> None:
        """Register fine-tuning related skills."""

        # LoRA Fine-tuning
        cls.register(SkillDefinition(
            name="lora_finetune",
            display_name="LoRA Fine-tuning",
            category=SkillCategory.FINETUNING,
            description="Low-Rank Adaptation - adds small trainable adapters to frozen base model. "
                       "Efficient fine-tuning without modifying original weights. "
                       "Use for accuracy recovery after aggressive compression.",
            tool_function_name="apply_lora_finetuning",
            tool_module="src.agents.quantization_agent",
            parameters=[
                ParameterSpec(
                    name="lora_rank",
                    type=ParameterType.INT,
                    description="LoRA rank (higher = more capacity)",
                    required=False,
                    default=16,
                    choices=[8, 16, 32, 64],
                ),
                ParameterSpec(
                    name="lora_alpha",
                    type=ParameterType.INT,
                    description="LoRA scaling factor (typically 2x rank)",
                    required=False,
                    default=32,
                ),
                ParameterSpec(
                    name="training_steps",
                    type=ParameterType.INT,
                    description="Number of training steps",
                    required=False,
                    default=100,
                    min_value=50,
                    max_value=1000,
                ),
            ],
            dependencies=["peft", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=8.0,
                requires_cuda=True,
            ),
            compatible_with=[
                "quantization_gptq", "quantization_autoround", "quantization_int8",
                "pruning_magnitude", "pruning_structured"
            ],
            estimated_time_factor=2.0,
        ))

        # QLoRA Fine-tuning
        cls.register(SkillDefinition(
            name="qlora_finetune",
            display_name="QLoRA Fine-tuning",
            category=SkillCategory.FINETUNING,
            description="Quantized LoRA - loads base model in 4-bit and trains LoRA adapters. "
                       "Enables fine-tuning of large models on consumer GPUs. "
                       "Memory-efficient alternative to standard LoRA.",
            tool_function_name="apply_qlora_finetuning",
            tool_module="src.agents.quantization_agent",
            parameters=[
                ParameterSpec(
                    name="lora_rank",
                    type=ParameterType.INT,
                    description="LoRA rank",
                    required=False,
                    default=16,
                    choices=[8, 16, 32, 64],
                ),
                ParameterSpec(
                    name="bits",
                    type=ParameterType.INT,
                    description="Base model quantization bits",
                    required=False,
                    default=4,
                    choices=[4],
                ),
                ParameterSpec(
                    name="training_steps",
                    type=ParameterType.INT,
                    description="Number of training steps",
                    required=False,
                    default=100,
                ),
            ],
            dependencies=["peft", "bitsandbytes", "transformers"],
            hardware_requirements=HardwareSpec(
                min_vram_gb=4.0,
                requires_cuda=True,
            ),
            compatible_with=["pruning_magnitude"],
            incompatible_with=["quantization_gptq", "quantization_awq"],  # Already quantized
            estimated_time_factor=2.5,
        ))

    @classmethod
    def _register_evaluation_skills(cls) -> None:
        """Register evaluation-related skills."""

        cls.register(SkillDefinition(
            name="evaluate_model",
            display_name="Model Evaluation",
            category=SkillCategory.EVALUATION,
            description="Evaluate model on benchmarks using lm-eval harness. "
                       "Measures accuracy, latency, and memory usage.",
            tool_function_name="evaluate_model",
            tool_module="src.agents.evaluation_agent",
            parameters=[
                ParameterSpec(
                    name="benchmarks",
                    type=ParameterType.LIST,
                    description="List of benchmark names",
                    required=True,
                ),
                ParameterSpec(
                    name="use_proxy",
                    type=ParameterType.BOOL,
                    description="Use proxy (fast) evaluation",
                    required=False,
                    default=True,
                ),
            ],
            dependencies=["transformers", "datasets"],
            hardware_requirements=HardwareSpec(
                supports_cpu=True,
            ),
            estimated_time_factor=1.0,
        ))

    @classmethod
    def _register_search_skills(cls) -> None:
        """Register search/optimization skills."""

        cls.register(SkillDefinition(
            name="bayesian_search",
            display_name="Bayesian Optimization Search",
            category=SkillCategory.SEARCH,
            description="Use Bayesian optimization to suggest promising strategies. "
                       "Learns from history to balance exploration and exploitation.",
            tool_function_name="bayesian_optimization_search",
            tool_module="src.agents.search_agent",
            parameters=[
                ParameterSpec(
                    name="n_suggestions",
                    type=ParameterType.INT,
                    description="Number of suggestions to generate",
                    required=False,
                    default=3,
                ),
            ],
            dependencies=["scipy"],
            hardware_requirements=HardwareSpec(
                supports_cpu=True,
            ),
            estimated_time_factor=0.1,
        ))

        cls.register(SkillDefinition(
            name="evolutionary_search",
            display_name="Evolutionary Search",
            category=SkillCategory.SEARCH,
            description="Use evolutionary algorithms to explore strategy space. "
                       "Good for finding novel combinations.",
            tool_function_name="evolutionary_search",
            tool_module="src.agents.search_agent",
            parameters=[
                ParameterSpec(
                    name="population_size",
                    type=ParameterType.INT,
                    description="Population size",
                    required=False,
                    default=10,
                ),
            ],
            dependencies=["torch"],
            hardware_requirements=HardwareSpec(
                supports_cpu=True,
            ),
            estimated_time_factor=0.2,
        ))

    @classmethod
    def reset(cls) -> None:
        """Reset the registry to empty state. Useful for testing."""
        cls._skills.clear()
        cls._tool_functions.clear()
        cls._initialized = False
        cls._is_package_available.cache_clear()


# Convenience functions for module-level access

def register_skill(skill: SkillDefinition, tool_function: Optional[Callable] = None) -> None:
    """Register a skill in the global registry."""
    SkillRegistry.register(skill, tool_function)


def get_skill(skill_name: str) -> Optional[SkillDefinition]:
    """Get a skill from the global registry."""
    return SkillRegistry.get(skill_name)


def get_available_skills() -> List[SkillDefinition]:
    """Get all available skills from the global registry."""
    return SkillRegistry.get_available()


def get_skills_prompt() -> str:
    """Generate an LLM prompt describing available skills."""
    return SkillRegistry.generate_llm_prompt()


__all__ = [
    "SkillRegistry",
    "register_skill",
    "get_skill",
    "get_available_skills",
    "get_skills_prompt",
]
