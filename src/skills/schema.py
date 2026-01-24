"""Skill definition schemas for the Agent Skills system.

This module defines the data models used for dynamic skill discovery,
skill composition (pipelines), and skill learning.
"""

from typing import Dict, List, Optional, Any, Callable, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class SkillCategory(str, Enum):
    """Categories of skills available in the system."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
    SEARCH = "search"
    ANALYSIS = "analysis"


class ParameterType(str, Enum):
    """Supported parameter types for skill parameters."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    ENUM = "enum"


class ParameterSpec(BaseModel):
    """Specification for a skill parameter."""
    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Description of the parameter")
    required: bool = Field(True, description="Whether this parameter is required")
    default: Optional[Any] = Field(None, description="Default value if not required")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric types")
    choices: Optional[List[Any]] = Field(None, description="Valid choices for enum types")

    class Config:
        use_enum_values = True


class HardwareSpec(BaseModel):
    """Hardware requirements specification."""
    min_vram_gb: Optional[float] = Field(None, description="Minimum VRAM in GB")
    recommended_vram_gb: Optional[float] = Field(None, description="Recommended VRAM in GB")
    requires_cuda: bool = Field(False, description="Whether CUDA is required")
    requires_gpu: bool = Field(False, description="Whether GPU is required")
    supports_cpu: bool = Field(True, description="Whether CPU execution is supported")
    min_ram_gb: Optional[float] = Field(None, description="Minimum RAM in GB")


class SkillDefinition(BaseModel):
    """Complete definition of a skill for dynamic discovery.

    This model contains all metadata needed for the coordinator to understand
    and use a skill, including its parameters, dependencies, and constraints.
    """
    name: str = Field(..., description="Unique skill identifier (e.g., 'quantization_gptq')")
    display_name: str = Field(..., description="Human-readable skill name")
    category: SkillCategory = Field(..., description="Skill category")
    description: str = Field(..., description="Detailed description for LLM to understand the skill")

    # Function reference (set at runtime during registration)
    tool_function_name: str = Field(..., description="Name of the tool function to call")
    tool_module: str = Field(..., description="Module path where the tool is defined")

    # Parameters
    parameters: List[ParameterSpec] = Field(
        default_factory=list,
        description="List of parameters the skill accepts"
    )

    # Dependencies
    dependencies: List[str] = Field(
        default_factory=list,
        description="Python package dependencies (e.g., ['auto_gptq', 'transformers'])"
    )

    # Hardware requirements
    hardware_requirements: HardwareSpec = Field(
        default_factory=HardwareSpec,
        description="Hardware requirements for this skill"
    )

    # Compatibility
    compatible_with: List[str] = Field(
        default_factory=list,
        description="List of skill names this skill can be combined with in a pipeline"
    )
    incompatible_with: List[str] = Field(
        default_factory=list,
        description="List of skill names that cannot be used with this skill"
    )

    # Performance characteristics
    estimated_time_factor: float = Field(
        1.0,
        description="Relative time cost factor (1.0 = baseline, 2.0 = twice as long)"
    )

    # Model compatibility
    compatible_model_families: List[str] = Field(
        default_factory=list,
        description="Model families this skill works best with (empty = all)"
    )

    # Metadata
    version: str = Field("1.0.0", description="Skill version")
    author: Optional[str] = Field(None, description="Skill author")

    class Config:
        use_enum_values = True

    def get_llm_description(self) -> str:
        """Generate a description suitable for LLM consumption."""
        params_desc = ""
        if self.parameters:
            param_lines = []
            for p in self.parameters:
                req = "(required)" if p.required else f"(default: {p.default})"
                choices = f", choices: {p.choices}" if p.choices else ""
                param_lines.append(f"    - {p.name}: {p.description} {req}{choices}")
            params_desc = "\n  Parameters:\n" + "\n".join(param_lines)

        return f"""- {self.display_name} ({self.name}):
  Category: {self.category}
  Description: {self.description}{params_desc}"""


# Pipeline-related schemas

class StepCondition(BaseModel):
    """Condition for conditional skill execution in a pipeline.

    Allows a pipeline step to execute based on metrics from previous steps.
    """
    metric: str = Field(
        ...,
        description="Metric to check (e.g., 'accuracy_delta', 'compression_ratio', 'accuracy')"
    )
    operator: Literal["gt", "lt", "eq", "gte", "lte", "ne"] = Field(
        ...,
        description="Comparison operator"
    )
    threshold: float = Field(..., description="Threshold value for comparison")
    on_false: Literal["skip", "abort", "fallback"] = Field(
        "skip",
        description="Action when condition is not met"
    )
    fallback_skill: Optional[str] = Field(
        None,
        description="Alternative skill to execute if on_false='fallback'"
    )
    fallback_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameters for the fallback skill"
    )

    def evaluate(self, value: float) -> bool:
        """Evaluate the condition against a value."""
        ops = {
            "gt": lambda x, t: x > t,
            "lt": lambda x, t: x < t,
            "eq": lambda x, t: abs(x - t) < 1e-6,
            "gte": lambda x, t: x >= t,
            "lte": lambda x, t: x <= t,
            "ne": lambda x, t: abs(x - t) >= 1e-6,
        }
        return ops[self.operator](value, self.threshold)


class PipelineStep(BaseModel):
    """A single step in a compression pipeline."""
    skill_name: str = Field(..., description="Name of the skill to execute")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the skill"
    )
    checkpoint_strategy: Literal["keep", "delete", "archive"] = Field(
        "keep",
        description="How to handle intermediate checkpoints"
    )
    condition: Optional[StepCondition] = Field(
        None,
        description="Optional condition for executing this step"
    )
    name: Optional[str] = Field(
        None,
        description="Optional name for this step (for logging)"
    )

    def get_display_name(self) -> str:
        """Get a display name for this step."""
        return self.name or self.skill_name


class PipelineResult(BaseModel):
    """Result from executing a compression pipeline."""
    success: bool = Field(..., description="Whether the pipeline completed successfully")
    final_checkpoint_path: Optional[str] = Field(
        None,
        description="Path to the final compressed model"
    )
    steps_executed: int = Field(..., description="Number of steps that were executed")
    steps_skipped: int = Field(0, description="Number of steps skipped due to conditions")
    step_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from each executed step"
    )
    total_time_sec: float = Field(..., description="Total execution time")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Aggregate metrics
    final_compression_ratio: Optional[float] = Field(None)
    final_model_size_gb: Optional[float] = Field(None)
    accuracy_delta: Optional[float] = Field(None)


# Skill Learning schemas

class SkillContext(BaseModel):
    """Context information for skill learning.

    Captures the state when a skill was executed to enable learning
    which skills work best in which situations.
    """
    model_family: str = Field(..., description="Model family (e.g., 'llama', 'mistral')")
    model_size_gb: float = Field(..., description="Model size in GB")
    target_objective: str = Field(
        "balanced",
        description="Primary objective: 'accuracy', 'latency', 'size', 'balanced'"
    )
    hardware_profile: str = Field(
        "unknown",
        description="Hardware profile description"
    )
    available_vram_gb: Optional[float] = Field(None, description="Available VRAM")
    previous_skills: List[str] = Field(
        default_factory=list,
        description="Skills already executed in this session"
    )
    current_pareto_gap: str = Field(
        "balanced",
        description="Current Pareto frontier gap: 'need_accuracy', 'need_speed', 'balanced'"
    )
    baseline_accuracy: Optional[float] = Field(None, description="Baseline model accuracy")
    current_accuracy: Optional[float] = Field(None, description="Current accuracy after previous skills")
    current_compression_ratio: Optional[float] = Field(1.0, description="Current compression ratio")


class SkillResult(BaseModel):
    """Result of executing a skill, used for learning."""
    skill_name: str = Field(..., description="Name of the executed skill")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for execution"
    )
    accuracy_delta: float = Field(
        0.0,
        description="Change in accuracy (negative = accuracy loss)"
    )
    latency_delta: float = Field(
        0.0,
        description="Change in latency in ms (negative = improvement)"
    )
    size_delta: float = Field(
        0.0,
        description="Change in model size in GB (negative = smaller)"
    )
    compression_ratio: float = Field(
        1.0,
        description="Compression ratio achieved"
    )
    is_pareto_optimal: bool = Field(
        False,
        description="Whether this result is on the Pareto frontier"
    )
    execution_time_sec: float = Field(..., description="Execution time in seconds")
    success: bool = Field(True, description="Whether the skill executed successfully")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class SkillRecord(BaseModel):
    """A single record in the skill memory."""
    context: SkillContext
    result: SkillResult
    timestamp: datetime = Field(default_factory=datetime.now)
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")


class SkillStatistics(BaseModel):
    """Aggregated statistics for a skill."""
    skill_name: str = Field(..., description="Name of the skill")
    total_executions: int = Field(0, description="Total number of executions")
    success_rate: float = Field(0.0, description="Success rate (0.0 to 1.0)")
    avg_accuracy_delta: float = Field(0.0, description="Average accuracy change")
    avg_latency_delta: float = Field(0.0, description="Average latency change")
    avg_compression_ratio: float = Field(1.0, description="Average compression ratio")
    avg_execution_time_sec: float = Field(0.0, description="Average execution time")
    pareto_hit_rate: float = Field(0.0, description="Rate of Pareto-optimal results")
    best_for_model_families: List[str] = Field(
        default_factory=list,
        description="Model families where this skill performs best"
    )
    best_for_objectives: List[str] = Field(
        default_factory=list,
        description="Objectives where this skill excels"
    )
    common_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Most commonly used parameter combinations"
    )


class SkillRecommendation(BaseModel):
    """A skill recommendation based on context and history."""
    skill_name: str = Field(..., description="Recommended skill name")
    confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0)"
    )
    expected_accuracy_delta: float = Field(0.0, description="Expected accuracy change")
    expected_compression_ratio: float = Field(1.0, description="Expected compression")
    reasoning: str = Field(..., description="Explanation for this recommendation")
    suggested_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested parameters based on history"
    )


# Export all schemas
__all__ = [
    # Enums
    "SkillCategory",
    "ParameterType",
    # Skill Definition
    "ParameterSpec",
    "HardwareSpec",
    "SkillDefinition",
    # Pipeline
    "StepCondition",
    "PipelineStep",
    "PipelineResult",
    # Learning
    "SkillContext",
    "SkillResult",
    "SkillRecord",
    "SkillStatistics",
    "SkillRecommendation",
]
