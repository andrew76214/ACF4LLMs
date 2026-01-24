"""Compression Pipeline for skill composition and chaining.

This module provides the ability to chain multiple compression skills
together in a pipeline, with support for conditional execution.
"""

import gc
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch

from src.skills.schema import (
    PipelineStep,
    PipelineResult,
    StepCondition,
    SkillResult,
)
from src.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class CompressionPipeline:
    """Multi-skill compression pipeline with conditional execution.

    Allows chaining multiple compression techniques (e.g., Pruning → Quantization → LoRA)
    with support for conditional steps based on intermediate results.

    Example:
        pipeline = CompressionPipeline([
            PipelineStep(skill_name="pruning_magnitude", params={"sparsity": 0.3}),
            PipelineStep(skill_name="quantization_gptq", params={"bit_width": 4}),
            PipelineStep(
                skill_name="lora_finetune",
                params={"lora_rank": 16},
                condition=StepCondition(
                    metric="accuracy_delta",
                    operator="lt",
                    threshold=-0.05,
                    on_false="skip"
                )
            ),
        ])

        result = pipeline.execute(model_path="meta-llama/Llama-2-7b-hf")
    """

    MAX_STEPS = 5  # Maximum number of steps in a pipeline

    def __init__(
        self,
        steps: List[PipelineStep],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the compression pipeline.

        Args:
            steps: List of pipeline steps to execute
            name: Optional name for the pipeline
            description: Optional description

        Raises:
            ValueError: If pipeline has more than MAX_STEPS
        """
        if len(steps) > self.MAX_STEPS:
            raise ValueError(
                f"Pipeline cannot have more than {self.MAX_STEPS} steps, "
                f"got {len(steps)}"
            )

        self.steps = steps
        self.name = name or f"pipeline_{len(steps)}_steps"
        self.description = description or ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionPipeline":
        """Create a pipeline from a dictionary specification.

        Args:
            data: Dictionary with pipeline specification

        Returns:
            CompressionPipeline instance
        """
        steps = []
        for step_data in data.get("steps", []):
            # Handle condition if present
            condition = None
            if "condition" in step_data:
                condition = StepCondition(**step_data["condition"])

            steps.append(PipelineStep(
                skill_name=step_data["skill"],
                params=step_data.get("params", {}),
                checkpoint_strategy=step_data.get("checkpoint_strategy", "keep"),
                condition=condition,
                name=step_data.get("name"),
            ))

        return cls(
            steps=steps,
            name=data.get("name"),
            description=data.get("description"),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the pipeline configuration.

        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        messages = []
        is_valid = True

        # Check step count
        if len(self.steps) > self.MAX_STEPS:
            messages.append(f"Error: Pipeline exceeds maximum of {self.MAX_STEPS} steps")
            is_valid = False

        if len(self.steps) == 0:
            messages.append("Error: Pipeline has no steps")
            is_valid = False
            return is_valid, messages

        # Check each step
        for i, step in enumerate(self.steps):
            skill = SkillRegistry.get(step.skill_name)

            if skill is None:
                messages.append(f"Warning: Step {i+1} uses unknown skill '{step.skill_name}'")
                continue

            # Check if skill is available
            available_skills = [s.name for s in SkillRegistry.get_available()]
            if step.skill_name not in available_skills:
                messages.append(
                    f"Warning: Step {i+1} skill '{step.skill_name}' "
                    "may not be available (missing dependencies)"
                )

            # Check compatibility with previous step
            if i > 0:
                prev_step = self.steps[i - 1]
                prev_skill = SkillRegistry.get(prev_step.skill_name)

                if prev_skill and skill:
                    if step.skill_name in prev_skill.incompatible_with:
                        messages.append(
                            f"Error: Step {i+1} '{step.skill_name}' is incompatible "
                            f"with step {i} '{prev_step.skill_name}'"
                        )
                        is_valid = False

            # Validate condition if present
            if step.condition:
                if step.condition.on_false == "fallback" and not step.condition.fallback_skill:
                    messages.append(
                        f"Warning: Step {i+1} has fallback action but no fallback_skill specified"
                    )

        return is_valid, messages

    def execute(
        self,
        model_path: str,
        state: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        baseline_accuracy: Optional[float] = None,
    ) -> PipelineResult:
        """Execute the compression pipeline.

        Args:
            model_path: Path to the input model or HuggingFace model name
            state: Optional state dictionary with additional context
            output_dir: Directory for intermediate and final outputs
            baseline_accuracy: Baseline accuracy for computing deltas

        Returns:
            PipelineResult with execution results
        """
        start_time = time.time()
        logger.info(f"Executing pipeline '{self.name}' with {len(self.steps)} steps")

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"data/pipelines/{self.name}_{timestamp}"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        current_model_path = model_path
        step_results: List[Dict[str, Any]] = []
        steps_executed = 0
        steps_skipped = 0
        cumulative_metrics = {
            "compression_ratio": 1.0,
            "model_size_gb": None,
            "accuracy_delta": 0.0,
            "accuracy": baseline_accuracy,
        }

        try:
            for i, step in enumerate(self.steps):
                step_name = step.get_display_name()
                logger.info(f"Step {i+1}/{len(self.steps)}: {step_name}")

                # Check condition if present
                if step.condition:
                    should_execute, action = self._evaluate_condition(
                        step.condition, cumulative_metrics
                    )

                    if not should_execute:
                        logger.info(f"Condition not met, action: {action}")

                        if action == "skip":
                            steps_skipped += 1
                            step_results.append({
                                "step": i + 1,
                                "skill": step.skill_name,
                                "status": "skipped",
                                "reason": "condition_not_met",
                            })
                            continue
                        elif action == "abort":
                            logger.info("Aborting pipeline due to condition")
                            break
                        elif action == "fallback":
                            # Execute fallback skill instead
                            if step.condition.fallback_skill:
                                logger.info(f"Executing fallback: {step.condition.fallback_skill}")
                                step = PipelineStep(
                                    skill_name=step.condition.fallback_skill,
                                    params=step.condition.fallback_params or {},
                                    checkpoint_strategy=step.checkpoint_strategy,
                                )
                            else:
                                steps_skipped += 1
                                continue

                # Execute the step
                step_output_dir = f"{output_dir}/step_{i+1:02d}_{step.skill_name}"
                Path(step_output_dir).mkdir(parents=True, exist_ok=True)

                result = self._execute_step(
                    step=step,
                    model_path=current_model_path,
                    output_dir=step_output_dir,
                    state=state,
                )

                step_results.append({
                    "step": i + 1,
                    "skill": step.skill_name,
                    "status": "success" if result.get("success", True) else "failed",
                    "result": result,
                })

                steps_executed += 1

                # Update tracking
                if "checkpoint_path" in result and result["checkpoint_path"]:
                    # Handle checkpoint strategy
                    if step.checkpoint_strategy == "delete" and current_model_path != model_path:
                        self._cleanup_checkpoint(current_model_path)
                    elif step.checkpoint_strategy == "archive" and current_model_path != model_path:
                        self._archive_checkpoint(current_model_path, output_dir)

                    current_model_path = result["checkpoint_path"]

                # Update cumulative metrics
                self._update_cumulative_metrics(cumulative_metrics, result, baseline_accuracy)

                # Check for failure
                if not result.get("success", True):
                    logger.error(f"Step {i+1} failed: {result.get('error')}")
                    break

                # Clean up GPU memory between steps
                self._cleanup_gpu_between_steps()

            # Build final result
            total_time = time.time() - start_time

            return PipelineResult(
                success=steps_executed > 0 and step_results[-1].get("status") == "success",
                final_checkpoint_path=current_model_path if current_model_path != model_path else None,
                steps_executed=steps_executed,
                steps_skipped=steps_skipped,
                step_results=step_results,
                total_time_sec=total_time,
                final_compression_ratio=cumulative_metrics.get("compression_ratio"),
                final_model_size_gb=cumulative_metrics.get("model_size_gb"),
                accuracy_delta=cumulative_metrics.get("accuracy_delta"),
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return PipelineResult(
                success=False,
                final_checkpoint_path=None,
                steps_executed=steps_executed,
                steps_skipped=steps_skipped,
                step_results=step_results,
                total_time_sec=time.time() - start_time,
                error_message=str(e),
            )

    def _execute_step(
        self,
        step: PipelineStep,
        model_path: str,
        output_dir: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single pipeline step.

        Args:
            step: The step to execute
            model_path: Path to the input model
            output_dir: Output directory for this step
            state: Optional state context

        Returns:
            Dictionary with step results
        """
        skill = SkillRegistry.get(step.skill_name)
        if skill is None:
            return {"success": False, "error": f"Unknown skill: {step.skill_name}"}

        tool_func = SkillRegistry.get_tool_function(step.skill_name)
        if tool_func is None:
            return {"success": False, "error": f"Could not load tool function for: {step.skill_name}"}

        # Build parameters
        params = {**step.params}

        # Get category value (handle both string and enum)
        category = skill.category if isinstance(skill.category, str) else skill.category.value

        # Add model path with appropriate parameter name
        if category == "quantization":
            params["model_path"] = model_path
            params["output_dir"] = output_dir
            # Set method for quantization skills
            if "quantization_" in step.skill_name:
                method = step.skill_name.replace("quantization_", "")
                params["method"] = method
        elif category == "pruning":
            params["model_path"] = model_path
            params["output_dir"] = output_dir
            # Set method for pruning skills
            if "pruning_" in step.skill_name:
                method = step.skill_name.replace("pruning_", "")
                params["method"] = method
        elif category == "finetuning":
            params["model_path"] = model_path
            params["output_dir"] = output_dir

        try:
            # Execute the tool function
            logger.info(f"Executing {step.skill_name} with params: {params}")
            result = tool_func.invoke(params)

            if isinstance(result, dict):
                result["success"] = True
                return result
            else:
                return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _evaluate_condition(
        self,
        condition: StepCondition,
        metrics: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Evaluate a step condition against current metrics.

        Args:
            condition: The condition to evaluate
            metrics: Current cumulative metrics

        Returns:
            Tuple of (should_execute, action_if_false)
        """
        metric_value = metrics.get(condition.metric)

        if metric_value is None:
            logger.warning(f"Metric '{condition.metric}' not available, defaulting to execute")
            return True, ""

        should_execute = condition.evaluate(metric_value)

        if should_execute:
            return True, ""
        else:
            return False, condition.on_false

    def _update_cumulative_metrics(
        self,
        metrics: Dict[str, Any],
        step_result: Dict[str, Any],
        baseline_accuracy: Optional[float],
    ) -> None:
        """Update cumulative metrics after a step execution.

        Args:
            metrics: Current cumulative metrics (modified in place)
            step_result: Result from the step execution
            baseline_accuracy: Baseline accuracy for computing deltas
        """
        # Update compression ratio
        if "compression_ratio" in step_result:
            metrics["compression_ratio"] *= step_result["compression_ratio"]

        # Update model size
        if "model_size_gb" in step_result:
            metrics["model_size_gb"] = step_result["model_size_gb"]

        # Update accuracy
        if "accuracy" in step_result:
            metrics["accuracy"] = step_result["accuracy"]
            if baseline_accuracy is not None:
                metrics["accuracy_delta"] = step_result["accuracy"] - baseline_accuracy
        elif "accuracy_delta" in step_result:
            # Some steps may report delta directly
            if metrics["accuracy"] is not None:
                metrics["accuracy"] += step_result["accuracy_delta"]
            metrics["accuracy_delta"] += step_result["accuracy_delta"]

    def _cleanup_checkpoint(self, checkpoint_path: str) -> None:
        """Delete an intermediate checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint to delete
        """
        try:
            path = Path(checkpoint_path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.debug(f"Deleted checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")

    def _archive_checkpoint(self, checkpoint_path: str, output_dir: str) -> None:
        """Archive an intermediate checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint to archive
            output_dir: Base output directory for archives
        """
        try:
            archive_dir = Path(output_dir) / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)

            src_path = Path(checkpoint_path)
            if src_path.exists():
                dst_path = archive_dir / src_path.name
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                logger.debug(f"Archived checkpoint: {checkpoint_path} -> {dst_path}")
        except Exception as e:
            logger.warning(f"Failed to archive checkpoint {checkpoint_path}: {e}")

    def _cleanup_gpu_between_steps(self) -> None:
        """Clean up GPU memory between pipeline steps.

        This helps prevent memory fragmentation and OOM errors when
        executing multi-step pipelines.
        """
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cleaned up between pipeline steps")
        except Exception as e:
            logger.debug(f"GPU cleanup failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation.

        Returns:
            Dictionary representation of the pipeline
        """
        return {
            "name": self.name,
            "description": self.description,
            "max_steps": self.MAX_STEPS,
            "steps": [
                {
                    "skill": step.skill_name,
                    "params": step.params,
                    "checkpoint_strategy": step.checkpoint_strategy,
                    "condition": step.condition.model_dump() if step.condition else None,
                    "name": step.name,
                }
                for step in self.steps
            ],
        }

    def __repr__(self) -> str:
        steps_str = " -> ".join(s.skill_name for s in self.steps)
        return f"CompressionPipeline(name='{self.name}', steps=[{steps_str}])"


class PipelineBuilder:
    """Fluent builder for creating compression pipelines."""

    def __init__(self, name: Optional[str] = None):
        """Initialize the pipeline builder.

        Args:
            name: Optional name for the pipeline
        """
        self.name = name
        self.description = ""
        self.steps: List[PipelineStep] = []

    def add_quantization(
        self,
        method: str = "gptq",
        bit_width: int = 4,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a quantization step.

        Args:
            method: Quantization method (gptq, autoround, int8, awq)
            bit_width: Quantization bit width
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        params = {"bit_width": bit_width, **kwargs}
        if method == "int8":
            params.pop("bit_width", None)  # INT8 doesn't need bit_width

        self.steps.append(PipelineStep(
            skill_name=f"quantization_{method}",
            params=params,
        ))
        return self

    def add_pruning(
        self,
        method: str = "magnitude",
        sparsity: float = 0.3,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a pruning step.

        Args:
            method: Pruning method (magnitude, structured)
            sparsity: Target sparsity ratio
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        self.steps.append(PipelineStep(
            skill_name=f"pruning_{method}",
            params={"sparsity": sparsity, **kwargs},
        ))
        return self

    def add_finetuning(
        self,
        method: str = "lora",
        lora_rank: int = 16,
        condition: Optional[StepCondition] = None,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a fine-tuning step.

        Args:
            method: Fine-tuning method (lora, qlora)
            lora_rank: LoRA rank
            condition: Optional condition for this step
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        self.steps.append(PipelineStep(
            skill_name=f"{method}_finetune",
            params={"lora_rank": lora_rank, **kwargs},
            condition=condition,
        ))
        return self

    def add_conditional_recovery(
        self,
        accuracy_threshold: float = -0.05,
        lora_rank: int = 16,
    ) -> "PipelineBuilder":
        """Add conditional LoRA recovery step.

        Only executes if accuracy drop exceeds threshold.

        Args:
            accuracy_threshold: Accuracy delta threshold (negative)
            lora_rank: LoRA rank for recovery

        Returns:
            Self for chaining
        """
        self.steps.append(PipelineStep(
            skill_name="lora_finetune",
            params={"lora_rank": lora_rank},
            condition=StepCondition(
                metric="accuracy_delta",
                operator="lt",
                threshold=accuracy_threshold,
                on_false="skip",
            ),
        ))
        return self

    def with_description(self, description: str) -> "PipelineBuilder":
        """Set pipeline description.

        Args:
            description: Pipeline description

        Returns:
            Self for chaining
        """
        self.description = description
        return self

    def build(self) -> CompressionPipeline:
        """Build the compression pipeline.

        Returns:
            Configured CompressionPipeline
        """
        return CompressionPipeline(
            steps=self.steps,
            name=self.name,
            description=self.description,
        )


__all__ = [
    "CompressionPipeline",
    "PipelineBuilder",
]
