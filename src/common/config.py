"""Unified configuration system for the Agentic Compression Framework.

This module provides a centralized configuration system supporting:
- YAML/JSON configuration files
- Environment variable overrides (prefix: GREENAI_)
- CLI parameter overrides (highest priority)
- Sensible defaults

Priority order (highest to lowest):
1. CLI arguments
2. Environment variables
3. Configuration file
4. Default values
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class CoordinatorConfig(BaseModel):
    """Configuration for the LLM coordinator."""

    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model for coordinator decisions (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM decision-making (0=deterministic, 2=creative)"
    )
    worker_model: str = Field(
        default="gpt-4o",
        description="LLM model for worker agents (typically same as coordinator)"
    )
    worker_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for worker agents (0 recommended for consistency)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retries for LLM API calls"
    )


class QuantizationConfig(BaseModel):
    """Configuration for quantization operations."""

    default_bit_width: int = Field(
        default=4,
        description="Default bit width for quantization (2, 3, 4, or 8)"
    )
    default_method: Literal["autoround", "gptq", "awq", "int8"] = Field(
        default="gptq",
        description="Default quantization method"
    )
    group_size: int = Field(
        default=128,
        description="Group size for GPTQ/AutoRound quantization"
    )
    calibration_samples: int = Field(
        default=512,
        ge=32,
        description="Number of calibration samples for quantization"
    )
    calibration_seq_length: int = Field(
        default=2048,
        description="Sequence length for calibration data"
    )
    sym: bool = Field(
        default=False,
        description="Use symmetric quantization"
    )

    @field_validator("default_bit_width")
    @classmethod
    def validate_bit_width(cls, v: int) -> int:
        if v not in [2, 3, 4, 8]:
            raise ValueError(f"bit_width must be 2, 3, 4, or 8, got {v}")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    use_proxy: bool = Field(
        default=True,
        description="Use proxy (subset) evaluation for faster iteration"
    )
    proxy_samples: int = Field(
        default=200,
        ge=10,
        description="Number of samples for proxy evaluation"
    )
    full_eval_samples: Optional[int] = Field(
        default=None,
        description="Number of samples for full evaluation (None = use full dataset)"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for evaluation"
    )
    measure_carbon: bool = Field(
        default=True,
        description="Measure carbon emissions during evaluation"
    )
    carbon_inference_count: int = Field(
        default=500,
        ge=10,
        description="Number of inferences for carbon measurement"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for evaluation (None = auto-detect cuda/cpu)"
    )


class SearchConfig(BaseModel):
    """Configuration for strategy search algorithms."""

    method: Literal["random", "bayesian", "evolutionary", "bandit"] = Field(
        default="bandit",
        description="Search algorithm for strategy exploration"
    )
    exploration_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Exploration vs exploitation ratio"
    )
    ucb_exploration_param: float = Field(
        default=2.0,
        ge=0.0,
        description="UCB exploration parameter for bandit search"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for evolutionary search"
    )
    population_size: int = Field(
        default=10,
        ge=2,
        description="Population size for evolutionary search"
    )


class RewardConfig(BaseModel):
    """Configuration for the multi-objective reward function."""

    accuracy_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for accuracy in reward calculation"
    )
    latency_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for latency (lower is better)"
    )
    memory_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for memory usage (lower is better)"
    )
    energy_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for energy/carbon (lower is better)"
    )
    min_accuracy: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable accuracy threshold"
    )
    max_latency_ms: Optional[float] = Field(
        default=None,
        description="Maximum acceptable latency in ms (None = no limit)"
    )
    max_memory_gb: Optional[float] = Field(
        default=None,
        description="Maximum acceptable memory in GB (None = no limit)"
    )


class FinetuningConfig(BaseModel):
    """Configuration for LoRA/QLoRA fine-tuning."""

    lora_rank: int = Field(
        default=16,
        ge=1,
        description="LoRA adapter rank"
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA alpha parameter"
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="LoRA dropout rate"
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        description="Learning rate for fine-tuning"
    )
    num_train_steps: int = Field(
        default=100,
        ge=1,
        description="Number of training steps"
    )
    warmup_steps: int = Field(
        default=10,
        ge=0,
        description="Number of warmup steps"
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Gradient accumulation steps"
    )
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Target modules for LoRA (None = auto-detect)"
    )


class TerminationConfig(BaseModel):
    """Configuration for optimization termination conditions."""

    max_episodes: int = Field(
        default=10,
        ge=1,
        description="Maximum number of compression episodes"
    )
    budget_hours: float = Field(
        default=2.0,
        gt=0.0,
        description="Time budget in hours"
    )
    convergence_patience: int = Field(
        default=5,
        ge=1,
        description="Episodes without Pareto improvement before stopping"
    )
    min_improvement: float = Field(
        default=0.001,
        ge=0.0,
        description="Minimum improvement to count as progress"
    )


class AdvancedConfig(BaseModel):
    """Unified configuration for the Agentic Compression Framework.

    This is the top-level configuration object that contains all subsystem configs.

    Example usage:
        # From YAML file
        config = AdvancedConfig.from_yaml("config/default.yaml")

        # With CLI overrides
        config = AdvancedConfig.from_yaml("config.yaml")
        config.coordinator.llm_model = "gpt-4-turbo"  # CLI override

        # From environment variables
        export GREENAI_COORDINATOR__LLM_MODEL=gpt-4-turbo
        export GREENAI_QUANTIZATION__DEFAULT_BIT_WIDTH=8
    """

    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    finetuning: FinetuningConfig = Field(default_factory=FinetuningConfig)
    termination: TerminationConfig = Field(default_factory=TerminationConfig)

    # Preset name if loaded from preset
    _preset_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AdvancedConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AdvancedConfig instance
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AdvancedConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            AdvancedConfig instance
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedConfig":
        """Load configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            AdvancedConfig instance
        """
        # Handle preset loading
        preset_name = data.pop("preset", None)
        if preset_name:
            config = cls.from_preset(preset_name)
            # Override with any additional settings
            config = config.merge(data)
            config._preset_name = preset_name
            return config

        return cls(**data)

    @classmethod
    def from_preset(cls, preset_name: str) -> "AdvancedConfig":
        """Load a predefined configuration preset.

        Available presets:
        - accuracy_focused: Prioritize accuracy over compression
        - latency_focused: Prioritize low latency
        - balanced: Balance between accuracy, latency, and size
        - memory_constrained: Minimize memory usage

        Args:
            preset_name: Name of the preset to load

        Returns:
            AdvancedConfig instance
        """
        presets = {
            "accuracy_focused": {
                "quantization": {
                    "default_bit_width": 8,
                    "default_method": "int8",
                },
                "reward": {
                    "accuracy_weight": 2.0,
                    "latency_weight": 0.1,
                    "memory_weight": 0.1,
                    "min_accuracy": 0.95,
                },
                "finetuning": {
                    "lora_rank": 32,
                    "num_train_steps": 200,
                },
            },
            "latency_focused": {
                "quantization": {
                    "default_bit_width": 4,
                    "default_method": "gptq",
                },
                "reward": {
                    "accuracy_weight": 0.5,
                    "latency_weight": 1.5,
                    "memory_weight": 0.5,
                    "min_accuracy": 0.85,
                },
                "evaluation": {
                    "proxy_samples": 100,  # Faster iteration
                },
            },
            "balanced": {
                "quantization": {
                    "default_bit_width": 4,
                    "default_method": "gptq",
                },
                "reward": {
                    "accuracy_weight": 1.0,
                    "latency_weight": 0.3,
                    "memory_weight": 0.3,
                    "min_accuracy": 0.9,
                },
            },
            "memory_constrained": {
                "quantization": {
                    "default_bit_width": 4,
                    "default_method": "awq",
                },
                "reward": {
                    "accuracy_weight": 0.5,
                    "latency_weight": 0.3,
                    "memory_weight": 1.5,
                    "min_accuracy": 0.85,
                    "max_memory_gb": 8.0,
                },
                "finetuning": {
                    "lora_rank": 8,  # Lower rank = less memory
                    "gradient_accumulation_steps": 8,
                },
            },
            "low_carbon": {
                "quantization": {
                    "default_bit_width": 4,
                    "default_method": "awq",
                },
                "evaluation": {
                    "measure_carbon": True,
                    "carbon_inference_count": 1000,
                },
                "reward": {
                    "accuracy_weight": 0.5,
                    "latency_weight": 0.2,
                    "memory_weight": 0.2,
                    "energy_weight": 2.0,
                    "min_accuracy": 0.85,
                },
                "termination": {
                    "convergence_patience": 7,
                },
            },
            "low_cost": {
                "quantization": {
                    "default_bit_width": 4,
                    "default_method": "gptq",
                },
                "reward": {
                    "accuracy_weight": 0.5,
                    "latency_weight": 1.0,
                    "memory_weight": 1.5,
                    "energy_weight": 0.5,
                    "min_accuracy": 0.85,
                    "max_memory_gb": 8.0,
                },
                "evaluation": {
                    "use_proxy": True,
                    "proxy_samples": 100,
                },
                "finetuning": {
                    "lora_rank": 8,
                    "num_train_steps": 50,
                    "gradient_accumulation_steps": 8,
                },
            },
        }

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        config = cls.from_dict(presets[preset_name])
        config._preset_name = preset_name
        return config

    @classmethod
    def from_env(cls, prefix: str = "GREENAI") -> "AdvancedConfig":
        """Load configuration from environment variables.

        Environment variable format: {PREFIX}_{SECTION}__{FIELD}
        Example: GREENAI_COORDINATOR__LLM_MODEL=gpt-4-turbo

        Note: Uses double underscore (__) to separate section from field.

        Args:
            prefix: Environment variable prefix

        Returns:
            AdvancedConfig with environment overrides
        """
        config = cls()

        for key, value in os.environ.items():
            if not key.startswith(f"{prefix}_"):
                continue

            # Parse key: PREFIX_SECTION__FIELD
            parts = key[len(prefix) + 1:].lower().split("__")
            if len(parts) != 2:
                continue

            section, field = parts

            # Get the section config
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, field):
                    # Convert value to appropriate type
                    field_info = section_config.model_fields.get(field)
                    if field_info:
                        try:
                            # Handle type conversion
                            annotation = field_info.annotation
                            if annotation == bool:
                                value = value.lower() in ("true", "1", "yes")
                            elif annotation == int:
                                value = int(value)
                            elif annotation == float:
                                value = float(value)
                            # str stays as is

                            setattr(section_config, field, value)
                        except (ValueError, TypeError) as e:
                            warnings.warn(f"Failed to parse env var {key}: {e}")

        return config

    def merge(self, overrides: Dict[str, Any]) -> "AdvancedConfig":
        """Merge additional configuration overrides.

        Args:
            overrides: Dictionary of overrides to apply

        Returns:
            New AdvancedConfig with merged values
        """
        # Get current config as dict
        current = self.model_dump()

        # Deep merge
        def deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(current, overrides)
        return AdvancedConfig(**merged)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the configuration
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the configuration
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


def load_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    apply_env: bool = True,
) -> AdvancedConfig:
    """Load configuration with priority: CLI > Env > File > Preset > Defaults.

    This is the main entry point for loading configuration in the CLI.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        preset: Name of preset to use as base
        cli_overrides: Dictionary of CLI argument overrides
        apply_env: Whether to apply environment variable overrides

    Returns:
        Merged AdvancedConfig

    Example:
        config = load_config(
            config_path="config/my_config.yaml",
            preset="balanced",
            cli_overrides={
                "coordinator": {"llm_model": "gpt-4-turbo"},
                "quantization": {"default_bit_width": 8},
            }
        )
    """
    # Start with defaults or preset
    if preset:
        config = AdvancedConfig.from_preset(preset)
    else:
        config = AdvancedConfig()

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.suffix in [".yaml", ".yml"]:
            file_config = AdvancedConfig.from_yaml(path)
        elif path.suffix == ".json":
            file_config = AdvancedConfig.from_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        # Merge file config
        config = config.merge(file_config.model_dump())

    # Apply environment variables
    if apply_env:
        env_config = AdvancedConfig.from_env()
        config = config.merge(env_config.model_dump())

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        config = config.merge(cli_overrides)

    return config


# Convenience function for backward compatibility
def get_default_config() -> AdvancedConfig:
    """Get the default configuration.

    Returns:
        AdvancedConfig with all defaults
    """
    return AdvancedConfig()
