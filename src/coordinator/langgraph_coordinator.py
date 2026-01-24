"""LangGraph 1.0-based multi-agent coordinator for compression optimization.

This module implements a fully autonomous LLM-driven multi-agent system
for model compression optimization using LangGraph's state machine.

Enhanced with:
- Dynamic Skill Discovery: Auto-discover available compression methods
- Skill Composition: Chain multiple techniques (e.g., Pruning → Quantization → LoRA)
- Skill Learning: Learn from history to make better recommendations
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

import torch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.coordinator.state import (
    CompressionState,
    create_initial_state,
    get_pareto_summary,
    get_recent_history,
    check_termination,
)
from src.coordinator.spec_inference import infer_spec
from src.coordinator.pareto import ParetoFrontier
from src.common.schemas import (
    ModelSpec,
    CompressionStrategy,
    EvaluationResult,
    CompressionMethod,
)

# Import tools from agents
from src.agents.quantization_agent import (
    quantize_model,
    estimate_quantization_vram,
    validate_quantization_compatibility,
    list_available_quantization_methods,
    apply_lora_finetuning,
    apply_qlora_finetuning,
    apply_asvd_compression,
)
from src.agents.evaluation_agent import (
    evaluate_model,
    run_proxy_evaluation,
    measure_inference_latency,
    measure_memory_usage,
    estimate_energy_consumption,
)
from src.agents.search_agent import (
    bayesian_optimization_search,
    evolutionary_search,
    multi_armed_bandit_search,
    population_based_training,
    check_early_stopping,
    compute_bandit_reward,
    update_bandit_rewards,
    analyze_search_history,
)
from src.agents.pruning_agent import (
    prune_model,
    estimate_pruning_speedup,
    validate_pruning_compatibility,
    list_available_pruning_methods,
)

# Import skills system
from src.skills import (
    SkillRegistry,
    get_available_skills,
    get_skills_prompt,
    get_global_memory,
    SkillContext,
    SkillResult,
    CompressionPipeline,
    get_template,
    get_template_names,
    generate_llm_template_prompt,
    get_templates_for_model_family,
    PIPELINE_TEMPLATES,
)


# Base system prompt (will be enhanced with dynamic skills)
COORDINATOR_SYSTEM_PROMPT_BASE = """You are an expert model compression optimization coordinator.
Your goal is to find Pareto-optimal compression strategies that balance:
- Accuracy (higher is better)
- Latency (lower is better)
- Memory usage (lower is better)
- Model size (lower is better)
- Carbon emissions / CO2 (lower is better)

Consider environmental impact when making decisions. Lower bit quantization
typically results in lower energy consumption and CO2 emissions.

You make autonomous decisions about which compression strategies to try next based on:
1. Current Pareto frontier analysis - identify gaps in the accuracy/latency/size/CO2 trade-off space
2. Historical results - learn from successful and failed strategies
3. Model characteristics - choose methods suitable for the model family
4. Skill recommendations - leverage learned knowledge from past experiments
5. Carbon efficiency - prefer strategies that reduce environmental impact

Available actions:
- "quantization": Apply a single quantization strategy
- "asvd": Apply ASVD (Activation-aware SVD) compression with rank_ratio control
- "lora": Apply LoRA fine-tuning (for accuracy recovery)
- "qlora": Apply QLoRA (4-bit base + LoRA, memory-efficient)
- "pruning": Apply pruning to remove weights
- "pipeline": Execute a multi-step compression pipeline (chains multiple techniques)
- "search": Use optimization algorithms to suggest better strategies
- "end": Terminate optimization (when converged or budget exhausted)

When choosing a single skill (quantization/lora/qlora/pruning), specify parameters as described below.

When choosing "pipeline", specify:
- pipeline_name: Name of a predefined pipeline template, OR
- pipeline_steps: Custom list of steps to execute

Compression strategy tips:
- Use pipelines for maximum compression (e.g., Pruning → Quantization → LoRA recovery)
- LoRA/QLoRA can recover accuracy after aggressive compression
- Pipelines can have conditional steps that only execute if certain conditions are met
- Learn from history: skills that worked well before are likely to work again

Always explain your reasoning before deciding."""


def generate_coordinator_prompt(model_family: Optional[str] = None) -> str:
    """Generate a dynamic coordinator system prompt with available skills.

    Args:
        model_family: Optional model family to customize recommendations

    Returns:
        Complete system prompt with skill descriptions
    """
    prompt_parts = [COORDINATOR_SYSTEM_PROMPT_BASE]

    # Add dynamic skill descriptions
    skills_prompt = get_skills_prompt()
    if skills_prompt:
        prompt_parts.append("\n\n" + skills_prompt)

    # Add pipeline templates
    templates_prompt = generate_llm_template_prompt()
    if templates_prompt:
        prompt_parts.append("\n\n" + templates_prompt)

    # Add model-specific recommendations
    if model_family:
        recommended_templates = get_templates_for_model_family(model_family)
        if recommended_templates:
            prompt_parts.append(
                f"\n\nRecommended pipelines for {model_family} models: "
                f"{', '.join(recommended_templates)}"
            )

    return "\n".join(prompt_parts)

QUANTIZATION_AGENT_PROMPT = """You are a quantization specialist agent.
Your job is to apply quantization to compress neural network models.

Available methods:
- autoround: Adaptive rounding, good for maintaining accuracy (supports 2,3,4,8 bits)
- gptq: Post-training quantization, fast and efficient (supports 2,3,4,8 bits)
- int8: 8-bit quantization via BitsAndBytes, simple and reliable
- awq: Activation-aware weight quantization, best accuracy preservation (supports 4 bits)
- asvd: Activation-aware SVD compression, smooth accuracy trade-off (rank_ratio 0.3-0.7)
- lora: LoRA fine-tuning - adds trainable adapters without modifying base weights
- qlora: QLoRA - loads base model in 4-bit, trains LoRA adapters (memory efficient)

Consider model size and VRAM constraints when selecting parameters."""

EVALUATION_AGENT_PROMPT = """You are an evaluation specialist agent.
Your job is to measure the performance of compressed models across multiple benchmarks.

You evaluate:
- Accuracy on benchmarks (GSM8K, CommonsenseQA, TruthfulQA, HumanEval, BIG-Bench Hard)
- Inference latency (ms)
- Memory usage (GB)
- Throughput (tokens/sec)

Use proxy evaluation for quick assessment, then full evaluation for promising candidates."""


class LangGraphCoordinator:
    """LangGraph-based multi-agent coordinator for compression optimization.

    This coordinator uses GPT-4o to make autonomous decisions about compression
    strategies and delegates execution to specialized worker agents.

    Enhanced with:
    - Dynamic skill discovery and registration
    - Skill composition via pipelines
    - Skill learning from historical results
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        max_episodes: int = 10,
        budget_hours: float = 2.0,
        experiment_name: Optional[str] = None,
        use_skill_memory: bool = True,
    ):
        """Initialize the LangGraph coordinator.

        Args:
            model_name: HuggingFace model name to compress
            dataset: Target benchmark dataset
            max_episodes: Maximum compression episodes
            budget_hours: Time budget in hours
            experiment_name: Name for this experiment
            use_skill_memory: Whether to use skill learning (default: True)
        """
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model_name = model_name
        self.dataset = dataset
        self.max_episodes = max_episodes
        self.budget_hours = budget_hours
        self.use_skill_memory = use_skill_memory

        # Infer model specification
        self.model_spec = infer_spec(model_name, dataset)

        # Setup experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = model_name.split("/")[-1][:20]
            experiment_name = f"{model_short}_{dataset}_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = Path(f"data/experiments/{experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Pareto frontier
        self.pareto_frontier = ParetoFrontier(
            storage_path=str(self.experiment_dir / "pareto_frontier.json")
        )

        # Initialize skill memory for learning
        self.skill_memory = get_global_memory() if use_skill_memory else None

        # Discover available skills
        self.available_skills = get_available_skills()
        print(f"[Skills] Discovered {len(self.available_skills)} available skills")

        # Generate dynamic system prompt
        model_family = self.model_spec.model_family if self.model_spec else None
        self.coordinator_prompt = generate_coordinator_prompt(model_family)

        # Initialize LLMs
        self.llm_coordinator = ChatOpenAI(
            model="gpt-5.2",
            temperature=0.7,
        )
        self.llm_worker = ChatOpenAI(
            model="gpt-5.2",
            temperature=0,
        )

        # Build the graph
        self.graph = self._build_graph()

        # Save model spec
        with open(self.experiment_dir / "model_spec.json", "w") as f:
            json.dump(self.model_spec.model_dump(), f, indent=2, default=str)

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state machine.

        Returns:
            Compiled graph ready for execution
        """
        graph = StateGraph(CompressionState)

        # Add nodes
        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("quantization", self._quantization_node)
        graph.add_node("pruning", self._pruning_node)
        graph.add_node("pipeline", self._pipeline_node)  # New: multi-step pipeline
        graph.add_node("evaluation", self._evaluation_node)
        graph.add_node("search", self._search_node)
        graph.add_node("update_state", self._update_state_node)

        # Add edges
        graph.add_edge(START, "coordinator")

        # Conditional routing from coordinator
        graph.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "quantization": "quantization",
                "pruning": "pruning",
                "pipeline": "pipeline",
                "search": "search",
                "end": END,
            }
        )

        # Fixed edges
        graph.add_edge("quantization", "evaluation")
        graph.add_edge("pruning", "evaluation")
        graph.add_edge("pipeline", "evaluation")  # Pipeline goes to evaluation after completion
        graph.add_edge("evaluation", "update_state")
        graph.add_edge("search", "coordinator")
        graph.add_edge("update_state", "coordinator")

        return graph.compile()

    def _route_from_coordinator(self, state: CompressionState) -> str:
        """Route to the next node based on coordinator decision.

        Args:
            state: Current state

        Returns:
            Next node name
        """
        if state.get("should_terminate"):
            return "end"

        action = state.get("next_action", "end")
        # Route lora/qlora/asvd to the quantization node (handles all compression methods)
        if action in ["quantization", "lora", "qlora", "asvd"]:
            return "quantization"
        if action in ["pruning", "search", "pipeline"]:
            return action
        return "end"

    def _coordinator_node(self, state: CompressionState) -> Dict[str, Any]:
        """Coordinator node: LLM decides the next action.

        Enhanced with:
        - Skill recommendations from learning history
        - Pipeline support for multi-step compression
        - Dynamic skill descriptions

        Args:
            state: Current compression state

        Returns:
            State updates with next_action and action_params
        """
        # Check termination conditions
        should_terminate, reason = check_termination(state)
        if should_terminate:
            return {
                "should_terminate": True,
                "termination_reason": reason,
                "next_action": "end",
            }

        # Build context for LLM
        pareto_summary = get_pareto_summary(state)
        recent_history = get_recent_history(state)

        # Get skill recommendations from memory
        recommendations_str = ""
        skill_recommendations = None
        if self.skill_memory:
            skill_context = self._build_skill_context(state)
            recommendations = self.skill_memory.query(skill_context, n_recommendations=3)
            if recommendations:
                skill_recommendations = [r.model_dump() for r in recommendations]
                rec_lines = ["Skill Recommendations (based on history):"]
                for rec in recommendations:
                    rec_lines.append(
                        f"  - {rec.skill_name}: {rec.reasoning} "
                        f"(confidence: {rec.confidence:.0%})"
                    )
                recommendations_str = "\n".join(rec_lines)

        # Get recommended pipeline templates
        model_family = state['model_spec'].get('model_family', 'unknown')
        recommended_pipelines = get_templates_for_model_family(model_family)
        pipelines_str = f"Recommended pipelines for {model_family}: {', '.join(recommended_pipelines[:3])}"

        # Extract CO2 metrics from Pareto frontier for context
        pareto_frontier = state.get("pareto_frontier", [])
        co2_summary = ""
        if pareto_frontier:
            co2_values = [s.get("result", {}).get("co2_grams") for s in pareto_frontier
                         if s.get("result", {}).get("co2_grams") is not None]
            if co2_values:
                best_co2 = min(co2_values)
                avg_co2 = sum(co2_values) / len(co2_values)
                co2_summary = f"\nCarbon Metrics:\n- Best CO2: {best_co2:.2f}g\n- Average CO2: {avg_co2:.2f}g"

        context = f"""
Current State:
- Model: {state['model_name']}
- Dataset: {state['dataset']}
- Episode: {state['current_episode'] + 1}/{state['max_episodes']}
- Elapsed: {self._get_elapsed_hours(state):.2f}h / {state['budget_hours']}h
- Consecutive no-improvement: {state['consecutive_no_improvement']}

{pareto_summary}
{co2_summary}

{recent_history}

Model Info:
- Size: {state['model_spec'].get('model_size_gb', 'unknown')} GB
- Family: {model_family}
- Preferred methods: {state['model_spec'].get('preferred_methods', [])}

{recommendations_str}

{pipelines_str}

What should be the next action? Choose one:
1. "quantization" with specific method and bits
2. "asvd" with rank_ratio (0.3=aggressive, 0.5=balanced, 0.7=conservative)
3. "lora" or "qlora" with lora_rank for fine-tuning
4. "pruning" with method, sparsity, and granularity
5. "pipeline" with pipeline_name for multi-step compression
6. "search" to explore new strategies
7. "end" if converged or should stop

Note: Consider CO2 emissions when choosing strategies. Lower bit quantization
typically reduces energy consumption and carbon footprint.

Respond in JSON format:
For quantization: {{"action": "quantization", "method": "autoround|gptq|int8|awq", "bits": 4, "reasoning": "..."}}
For ASVD: {{"action": "asvd", "method": "asvd", "rank_ratio": 0.5, "reasoning": "..."}}
For LoRA/QLoRA: {{"action": "lora|qlora", "method": "lora|qlora", "lora_rank": 16, "reasoning": "..."}}
For pruning: {{"action": "pruning", "method": "magnitude|structured", "sparsity": 0.3, "granularity": "weight|channel|head", "reasoning": "..."}}
For pipeline: {{"action": "pipeline", "pipeline_name": "aggressive_compression|accuracy_recovery|...", "reasoning": "..."}}
"""

        # Call LLM with dynamic prompt
        messages = [
            SystemMessage(content=self.coordinator_prompt),
            HumanMessage(content=context),
        ]

        response = self.llm_coordinator.invoke(messages)

        # Parse response
        try:
            # Extract JSON from response
            response_text = response.content
            # Try to find JSON in response using balanced brace matching
            decision = None
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(response_text[start_idx:], start=start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if brace_count == 0:
                    json_str = response_text[start_idx:end_idx]
                    decision = json.loads(json_str)
            if decision is None:
                # Default to quantization with gptq 4-bit
                decision = {"action": "quantization", "method": "gptq", "bits": 4}
        except json.JSONDecodeError:
            decision = {"action": "quantization", "method": "gptq", "bits": 4}

        action = decision.get("action", "end")
        reasoning = decision.get("reasoning", "")

        # Log decision
        print(f"\n[Coordinator] Episode {state['current_episode'] + 1}")
        print(f"  Decision: {action}")
        if reasoning:
            print(f"  Reasoning: {reasoning[:100]}...")

        # Build action params
        action_params = {}
        pipeline_name = None

        if action == "quantization":
            action_params = {
                "method": decision.get("method", "gptq"),
                "bits": decision.get("bits", 4),
            }
        elif action == "asvd":
            action_params = {
                "method": "asvd",
                "rank_ratio": decision.get("rank_ratio", 0.5),
            }
        elif action in ["lora", "qlora"]:
            action_params = {
                "method": action,
                "lora_rank": decision.get("lora_rank", 16),
            }
        elif action == "pruning":
            action_params = {
                "method": decision.get("method", "magnitude"),
                "sparsity": decision.get("sparsity", 0.3),
                "granularity": decision.get("granularity", "weight"),
            }
        elif action == "pipeline":
            pipeline_name = decision.get("pipeline_name", "balanced_quality")
            action_params = {
                "pipeline_name": pipeline_name,
            }
            print(f"  Pipeline: {pipeline_name}")

        return {
            "next_action": action,
            "action_params": action_params,
            "pipeline_name": pipeline_name,
            "skill_recommendations": skill_recommendations,
            "messages": [
                AIMessage(content=f"Decided: {action}. {reasoning}")
            ],
        }

    def _build_skill_context(self, state: CompressionState) -> SkillContext:
        """Build a SkillContext from the current state for querying skill memory.

        Args:
            state: Current compression state

        Returns:
            SkillContext for memory query
        """
        model_spec = state.get("model_spec", {})
        history = state.get("history", [])

        # Determine current Pareto gap
        pareto_gap = "balanced"
        pareto_frontier = state.get("pareto_frontier", [])
        if pareto_frontier:
            accuracies = [s.get("result", {}).get("accuracy", 0) for s in pareto_frontier]
            compressions = [s.get("result", {}).get("compression_ratio", 1) for s in pareto_frontier]
            max_acc = max(accuracies) if accuracies else 0
            max_comp = max(compressions) if compressions else 1

            if max_acc < 0.85:
                pareto_gap = "need_accuracy"
            elif max_comp < 2.0:
                pareto_gap = "need_compression"

        # Get previous skills from history
        previous_skills = []
        for entry in history[-5:]:  # Last 5 episodes
            strategy = entry.get("strategy", {})
            methods = strategy.get("methods", [])
            if methods:
                method = methods[0]
                if isinstance(method, dict):
                    method = method.get("value", method.get("name", "unknown"))
                previous_skills.append(str(method))

        return SkillContext(
            model_family=model_spec.get("model_family", "unknown"),
            model_size_gb=model_spec.get("model_size_gb", 1.0),
            target_objective=model_spec.get("primary_objective", "balanced"),
            hardware_profile="auto",
            previous_skills=previous_skills,
            current_pareto_gap=pareto_gap,
            baseline_accuracy=state.get("baseline_accuracy"),
            current_compression_ratio=state.get("compression_ratio", 1.0),
        )

    def _quantization_node(self, state: CompressionState) -> Dict[str, Any]:
        """Quantization node: Apply compression strategy (quantization or LoRA/QLoRA).

        Args:
            state: Current compression state

        Returns:
            State updates with current_strategy and compressed_model_path
        """
        params = state.get("action_params", {})
        method = params.get("method", "gptq")

        # Pre-episode resource check
        model_size = state.get("model_spec", {}).get("model_size_gb", 7.0)
        can_proceed, warning = self._pre_episode_resource_check(method, model_size)
        if not can_proceed:
            print(f"  [Resource] {warning}")
            return {
                "current_strategy": None,
                "compressed_model_path": None,
                "error": warning,
                "should_skip_evaluation": True,
            }

        # Handle ASVD compression
        if method == "asvd":
            rank_ratio = params.get("rank_ratio", 0.5)
            print(f"  Applying ASVD compression with rank_ratio={rank_ratio}...")

            # Create strategy
            strategy = CompressionStrategy(
                episode_id=state["current_episode"],
                strategy_id=f"strategy_{state['current_episode']:03d}",
                methods=[CompressionMethod.ASVD],
                quantization_method="asvd",
                asvd_rank_ratio=rank_ratio,
                calibration_dataset=state["dataset"],
            )

            # Save strategy
            episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            with open(episode_dir / "strategy.json", "w") as f:
                json.dump(strategy.model_dump(), f, indent=2, default=str)

            # Execute ASVD compression
            result = None
            try:
                result = apply_asvd_compression.invoke({
                    "model_path": state["model_name"],
                    "rank_ratio": rank_ratio,
                    "calibration_samples": state["model_spec"].get("calibration_samples", 512),
                    "calibration_dataset": state["dataset"],
                })
                compressed_path = result.get("checkpoint_path", f"data/checkpoints/{strategy.strategy_id}")
                print(f"  ASVD-compressed model saved to: {compressed_path}")
            except RuntimeError as e:
                from src.common.gpu_utils import handle_cuda_error, CUDAErrorRecovery
                is_recoverable, error_type = handle_cuda_error(e)
                if error_type == "oom":
                    print(f"  [CUDA] OOM detected, attempting recovery...")
                    CUDAErrorRecovery.attempt_oom_recovery()
                elif error_type == "illegal_memory_access":
                    print(f"  [CUDA] Fatal error detected, resetting state...")
                    CUDAErrorRecovery.reset_cuda_state()
                print(f"  ASVD compression failed: {e}")
                compressed_path = None
            except Exception as e:
                print(f"  ASVD compression failed: {e}")
                compressed_path = None

            # Extract compression metrics
            compression_ratio = result.get("compression_ratio", 1.0) if result else 1.0
            model_size_gb = result.get("model_size_gb", state["model_spec"].get("model_size_gb", 1.0)) if result else state["model_spec"].get("model_size_gb", 1.0)

            return {
                "current_strategy": strategy.model_dump(),
                "compressed_model_path": compressed_path,
                "compression_ratio": compression_ratio,
                "compressed_model_size_gb": model_size_gb,
            }

        # Handle LoRA/QLoRA fine-tuning
        if method in ["lora", "qlora"]:
            lora_rank = params.get("lora_rank", 16)
            print(f"  Applying {method.upper()} fine-tuning with rank={lora_rank}...")

            # Create strategy
            strategy = CompressionStrategy(
                episode_id=state["current_episode"],
                strategy_id=f"strategy_{state['current_episode']:03d}",
                methods=[CompressionMethod(method.lower())],
                lora_rank=lora_rank,
                calibration_dataset=state["dataset"],
            )

            # Save strategy
            episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            with open(episode_dir / "strategy.json", "w") as f:
                json.dump(strategy.model_dump(), f, indent=2, default=str)

            # Execute LoRA/QLoRA fine-tuning
            result = None
            try:
                if method == "lora":
                    result = apply_lora_finetuning.invoke({
                        "model_path": state["model_name"],
                        "lora_rank": lora_rank,
                        "calibration_dataset": state["dataset"],
                    })
                else:  # qlora
                    result = apply_qlora_finetuning.invoke({
                        "model_path": state["model_name"],
                        "lora_rank": lora_rank,
                        "bits": 4,
                        "calibration_dataset": state["dataset"],
                    })
                compressed_path = result.get("checkpoint_path", f"data/checkpoints/{strategy.strategy_id}")
                print(f"  Adapter saved to: {compressed_path}")
            except RuntimeError as e:
                from src.common.gpu_utils import handle_cuda_error, CUDAErrorRecovery
                is_recoverable, error_type = handle_cuda_error(e)
                if error_type == "oom":
                    print(f"  [CUDA] OOM detected, attempting recovery...")
                    CUDAErrorRecovery.attempt_oom_recovery()
                elif error_type == "illegal_memory_access":
                    print(f"  [CUDA] Fatal error detected, resetting state...")
                    CUDAErrorRecovery.reset_cuda_state()
                print(f"  {method.upper()} fine-tuning failed: {e}")
                compressed_path = None
            except Exception as e:
                print(f"  {method.upper()} fine-tuning failed: {e}")
                compressed_path = None

            # Extract compression metrics
            compression_ratio = result.get("compression_ratio", 1.0) if result else 1.0
            model_size_gb = result.get("model_size_gb", state["model_spec"].get("model_size_gb", 1.0)) if result else state["model_spec"].get("model_size_gb", 1.0)

        else:
            # Handle standard quantization methods
            bits = params.get("bits", 4)
            print(f"  Applying {method} {bits}-bit quantization...")

            # Create strategy
            strategy = CompressionStrategy(
                episode_id=state["current_episode"],
                strategy_id=f"strategy_{state['current_episode']:03d}",
                methods=[CompressionMethod(method)],
                quantization_method=method,
                quantization_bits=bits,
                calibration_dataset=state["dataset"],
            )

            # Save strategy
            episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            with open(episode_dir / "strategy.json", "w") as f:
                json.dump(strategy.model_dump(), f, indent=2, default=str)

            # Execute quantization
            result = None
            try:
                result = quantize_model.invoke({
                    "model_path": state["model_name"],
                    "method": method,
                    "bit_width": bits,
                    "calibration_samples": state["model_spec"].get("calibration_samples", 512),
                })
                compressed_path = result.get("checkpoint_path", f"data/checkpoints/{strategy.strategy_id}")
                print(f"  Compressed model saved to: {compressed_path}")
            except RuntimeError as e:
                from src.common.gpu_utils import handle_cuda_error, CUDAErrorRecovery
                is_recoverable, error_type = handle_cuda_error(e)
                if error_type == "oom":
                    print(f"  [CUDA] OOM detected, attempting recovery...")
                    CUDAErrorRecovery.attempt_oom_recovery()
                elif error_type == "illegal_memory_access":
                    print(f"  [CUDA] Fatal error detected, resetting state...")
                    CUDAErrorRecovery.reset_cuda_state()
                print(f"  Quantization failed: {e}")
                compressed_path = None
            except Exception as e:
                print(f"  Quantization failed: {e}")
                compressed_path = None

            # Extract compression metrics
            compression_ratio = result.get("compression_ratio", 1.0) if result else 1.0
            model_size_gb = result.get("model_size_gb", state["model_spec"].get("model_size_gb", 1.0)) if result else state["model_spec"].get("model_size_gb", 1.0)

        return {
            "current_strategy": strategy.model_dump(),
            "compressed_model_path": compressed_path,
            "compression_ratio": compression_ratio,
            "compressed_model_size_gb": model_size_gb,
        }

    def _pruning_node(self, state: CompressionState) -> Dict[str, Any]:
        """Pruning node: Apply pruning compression strategy.

        Args:
            state: Current compression state

        Returns:
            State updates with current_strategy and compressed_model_path
        """
        params = state.get("action_params", {})
        method = params.get("method", "magnitude")
        sparsity = params.get("sparsity", 0.3)
        granularity = params.get("granularity", "weight")

        # Pre-episode resource check
        model_size = state.get("model_spec", {}).get("model_size_gb", 7.0)
        can_proceed, warning = self._pre_episode_resource_check("pruning", model_size)
        if not can_proceed:
            print(f"  [Resource] {warning}")
            return {
                "current_strategy": None,
                "compressed_model_path": None,
                "error": warning,
                "should_skip_evaluation": True,
            }

        print(f"  Applying {method} pruning with {sparsity*100:.0f}% sparsity...")

        # Create strategy
        strategy = CompressionStrategy(
            episode_id=state["current_episode"],
            strategy_id=f"strategy_{state['current_episode']:03d}",
            methods=[CompressionMethod.PRUNING],
            pruning_ratio=sparsity,
            pruning_method=method,
            pruning_granularity=granularity,
        )

        # Save strategy
        episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "strategy.json", "w") as f:
            json.dump(strategy.model_dump(), f, indent=2, default=str)

        # Execute pruning
        result = None
        try:
            result = prune_model.invoke({
                "model_path": state["model_name"],
                "method": method,
                "sparsity": sparsity,
                "granularity": granularity,
            })
            compressed_path = result.get("checkpoint_path", f"data/checkpoints/{strategy.strategy_id}")
            print(f"  Pruned model saved to: {compressed_path}")
        except RuntimeError as e:
            from src.common.gpu_utils import handle_cuda_error, CUDAErrorRecovery
            is_recoverable, error_type = handle_cuda_error(e)
            if error_type == "oom":
                print(f"  [CUDA] OOM detected, attempting recovery...")
                CUDAErrorRecovery.attempt_oom_recovery()
            elif error_type == "illegal_memory_access":
                print(f"  [CUDA] Fatal error detected, resetting state...")
                CUDAErrorRecovery.reset_cuda_state()
            print(f"  Pruning failed: {e}")
            compressed_path = None
        except Exception as e:
            print(f"  Pruning failed: {e}")
            compressed_path = None

        # Extract compression metrics
        compression_ratio = result.get("compression_ratio", 1.0) if result else 1.0
        model_size_gb = result.get("model_size_gb", state["model_spec"].get("model_size_gb", 1.0)) if result else state["model_spec"].get("model_size_gb", 1.0)

        return {
            "current_strategy": strategy.model_dump(),
            "compressed_model_path": compressed_path,
            "compression_ratio": compression_ratio,
            "compressed_model_size_gb": model_size_gb,
        }

    def _pipeline_node(self, state: CompressionState) -> Dict[str, Any]:
        """Pipeline node: Execute a multi-step compression pipeline.

        Pipelines allow chaining multiple compression techniques with
        conditional execution based on intermediate results.

        Args:
            state: Current compression state

        Returns:
            State updates with current_strategy and compressed_model_path
        """
        params = state.get("action_params", {})
        pipeline_name = params.get("pipeline_name", "balanced_quality")

        # Pre-episode resource check for pipeline
        model_size = state.get("model_spec", {}).get("model_size_gb", 7.0)
        can_proceed, warning = self._pre_episode_resource_check("pipeline", model_size)
        if not can_proceed:
            print(f"  [Resource] {warning}")
            return {
                "current_strategy": None,
                "compressed_model_path": None,
                "error": warning,
                "should_skip_evaluation": True,
            }

        print(f"  Executing pipeline: {pipeline_name}")

        # Get the pipeline template
        pipeline = get_template(pipeline_name)
        if pipeline is None:
            print(f"  Warning: Pipeline '{pipeline_name}' not found, using balanced_quality")
            pipeline = get_template("balanced_quality")

        if pipeline is None:
            # Fallback to single quantization if templates fail
            print(f"  Error: Could not load any pipeline, falling back to GPTQ")
            return self._quantization_node({
                **state,
                "action_params": {"method": "gptq", "bits": 4}
            })

        # Validate pipeline
        is_valid, messages = pipeline.validate()
        for msg in messages:
            print(f"  {msg}")

        # Execute the pipeline
        pipeline_output_dir = (
            Path(state["experiment_dir"]) /
            f"episode_{state['current_episode']:03d}_pipeline"
        )

        try:
            result = pipeline.execute(
                model_path=state["model_name"],
                state=state,
                output_dir=str(pipeline_output_dir),
                baseline_accuracy=state.get("baseline_accuracy"),
            )

            print(f"  Pipeline completed: {result.steps_executed} steps executed, "
                  f"{result.steps_skipped} skipped")

            if result.success and result.final_checkpoint_path:
                compressed_path = result.final_checkpoint_path
                compression_ratio = result.final_compression_ratio or 1.0
                model_size_gb = result.final_model_size_gb or state["model_spec"].get("model_size_gb", 1.0)
            else:
                print(f"  Pipeline failed: {result.error_message}")
                compressed_path = None
                compression_ratio = 1.0
                model_size_gb = state["model_spec"].get("model_size_gb", 1.0)

        except RuntimeError as e:
            from src.common.gpu_utils import handle_cuda_error, CUDAErrorRecovery
            is_recoverable, error_type = handle_cuda_error(e)
            if error_type == "oom":
                print(f"  [CUDA] OOM detected, attempting recovery...")
                CUDAErrorRecovery.attempt_oom_recovery()
            elif error_type == "illegal_memory_access":
                print(f"  [CUDA] Fatal error detected, resetting state...")
                CUDAErrorRecovery.reset_cuda_state()
            print(f"  Pipeline execution error: {e}")
            compressed_path = None
            compression_ratio = 1.0
            model_size_gb = state["model_spec"].get("model_size_gb", 1.0)
            result = None
        except Exception as e:
            print(f"  Pipeline execution error: {e}")
            compressed_path = None
            compression_ratio = 1.0
            model_size_gb = state["model_spec"].get("model_size_gb", 1.0)
            result = None

        # Create strategy representing the pipeline
        # Use the pipeline name as the method
        strategy = CompressionStrategy(
            episode_id=state["current_episode"],
            strategy_id=f"strategy_{state['current_episode']:03d}",
            methods=[CompressionMethod.GPTQ],  # Placeholder - pipelines can have multiple
            quantization_method=f"pipeline:{pipeline_name}",
            calibration_dataset=state["dataset"],
        )

        # Save strategy
        episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "strategy.json", "w") as f:
            strategy_data = strategy.model_dump()
            strategy_data["pipeline_name"] = pipeline_name
            strategy_data["pipeline_result"] = result.model_dump() if result else None
            json.dump(strategy_data, f, indent=2, default=str)

        return {
            "current_strategy": strategy.model_dump(),
            "compressed_model_path": compressed_path,
            "compression_ratio": compression_ratio,
            "compressed_model_size_gb": model_size_gb,
            "current_pipeline": pipeline.to_dict().get("steps") if pipeline else None,
            "pipeline_name": pipeline_name,
        }

    def _evaluation_node(self, state: CompressionState) -> Dict[str, Any]:
        """Evaluation node: Measure compressed model performance.

        Args:
            state: Current compression state

        Returns:
            State updates with current_result
        """
        strategy = state.get("current_strategy", {})
        compressed_path = state.get("compressed_model_path")

        print(f"  Evaluating on {state['dataset']}...")

        # Run evaluation
        eval_result = None
        carbon_result = None

        try:
            eval_result = evaluate_model.invoke({
                "checkpoint_path": compressed_path or state["model_name"],
                "benchmarks": [state["dataset"]],
                "use_proxy": True,
                "proxy_samples": 200,
            })
        except Exception as e:
            print(f"  Benchmark evaluation failed: {e}")

        # Measure carbon emissions (500 inferences for balance between speed and accuracy)
        print(f"  Measuring carbon emissions (500 inferences)...")
        try:
            carbon_result = estimate_energy_consumption.invoke({
                "checkpoint_path": compressed_path or state["model_name"],
                "num_inferences": 500,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            })
            print(f"  Carbon: {carbon_result.get('co2_grams', 0):.2f}g CO2, "
                  f"{carbon_result.get('energy_per_inference_joules', 0):.4f}J/inference")
        except Exception as e:
            print(f"  Carbon measurement failed: {e}")
            carbon_result = None

        # Build EvaluationResult with carbon metrics
        if eval_result:
            result = EvaluationResult(
                strategy_id=strategy.get("strategy_id", "unknown"),
                checkpoint_path=compressed_path or state["model_name"],
                model_size_gb=state.get("compressed_model_size_gb") or state["model_spec"].get("model_size_gb", 1.0),
                compression_ratio=state.get("compression_ratio") or 1.0,
                accuracy=eval_result.get("average_accuracy", 0.0),
                latency_ms=eval_result.get("latency_ms", 100.0),
                throughput_tokens_per_sec=eval_result.get("throughput", 100.0),
                memory_gb=eval_result.get("memory_gb", 1.0),
                benchmark_scores=eval_result.get("benchmark_scores", {}),
                evaluation_time_sec=eval_result.get("evaluation_time", 60.0),
                # Carbon metrics
                energy_joules=carbon_result.get("energy_per_inference_joules") if carbon_result else None,
                energy_kwh=carbon_result.get("energy_kwh") if carbon_result else None,
                co2_grams=carbon_result.get("co2_grams") if carbon_result else None,
                co2_kg=carbon_result.get("co2_kg") if carbon_result else None,
                num_inferences_measured=500 if carbon_result else None,
                is_carbon_mock=carbon_result.get("is_mock", True) if carbon_result else True,
            )
        else:
            # Create minimal result with carbon if available
            result = EvaluationResult(
                strategy_id=strategy.get("strategy_id", "unknown"),
                checkpoint_path=compressed_path or state["model_name"],
                model_size_gb=state.get("compressed_model_size_gb") or state["model_spec"].get("model_size_gb", 1.0),
                compression_ratio=state.get("compression_ratio") or 1.0,
                accuracy=0.0,
                latency_ms=1000.0,
                throughput_tokens_per_sec=10.0,
                memory_gb=10.0,
                benchmark_scores={},
                evaluation_time_sec=0.0,
                # Carbon metrics
                energy_joules=carbon_result.get("energy_per_inference_joules") if carbon_result else None,
                energy_kwh=carbon_result.get("energy_kwh") if carbon_result else None,
                co2_grams=carbon_result.get("co2_grams") if carbon_result else None,
                co2_kg=carbon_result.get("co2_kg") if carbon_result else None,
                num_inferences_measured=500 if carbon_result else None,
                is_carbon_mock=carbon_result.get("is_mock", True) if carbon_result else True,
            )

        co2_str = f", CO2={result.co2_grams:.2f}g" if result.co2_grams else ""
        print(f"  Result: acc={result.accuracy:.3f}, latency={result.latency_ms:.1f}ms{co2_str}")

        return {
            "current_result": result.model_dump(),
        }

    def _search_node(self, state: CompressionState) -> Dict[str, Any]:
        """Search node: Use optimization to suggest strategies.

        Uses multi-armed bandit with real rewards from evaluation history.

        Args:
            state: Current compression state

        Returns:
            State updates with suggested action_params
        """
        print("  Running search optimization with real rewards...")

        history = state.get("history", [])
        available_methods = ["autoround", "gptq", "int8", "awq"]

        # Build rewards history from evaluation results
        rewards_history: Dict[str, List[float]] = state.get("bandit_rewards", {})

        # Update rewards from history if not already tracked
        if not rewards_history and history:
            for entry in history:
                strategy = entry.get("strategy", {})
                result = entry.get("result", {})

                # Extract method from strategy
                quant_method = strategy.get("quantization_method")
                if quant_method and quant_method in available_methods:
                    # Compute reward from result
                    reward = compute_bandit_reward(result)
                    rewards_history = update_bandit_rewards(
                        rewards_history, quant_method, result
                    )

        try:
            # Use multi-armed bandit for method selection with real rewards
            suggestion = multi_armed_bandit_search.invoke({
                "arms": available_methods,
                "rewards_history": rewards_history,
                "exploration_strategy": "ucb",
                "exploration_param": 2.0,
            })

            selected_arm = suggestion.get("selected_arm", "gptq")
            arm_stats = suggestion.get("arm_statistics", {})

            # Log arm statistics
            if arm_stats:
                print(f"  Arm statistics:")
                for arm, stats in arm_stats.items():
                    if stats["count"] > 0:
                        print(f"    {arm}: mean={stats['mean']:.3f}, count={stats['count']}")

            method = selected_arm
            # Select bits based on method performance and model size
            model_size = state["model_spec"].get("model_size_gb", 1.0)
            if model_size > 30:
                bits = 4  # Larger models benefit more from 4-bit
            elif method == "int8":
                bits = 8
            else:
                bits = 4  # Default to 4-bit for most methods

        except Exception as e:
            print(f"  Search failed: {e}")
            import traceback
            traceback.print_exc()
            method = "gptq"
            bits = 4

        print(f"  Search suggests: {method} {bits}-bit")

        return {
            "next_action": "quantization",
            "action_params": {"method": method, "bits": bits},
            "bandit_rewards": rewards_history,
        }

    def _cleanup_gpu_memory(self, force: bool = False) -> Dict[str, float]:
        """Force GPU memory cleanup between episodes.

        Uses robust cleanup with synchronization to prevent memory
        fragmentation and illegal memory access errors.

        Args:
            force: If True, perform more aggressive cleanup

        Returns:
            Dictionary with cleanup statistics
        """
        from src.common.gpu_utils import robust_gpu_cleanup
        result = robust_gpu_cleanup(force_gc=True, reset_peak_stats=True)
        if result.get("freed_gb", 0) > 0.1:
            print(f"  [GPU] Freed {result['freed_gb']:.2f} GB memory")
        return result

    def _pre_episode_resource_check(
        self,
        method: str,
        model_size_gb: float
    ) -> tuple[bool, str]:
        """Check if sufficient resources are available before starting episode.

        Args:
            method: The compression method to be used
            model_size_gb: Size of the model in GB

        Returns:
            Tuple of (can_proceed, warning_message)
        """
        from src.common.gpu_utils import check_gpu_memory_available, robust_gpu_cleanup, estimate_operation_vram

        # Estimate required VRAM based on operation type
        if method in ["lora", "qlora"]:
            required_gb = estimate_operation_vram(model_size_gb, method)
        elif method in ["autoround", "gptq", "awq", "int8", "asvd"]:
            required_gb = estimate_operation_vram(model_size_gb, "quantization")
        else:
            required_gb = model_size_gb * 2.5

        is_available, available_gb = check_gpu_memory_available(required_gb)

        if not is_available:
            # Attempt cleanup and recheck
            print(f"  [Resource] Low VRAM ({available_gb:.1f} GB), attempting cleanup...")
            robust_gpu_cleanup(force_gc=True)
            is_available, available_gb = check_gpu_memory_available(required_gb)

            if not is_available:
                return False, f"Insufficient VRAM: need {required_gb:.1f}GB, have {available_gb:.1f}GB"

        return True, ""

    def _update_state_node(self, state: CompressionState) -> Dict[str, Any]:
        """Update state after evaluation.

        Enhanced to record skill results to memory for learning.

        Args:
            state: Current compression state

        Returns:
            State updates with history, pareto_frontier, episode counter
        """
        # Clean up GPU memory before processing to prevent OOM between episodes
        self._cleanup_gpu_memory()

        strategy = state.get("current_strategy", {})
        result = state.get("current_result", {})

        # Convert back to Pydantic models for Pareto update
        strategy_obj = CompressionStrategy(**strategy)
        result_obj = EvaluationResult(**result)

        # Update Pareto frontier
        is_pareto, dominated = self.pareto_frontier.add_solution(strategy_obj, result_obj)

        if is_pareto:
            print(f"  Added to Pareto frontier (dominates {len(dominated)} solutions)")
            consecutive_no_improvement = 0
        else:
            print(f"  Not Pareto optimal")
            consecutive_no_improvement = state.get("consecutive_no_improvement", 0) + 1

        # Record to skill memory for learning
        if self.skill_memory:
            self._record_skill_result(state, strategy, result, is_pareto)

        # Save episode results
        episode_dir = Path(state["experiment_dir"]) / f"episode_{state['current_episode']:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(episode_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Update history
        new_history_entry = {
            "strategy": strategy,
            "result": result,
        }

        # Get updated Pareto frontier as list of dicts
        pareto_solutions = [
            {"strategy": sol.strategy.model_dump(), "result": sol.result.model_dump()}
            for sol in self.pareto_frontier.get_frontier()
        ]

        # Update bandit rewards with this episode's result
        bandit_rewards = state.get("bandit_rewards", {})
        quant_method = strategy.get("quantization_method")
        if quant_method:
            # Extract base method name (strip any pipeline: prefix)
            if quant_method.startswith("pipeline:"):
                method_name = None  # Skip pipeline methods for MAB
            else:
                method_name = quant_method.lower()

            if method_name and method_name in ["autoround", "gptq", "int8", "awq"]:
                bandit_rewards = update_bandit_rewards(
                    bandit_rewards, method_name, result
                )
                print(f"  [MAB] Updated rewards for {method_name}")

        return {
            "history": [new_history_entry],
            "pareto_frontier": pareto_solutions,
            "current_episode": state["current_episode"] + 1,
            "consecutive_no_improvement": consecutive_no_improvement,
            "bandit_rewards": bandit_rewards,
            "current_strategy": None,
            "current_result": None,
            "compressed_model_path": None,
            "compression_ratio": None,
            "compressed_model_size_gb": None,
            "current_pipeline": None,
            "pipeline_name": None,
        }

    def _record_skill_result(
        self,
        state: CompressionState,
        strategy: Dict[str, Any],
        result: Dict[str, Any],
        is_pareto: bool,
    ) -> None:
        """Record a skill execution result to memory for learning.

        Args:
            state: Current compression state
            strategy: Strategy that was executed
            result: Evaluation result
            is_pareto: Whether the result is Pareto-optimal
        """
        try:
            # Build skill context
            skill_context = self._build_skill_context(state)

            # Determine skill name from strategy
            methods = strategy.get("methods", [])
            if methods:
                method = methods[0]
                if isinstance(method, dict):
                    method = method.get("value", method.get("name", "unknown"))
                else:
                    method = str(method)
            else:
                method = "unknown"

            # Map to skill name
            quant_method = strategy.get("quantization_method", "")
            if quant_method and quant_method.startswith("pipeline:"):
                skill_name = f"pipeline_{quant_method.split(':')[1]}"
            elif method.lower() in ["gptq", "autoround", "int8", "awq"]:
                skill_name = f"quantization_{method.lower()}"
            elif method.lower() == "pruning":
                pruning_method = strategy.get("pruning_method", "magnitude")
                skill_name = f"pruning_{pruning_method}"
            elif method.lower() in ["lora", "qlora"]:
                skill_name = f"{method.lower()}_finetune"
            else:
                skill_name = f"unknown_{method.lower()}"

            # Extract params
            params = {}
            if strategy.get("quantization_bits"):
                params["bit_width"] = strategy["quantization_bits"]
            if strategy.get("pruning_ratio"):
                params["sparsity"] = strategy["pruning_ratio"]
            if strategy.get("lora_rank"):
                params["lora_rank"] = strategy["lora_rank"]

            # Calculate accuracy delta
            baseline_acc = state.get("baseline_accuracy", 1.0) or 1.0
            current_acc = result.get("accuracy", 0.0)
            accuracy_delta = current_acc - baseline_acc

            # Build skill result
            skill_result = SkillResult(
                skill_name=skill_name,
                params=params,
                accuracy_delta=accuracy_delta,
                compression_ratio=result.get("compression_ratio", 1.0),
                is_pareto_optimal=is_pareto,
                execution_time_sec=result.get("evaluation_time_sec", 0.0),
                success=True,
            )

            # Record to memory
            self.skill_memory.record(
                context=skill_context,
                result=skill_result,
                experiment_id=self.experiment_name,
            )

            print(f"  [Memory] Recorded {skill_name} result to skill memory")

        except Exception as e:
            print(f"  [Memory] Failed to record skill result: {e}")

    def _get_elapsed_hours(self, state: CompressionState) -> float:
        """Calculate elapsed time in hours."""
        start = datetime.fromisoformat(state["start_time"])
        return (datetime.now() - start).total_seconds() / 3600

    def run(self, interactive: bool = False) -> Dict[str, Any]:
        """Run the compression optimization.

        Args:
            interactive: Whether to print progress updates

        Returns:
            Final results dictionary
        """
        print("=" * 60)
        print("LangGraph Compression Coordinator")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Max episodes: {self.max_episodes}")
        print(f"Budget: {self.budget_hours}h")
        print(f"Experiment: {self.experiment_name}")
        print("=" * 60)

        # Create initial state
        initial_state = create_initial_state(
            model_name=self.model_name,
            dataset=self.dataset,
            model_spec=self.model_spec,
            max_episodes=self.max_episodes,
            budget_hours=self.budget_hours,
            experiment_dir=str(self.experiment_dir),
        )

        # Run the graph with increased recursion limit for complex pipelines
        try:
            if interactive:
                # Stream for visibility
                for step in self.graph.stream(initial_state, {"recursion_limit": 100}):
                    node_name = list(step.keys())[0]
                    if node_name == "__end__":
                        break
            else:
                final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})
        except Exception as e:
            print(f"\nError during optimization: {e}")
            import traceback
            traceback.print_exc()

        # Compile final results
        return self._compile_results()

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results from the optimization run.

        Returns:
            Results dictionary
        """
        frontier_summary = self.pareto_frontier.get_summary()

        # Get best solutions (including carbon)
        best_solutions = {
            "accuracy": None,
            "latency": None,
            "size": None,
            "carbon": None,
            "balanced": None,
        }

        best_acc = self.pareto_frontier.get_best_by_objective("accuracy")
        if best_acc:
            best_solutions["accuracy"] = best_acc.result.model_dump()

        best_lat = self.pareto_frontier.get_best_by_objective("latency")
        if best_lat:
            best_solutions["latency"] = best_lat.result.model_dump()

        best_size = self.pareto_frontier.get_best_by_objective("size")
        if best_size:
            best_solutions["size"] = best_size.result.model_dump()

        best_carbon = self.pareto_frontier.get_best_by_objective("carbon")
        if best_carbon:
            best_solutions["carbon"] = best_carbon.result.model_dump()

        balanced = self.pareto_frontier.get_balanced_solution()
        if balanced:
            best_solutions["balanced"] = balanced.result.model_dump()

        # Create visualization
        viz_path = self.pareto_frontier.visualize_frontier(
            str(self.experiment_dir / "pareto_visualization.html")
        )

        results = {
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.experiment_dir),
            "model": self.model_name,
            "dataset": self.dataset,
            "episodes_completed": frontier_summary["num_evaluated"],
            "frontier_summary": frontier_summary,
            "best_solutions": best_solutions,
            "visualization": viz_path,
        }

        # Save results
        with open(self.experiment_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results


# CLI-compatible function
def create_coordinator(
    model_name: str,
    dataset: str,
    max_episodes: int = 10,
    budget_hours: float = 2.0,
    experiment_name: Optional[str] = None,
) -> LangGraphCoordinator:
    """Create a LangGraph coordinator instance.

    Args:
        model_name: HuggingFace model name
        dataset: Target benchmark dataset
        max_episodes: Maximum compression episodes
        budget_hours: Time budget in hours
        experiment_name: Name for this experiment

    Returns:
        Configured LangGraphCoordinator
    """
    return LangGraphCoordinator(
        model_name=model_name,
        dataset=dataset,
        max_episodes=max_episodes,
        budget_hours=budget_hours,
        experiment_name=experiment_name,
    )
