"""LangGraph 1.0-based multi-agent coordinator for compression optimization.

This module implements a fully autonomous LLM-driven multi-agent system
for model compression optimization using LangGraph's state machine.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Literal

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
)
from src.agents.evaluation_agent import (
    evaluate_model,
    run_proxy_evaluation,
    measure_inference_latency,
    measure_memory_usage,
)
from src.agents.search_agent import (
    bayesian_optimization_search,
    evolutionary_search,
    multi_armed_bandit_search,
    analyze_search_history,
)
from src.agents.pruning_agent import (
    prune_model,
    estimate_pruning_speedup,
    validate_pruning_compatibility,
    list_available_pruning_methods,
)


# System prompts
COORDINATOR_SYSTEM_PROMPT = """You are an expert model compression optimization coordinator.
Your goal is to find Pareto-optimal compression strategies that balance accuracy, latency, memory, and model size.

You make autonomous decisions about which compression strategies to try next based on:
1. Current Pareto frontier analysis - identify gaps in the accuracy/latency/size trade-off space
2. Historical results - learn from successful and failed strategies
3. Model characteristics - choose methods suitable for the model family

Available actions:
- "quantization": Apply a quantization strategy (AutoRound, GPTQ, INT8, AWQ with various bit-widths)
- "lora": Apply LoRA fine-tuning (for accuracy recovery after aggressive compression)
- "qlora": Apply QLoRA (4-bit base + LoRA, memory-efficient training for large models)
- "pruning": Apply pruning to remove weights (magnitude or structured pruning)
- "search": Use optimization algorithms to suggest better strategies
- "end": Terminate optimization (when converged or budget exhausted)

When choosing quantization, specify:
- method: One of "autoround", "gptq", "int8", "awq"
- bits: 2, 3, 4, or 8

When choosing lora/qlora, specify:
- method: "lora" or "qlora"
- lora_rank: 8, 16, 32, or 64 (higher = more capacity but larger adapter)

When choosing pruning, specify:
- method: One of "magnitude", "structured"
- sparsity: Target sparsity ratio (0.1 to 0.7, e.g., 0.3 = 30% weights removed)
- granularity: For structured pruning, one of "channel", "head"

Compression strategy tips:
- Use LoRA/QLoRA to recover accuracy after aggressive quantization (e.g., 4-bit)
- QLoRA is preferred for large models due to memory efficiency
- Pruning can be combined with quantization - prune first, then quantize for maximum compression

Always explain your reasoning before deciding."""

QUANTIZATION_AGENT_PROMPT = """You are a quantization specialist agent.
Your job is to apply quantization to compress neural network models.

Available methods:
- autoround: Adaptive rounding, good for maintaining accuracy (supports 2,3,4,8 bits)
- gptq: Post-training quantization, fast and efficient (supports 2,3,4,8 bits)
- int8: 8-bit quantization via BitsAndBytes, simple and reliable
- awq: Activation-aware weight quantization, best accuracy preservation (supports 4 bits)
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
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        max_episodes: int = 10,
        budget_hours: float = 2.0,
        experiment_name: Optional[str] = None,
    ):
        """Initialize the LangGraph coordinator.

        Args:
            model_name: HuggingFace model name to compress
            dataset: Target benchmark dataset
            max_episodes: Maximum compression episodes
            budget_hours: Time budget in hours
            experiment_name: Name for this experiment
        """
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model_name = model_name
        self.dataset = dataset
        self.max_episodes = max_episodes
        self.budget_hours = budget_hours

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

        # Initialize LLMs
        self.llm_coordinator = ChatOpenAI(
            model="gpt-5.2",
            temperature=0.7,
        )
        self.llm_worker = ChatOpenAI(
            model="gpt-4o",
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
                "search": "search",
                "end": END,
            }
        )

        # Fixed edges
        graph.add_edge("quantization", "evaluation")
        graph.add_edge("pruning", "evaluation")
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
        # Route lora/qlora to the quantization node (handles all compression methods)
        if action in ["quantization", "lora", "qlora"]:
            return "quantization"
        if action in ["pruning", "search"]:
            return action
        return "end"

    def _coordinator_node(self, state: CompressionState) -> Dict[str, Any]:
        """Coordinator node: LLM decides the next action.

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

        context = f"""
Current State:
- Model: {state['model_name']}
- Dataset: {state['dataset']}
- Episode: {state['current_episode'] + 1}/{state['max_episodes']}
- Elapsed: {self._get_elapsed_hours(state):.2f}h / {state['budget_hours']}h
- Consecutive no-improvement: {state['consecutive_no_improvement']}

{pareto_summary}

{recent_history}

Model Info:
- Size: {state['model_spec'].get('model_size_gb', 'unknown')} GB
- Family: {state['model_spec'].get('model_family', 'unknown')}
- Preferred methods: {state['model_spec'].get('preferred_methods', [])}

What should be the next action? Choose one:
1. "quantization" with specific method and bits
2. "lora" or "qlora" with lora_rank for fine-tuning
3. "pruning" with method, sparsity, and granularity
4. "search" to explore new strategies
5. "end" if converged or should stop

Respond in JSON format:
For quantization: {{"action": "quantization", "method": "autoround|gptq|int8|awq", "bits": 4, "reasoning": "..."}}
For LoRA/QLoRA: {{"action": "lora|qlora", "method": "lora|qlora", "lora_rank": 16, "reasoning": "..."}}
For pruning: {{"action": "pruning", "method": "magnitude|structured", "sparsity": 0.3, "granularity": "weight|channel|head", "reasoning": "..."}}
"""

        # Call LLM
        messages = [
            SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = self.llm_coordinator.invoke(messages)

        # Parse response
        try:
            # Extract JSON from response
            response_text = response.content
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
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
        if action == "quantization":
            action_params = {
                "method": decision.get("method", "gptq"),
                "bits": decision.get("bits", 4),
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

        return {
            "next_action": action,
            "action_params": action_params,
            "messages": [
                AIMessage(content=f"Decided: {action}. {reasoning}")
            ],
        }

    def _quantization_node(self, state: CompressionState) -> Dict[str, Any]:
        """Quantization node: Apply compression strategy (quantization or LoRA/QLoRA).

        Args:
            state: Current compression state

        Returns:
            State updates with current_strategy and compressed_model_path
        """
        params = state.get("action_params", {})
        method = params.get("method", "gptq")

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
        try:
            eval_result = evaluate_model.invoke({
                "checkpoint_path": compressed_path or state["model_name"],
                "benchmarks": [state["dataset"]],
                "use_proxy": True,
                "proxy_samples": 200,
            })

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
            )
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            # Create minimal result
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
            )

        print(f"  Result: acc={result.accuracy:.3f}, latency={result.latency_ms:.1f}ms")

        return {
            "current_result": result.model_dump(),
        }

    def _search_node(self, state: CompressionState) -> Dict[str, Any]:
        """Search node: Use optimization to suggest strategies.

        Args:
            state: Current compression state

        Returns:
            State updates with suggested action_params
        """
        print("  Running search optimization...")

        history = state.get("history", [])

        try:
            # Use multi-armed bandit for method selection
            suggestion = multi_armed_bandit_search.invoke({
                "history": history,
                "available_methods": ["autoround", "gptq", "int8", "awq"],
                "exploration_rate": 0.2,
            })

            method = suggestion.get("recommended_method", "gptq")
            bits = suggestion.get("recommended_bits", 4)
        except Exception as e:
            print(f"  Search failed: {e}")
            method = "gptq"
            bits = 4

        print(f"  Search suggests: {method} {bits}-bit")

        return {
            "next_action": "quantization",
            "action_params": {"method": method, "bits": bits},
        }

    def _update_state_node(self, state: CompressionState) -> Dict[str, Any]:
        """Update state after evaluation.

        Args:
            state: Current compression state

        Returns:
            State updates with history, pareto_frontier, episode counter
        """
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

        return {
            "history": [new_history_entry],
            "pareto_frontier": pareto_solutions,
            "current_episode": state["current_episode"] + 1,
            "consecutive_no_improvement": consecutive_no_improvement,
            "current_strategy": None,
            "current_result": None,
            "compressed_model_path": None,
            "compression_ratio": None,
            "compressed_model_size_gb": None,
        }

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

        # Run the graph
        try:
            if interactive:
                # Stream for visibility
                for step in self.graph.stream(initial_state):
                    node_name = list(step.keys())[0]
                    if node_name == "__end__":
                        break
            else:
                final_state = self.graph.invoke(initial_state)
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

        # Get best solutions
        best_solutions = {
            "accuracy": None,
            "latency": None,
            "size": None,
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
