#!/usr/bin/env python3
"""
GSM8K Evaluation with Lazy Model Loading
=========================================

This script evaluates the A2A pipeline on GSM8K and GSM8K-Platinum datasets
using lazy model loading - models are only loaded when needed and unloaded
after use to minimize VRAM usage.

Key Features:
- On-demand model loading (load only when needed)
- Automatic model unloading after use
- Sequential agent activation
- Minimal VRAM footprint
- Full A2A pipeline without compromising architecture

Usage:
    python run_gsm8k_evaluation_lazy.py --limit 5
    python run_gsm8k_evaluation_lazy.py  # Full evaluation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import gc
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset
from loguru import logger
from codecarbon import EmissionsTracker

# Import agents and components
from agents.router import RouterAgent, RoutingDecision
from agents.solver import SolverAgent
from agents.judge import JudgeAgent
from models.model_manager import ModelManager
from utils.logging import PerformanceLogger

# GSM8K Prompt Template (proven successful from Colab)
GSM8K_PROMPT_TEMPLATE = r"""You are a precise mathematical reasoning assistant.
Please proceed as follows:
1. First give your step-by-step reasoning (you may use `<think> … </think>` tags).
2. Then on a new line, output your final numeric answer using exactly:
   `The answer is \boxed{{<number>}}`
   — with no additional text on that line.

Here are examples:

Q: A bag contains 3 red balls and 4 blue balls. If you remove 2 blue balls, how many balls remain total?
<think>
Total initial = 3 + 4 = 7
After removing 2 blue = 4 − 2 = 2
Total = 5
</think>
The answer is \boxed{{5}}

Q: Maria had 15 apples. She gave 7 to John and then bought 4 more. How many apples does she have now?
<think>
15 − 7 = 8
8 + 4 = 12
</think>
The answer is \boxed{{12}}

Now solve:
Q: {question}
"""


class GSM8KSolverAgent(SolverAgent):
    """Enhanced SolverAgent with proven GSM8K prompt template"""

    def get_gsm8k_system_prompt(self) -> str:
        """GSM8K specific system prompt"""
        return "You are a precise mathematical reasoning assistant specialized in solving math word problems step by step."

    def format_gsm8k_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format question using proven GSM8K prompt template"""
        # Use string replacement instead of .format() to avoid KeyError with <number>
        base_prompt = GSM8K_PROMPT_TEMPLATE.replace("{question}", question.strip())

        if context:
            # Add context if search is used (though math problems typically don't need it)
            base_prompt = f"Additional context:\n{context}\n\n" + base_prompt

        return base_prompt

    def solve(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve a GSM8K math problem using self-consistency

        Args:
            question: The math question to solve
            context: Optional additional context (rarely needed for math)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing the solution (with "response" key for compatibility)
        """
        # Format the prompt using GSM8K template
        user_prompt = self.format_gsm8k_prompt(question, context)
        system_prompt = self.get_gsm8k_system_prompt()

        # Create routing decision for self-consistency
        routing_decision = RoutingDecision(
            task_type="math",
            use_search=False,
            prompt_style="math",
            decoding_strategy="self_consistency",
            num_samples=10,
            temperature=0.6
        )

        # Use parent class's self-consistency method
        response = self._solve_with_self_consistency(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            routing_decision=routing_decision
        )

        # Return in format expected by evaluation script (with "response" key)
        return {
            "response": response.answer if hasattr(response, 'answer') else response.raw_output,
            "reasoning": response.reasoning if hasattr(response, 'reasoning') else "",
            "confidence": response.confidence if hasattr(response, 'confidence') else 1.0,
            "raw_output": response.raw_output if hasattr(response, 'raw_output') else str(response)
        }


class LazyModelManager:
    """
    ModelManager wrapper that loads/unloads models on demand.

    This allows running the full A2A pipeline with minimal VRAM usage
    by only keeping one model in memory at a time.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize lazy model manager."""
        self.config = config or {}
        self._model_manager: Optional[ModelManager] = None
        self._is_loaded = False
        logger.info("🔧 LazyModelManager initialized (models will load on-demand)")

    def load(self):
        """Load the model into memory."""
        if self._is_loaded:
            logger.debug("Model already loaded, skipping")
            return

        logger.info("📥 Loading model on-demand...")
        with PerformanceLogger("model loading"):
            self._model_manager = ModelManager(self.config)
            if not self._model_manager.load_model():
                raise RuntimeError("Failed to load model")
            self._is_loaded = True
        logger.info("✅ Model loaded successfully")

    def unload(self):
        """Unload the model from memory."""
        if not self._is_loaded:
            logger.debug("Model not loaded, skipping unload")
            return

        logger.info("📤 Unloading model to free VRAM...")
        if self._model_manager:
            self._model_manager.cleanup()
            del self._model_manager
            self._model_manager = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._is_loaded = False
        logger.info("✅ Model unloaded, VRAM freed")

    def generate(self, *args, **kwargs):
        """Generate text using the model (loads if needed)."""
        if not self._is_loaded:
            self.load()
        return self._model_manager.generate(*args, **kwargs)

    @property
    def model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()


class LazyGSM8KSolverAgent(GSM8KSolverAgent):
    """
    GSM8K Solver Agent with lazy model loading.

    Extends the standard GSM8KSolverAgent to support on-demand
    model loading and unloading.
    """

    def __init__(self, model_client: LazyModelManager):
        """Initialize with lazy model manager."""
        # Initialize parent with the lazy manager
        super().__init__(model_client=model_client)
        self.lazy_model = model_client
        logger.info("🧠 LazyGSM8KSolverAgent initialized")

    def solve(self, question: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve a question using lazy loading.

        The model is loaded before solving and can be unloaded after
        by the caller.
        """
        # Ensure model is loaded
        if not self.lazy_model.model_loaded:
            logger.info("🔄 Loading model for solving...")
            self.lazy_model.load()

        # Call parent solve method
        return super().solve(question, context, **kwargs)


def extract_gsm8k_answer(output: str) -> Optional[str]:
    """
    Extract numerical answer from model output.

    Supports multiple formats:
    1. \\boxed{number}
    2. "The answer is X"
    3. Numeric fallback
    """
    import re

    # Try boxed format first
    boxed_matches = re.findall(r"\\boxed\s*\{\s*([-]?\d+\.?\d*)\s*\}", output)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Try "answer is" format
    answer_phrases = [
        r"(?:the\s+)?answer\s+is\s+\$?\s*([-]?\d+\.?\d*)",
        r"(?:final\s+)?answer:\s*\$?\s*([-]?\d+\.?\d*)",
    ]
    for pattern in answer_phrases:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Fallback: last number in output
    numbers = re.findall(r"[-]?\d+\.?\d*", output)
    if numbers:
        return numbers[-1].strip()

    return None


def run_gsm8k_evaluation_lazy(
    dataset_name: str = "gsm8k",
    limit: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run GSM8K evaluation with lazy model loading.

    Args:
        dataset_name: "gsm8k" or "gsm8k_platinum"
        limit: Optional limit on number of questions
        output_file: Optional output file path

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"🚀 Starting GSM8K Evaluation with Lazy Loading on {dataset_name}")

    if limit:
        logger.info(f"📊 Limiting evaluation to {limit} questions")
    else:
        logger.info(f"📊 Full dataset evaluation")

    # Load dataset
    logger.info(f"📚 Loading {dataset_name} dataset...")
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "gsm8k_platinum":
        dataset = load_dataset("reasoning-machines/gsm8k-platinum", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    logger.info(f"✅ Loaded {len(dataset)} examples")

    # Initialize carbon tracking
    timestamp = int(time.time())
    emissions_filename = f"{dataset_name}_lazy_emissions_{timestamp}"
    os.makedirs("./carbon_emissions", exist_ok=True)

    tracker = EmissionsTracker(
        project_name=f"GSM8K_Lazy_{dataset_name}",
        output_dir="./carbon_emissions",
        output_file=emissions_filename,
        measure_power_secs=10,
        log_level="warning"
    )
    tracker.start()

    try:
        # Initialize components with lazy loading
        logger.info("🧠 Initializing components with lazy loading...")

        with PerformanceLogger("lazy manager initialization"):
            # Create lazy model manager (doesn't load yet)
            # Note: ModelManager will use global config which reads from environment variables
            lazy_model = LazyModelManager()

        with PerformanceLogger("router initialization"):
            router = RouterAgent()
            logger.info("✅ RouterAgent initialized")

        with PerformanceLogger("solver initialization"):
            # Solver with lazy loading
            solver = LazyGSM8KSolverAgent(model_client=lazy_model)
            logger.info("✅ LazyGSM8KSolverAgent initialized")

        with PerformanceLogger("judge initialization"):
            judge = JudgeAgent()
            logger.info("✅ JudgeAgent initialized")

        # Evaluation loop
        results = []
        correct = 0
        total = 0

        logger.info(f"🎯 Starting evaluation of {len(dataset)} questions...")

        for idx, example in enumerate(dataset, 1):
            question = example["question"]
            gold_answer = str(example["answer"]).split("####")[-1].strip()

            logger.info(f"\n=== Question {idx}/{len(dataset)} ===")
            logger.info(f"Question: {question}")
            logger.info(f"Gold Answer: {gold_answer}")

            try:
                with PerformanceLogger(f"question {idx}"):
                    # Step 1: Route (no model needed)
                    routing = router.route(question, dataset=dataset_name)
                    logger.info(f"🎯 Routed as: {routing.task_type}")

                    # Step 2: Load model and solve
                    logger.info("🔄 Loading model for this question...")
                    lazy_model.load()

                    with PerformanceLogger(f"solving question {idx}"):
                        solution = solver.solve(question)

                    # Step 3: Unload model immediately after solving
                    logger.info("🔄 Unloading model...")
                    lazy_model.unload()

                    # Step 4: Extract and judge (no model needed)
                    predicted_answer = extract_gsm8k_answer(solution["response"])

                    judgment = judge.judge_single(
                        prediction=predicted_answer,
                        gold_answer=gold_answer,
                        task_type="math"
                    )
                    is_correct = judgment.correct

                    if is_correct:
                        correct += 1
                    total += 1

                logger.info(f"🤖 Predicted: {predicted_answer}")
                logger.info(f"🎯 Gold: {gold_answer}")
                logger.info(f"{'✅' if is_correct else '❌'} Correct: {is_correct}")

                results.append({
                    "question_id": idx,
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct,
                    "raw_output": solution["response"][:500]  # Truncate for storage
                })

                # Progress update every 10 questions
                if idx % 10 == 0:
                    accuracy = correct / total
                    logger.info(f"📊 Progress: {idx}/{len(dataset)} | Accuracy: {accuracy:.3f} ({correct}/{total})")

            except Exception as e:
                logger.error(f"❌ Failed to process question {idx}: {e}")
                # Make sure to unload model even on error
                lazy_model.unload()
                continue

        # Final results
        accuracy = correct / total if total > 0 else 0.0

        logger.info("\n" + "="*60)
        logger.info(f"📊 Final Results for {dataset_name}")
        logger.info("="*60)
        logger.info(f"Total Questions: {total}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Stop carbon tracking
        emissions = tracker.stop()
        if emissions:
            logger.info(f"🌱 Carbon Emissions: {emissions:.6f} kg CO2 equivalent")
        else:
            logger.warning("Carbon emissions tracking returned None")
            emissions = 0.0

        # Prepare output
        output_data = {
            "metadata": {
                "dataset": dataset_name,
                "total_questions": total,
                "correct": correct,
                "accuracy": accuracy,
                "carbon_emissions_kg": emissions,
                "loading_strategy": "lazy",
                "timestamp": timestamp
            },
            "results": results
        }

        # Save results
        if output_file is None:
            output_file = f"{dataset_name}_lazy_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"💾 Results saved to: {output_file}")

        # Cleanup
        lazy_model.unload()

        return output_data

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        tracker.stop()
        raise

    finally:
        # Final cleanup
        logger.info("🧹 Final cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GSM8K Evaluation with Lazy Model Loading"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (default: full dataset)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "gsm8k_platinum", "both"],
        default="both",
        help="Dataset to evaluate (default: both)"
    )

    args = parser.parse_args()

    logger.info("🚀 Starting GSM8K Lazy Loading Evaluation")
    logger.info("="*60)

    if args.dataset in ["gsm8k", "both"]:
        logger.info("\n" + "="*60)
        logger.info("📊 Evaluating GSM8K Main Dataset")
        logger.info("="*60)

        output_file = args.output or f"gsm8k_lazy_results_{int(time.time())}.json"
        run_gsm8k_evaluation_lazy(
            dataset_name="gsm8k",
            limit=args.limit,
            output_file=output_file
        )

    if args.dataset in ["gsm8k_platinum", "both"]:
        logger.info("\n" + "="*60)
        logger.info("📊 Evaluating GSM8K Platinum Dataset")
        logger.info("="*60)

        output_file = args.output or f"gsm8k_platinum_lazy_results_{int(time.time())}.json"
        run_gsm8k_evaluation_lazy(
            dataset_name="gsm8k_platinum",
            limit=args.limit,
            output_file=output_file
        )

    logger.info("\n✅ All evaluations complete!")


if __name__ == "__main__":
    main()
