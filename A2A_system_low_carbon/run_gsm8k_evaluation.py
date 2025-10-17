#!/usr/bin/env python3
"""
GSM8K Evaluation Script for A2A Pipeline
Integrates successful Colab techniques with A2A architecture
Uses proven prompt template, generation settings, and answer parsing
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"

from datasets import load_dataset
from codecarbon import EmissionsTracker
from models.model_manager import ModelManager
from agents.router import RouterAgent, RoutingDecision
from agents.solver import SolverAgent
from agents.judge import JudgeAgent
from utils.logging import setup_logger, get_logger, PerformanceLogger
import gc
import torch

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("gsm8k_evaluation")

# Proven successful prompt template from Colab
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

def extract_gsm8k_answer(output: str) -> Optional[str]:
    """
    Robust answer extraction using proven Colab technique
    Try: boxed -> phrase -> numeric fallback
    """
    # Try boxed format first
    boxed_matches = re.findall(r"\\boxed\s*\{\s*([-]?\d+\.?\d*)\s*\}", output)
    if boxed_matches:
        return boxed_matches[-1]

    # Try "The answer is" phrase
    phrase_matches = re.findall(r"The answer is\s*([-]?\d+\.?\d*)", output, re.IGNORECASE)
    if phrase_matches:
        return phrase_matches[-1]

    # Fallback to any numeric value
    numeric_matches = re.findall(r"[-]?\d+\.?\d*", output)
    if numeric_matches:
        return numeric_matches[-1]

    return None

def extract_ground_truth_answer(answer_text: str) -> Optional[str]:
    """Extract ground truth answer from GSM8K format"""
    # GSM8K format typically has #### followed by the answer
    if "####" in answer_text:
        part = answer_text.split("####")[-1].strip()
        # Remove trailing punctuation
        part = part.rstrip(". ")
        return part

    # Fallback to last numeric token
    numeric_matches = re.findall(r"[-]?\d+\.?\d*", answer_text)
    if numeric_matches:
        return numeric_matches[-1]

    return None

def compare_numeric_answers(pred_str: str, gold_str: str) -> bool:
    """Compare numeric answers with floating point tolerance"""
    try:
        if "." in pred_str or "." in gold_str:
            pred_val = float(pred_str)
            gold_val = float(gold_str)
        else:
            pred_val = int(pred_str)
            gold_val = int(gold_str)

        return abs(pred_val - gold_val) < 1e-9
    except (ValueError, TypeError):
        # Fallback to string comparison
        return pred_str.strip() == gold_str.strip()

def run_gsm8k_evaluation(dataset_name: str = "gsm8k", sample_size: Optional[int] = None):
    """Run GSM8K evaluation with proven techniques"""

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"GSM8K_A2A_{dataset_name}",
        output_dir="./carbon_emissions",
        output_file=f"gsm8k_{dataset_name}_emissions_{int(time.time())}.csv",
        log_level="INFO",
        measure_power_secs=10,
        save_to_file=True,
        save_to_api=False,
        co2_signal_api_token=None,
        tracking_mode="process"
    )

    # Start tracking emissions
    tracker.start()

    logger.info(f"🚀 Starting GSM8K Evaluation on {dataset_name}")
    if sample_size:
        logger.info(f"📊 Sample size: {sample_size}")
    else:
        logger.info(f"📊 Full dataset evaluation")

    # Load dataset
    logger.info(f"📚 Loading {dataset_name} dataset...")
    try:
        if dataset_name == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main")
            eval_data = dataset["test"]
        elif dataset_name == "gsm8k_platinum":
            dataset = load_dataset("madrylab/gsm8k-platinum", "main")
            eval_data = dataset["test"]  # or "train" depending on split
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Sample data if specified
        if sample_size and sample_size < len(eval_data):
            eval_data = eval_data.select(range(sample_size))

        logger.info(f"✅ Loaded {len(eval_data)} examples")

    except Exception as e:
        logger.error(f"Failed to load {dataset_name} dataset: {e}")
        tracker.stop()
        return

    # Initialize components SEQUENTIALLY to avoid CUDA OOM
    logger.info("🧠 Initializing model components sequentially...")

    model_manager = None
    router_agent = None
    solver_agent = None
    judge_agent = None

    try:
        # Step 1: Initialize model manager with proven settings
        with PerformanceLogger("model_manager initialization"):
            model_manager = ModelManager()
            # Override generation config for GSM8K proven settings
            if hasattr(model_manager, 'generation_config'):
                model_manager.generation_config.update({
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,
                    "num_return_sequences": 1,
                })
            if not model_manager.load_model():
                raise RuntimeError("Failed to load model")
            logger.info("✅ ModelManager initialized with GSM8K settings")

        # Step 2: Initialize router agent
        with PerformanceLogger("router initialization"):
            router_agent = RouterAgent()
            logger.info("✅ RouterAgent initialized")

        # Step 3: Initialize enhanced solver agent
        with PerformanceLogger("solver initialization"):
            solver_agent = GSM8KSolverAgent(model_client=model_manager)
            logger.info("✅ GSM8KSolverAgent initialized")

        # Step 4: Initialize judge agent
        with PerformanceLogger("judge initialization"):
            judge_agent = JudgeAgent()
            logger.info("✅ JudgeAgent initialized")

        # Run evaluation
        results = []
        correct_predictions = 0
        total_questions = len(eval_data)

        logger.info(f"🎯 Starting evaluation of {total_questions} questions...")

        for i, example in enumerate(eval_data):
            question = example["question"]
            answer_text = example["answer"]
            gold_answer = extract_ground_truth_answer(answer_text)

            logger.info(f"\n=== Question {i+1}/{total_questions} ===")
            logger.info(f"Question: {question}")
            logger.info(f"Gold Answer: {gold_answer}")

            try:
                start_time = time.time()

                # Route the question (should be math for GSM8K)
                routing_decision = router_agent.route(
                    query=question,
                    dataset=dataset_name
                )

                logger.info(f"🎯 Routed as: {routing_decision.task_type}")

                # Create custom prompt for GSM8K
                gsm8k_prompt = solver_agent.format_gsm8k_prompt(question)

                # Solve using custom generation settings
                solver_response = solver_agent.solve(
                    question=gsm8k_prompt,
                    routing_decision=routing_decision,
                    context=None  # Math problems typically don't need search context
                )

                # Extract answer using proven technique
                predicted_answer = extract_gsm8k_answer(solver_response.answer)
                if not predicted_answer:
                    predicted_answer = extract_gsm8k_answer(solver_response.raw_output)

                elapsed_time = time.time() - start_time

                # Check correctness using numeric comparison
                is_correct = False
                if predicted_answer and gold_answer:
                    is_correct = compare_numeric_answers(predicted_answer, gold_answer)

                if is_correct:
                    correct_predictions += 1

                logger.info(f"🤖 Predicted: {predicted_answer}")
                logger.info(f"🎯 Gold: {gold_answer}")
                logger.info(f"{'✅' if is_correct else '❌'} Correct: {is_correct}")
                logger.info(f"⏱️  Time: {elapsed_time:.2f}s")

                # Store result
                result = {
                    "id": i,
                    "question": question,
                    "answer_text": answer_text,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "raw_output": solver_response.raw_output,
                    "reasoning": solver_response.reasoning,
                    "is_correct": is_correct,
                    "confidence": solver_response.confidence,
                    "elapsed_time": elapsed_time,
                    "routing_decision": {
                        "task_type": routing_decision.task_type,
                        "decoding_strategy": routing_decision.decoding_strategy,
                        "temperature": routing_decision.temperature
                    },
                    "gsm8k_prompt_used": gsm8k_prompt
                }
                results.append(result)

                # Log running accuracy
                current_accuracy = correct_predictions / (i + 1)
                logger.info(f"📊 Running Accuracy: {current_accuracy:.3f} ({correct_predictions}/{i+1})")

                # Debug first few examples and errors
                if i < 5 or not is_correct:
                    logger.info(f"🔍 Debug Info:")
                    logger.info(f"  Raw Output: {solver_response.raw_output[:200]}...")
                    logger.info(f"  Extraction: pred='{predicted_answer}', gold='{gold_answer}'")

            except Exception as e:
                logger.error(f"❌ Failed to process question {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Store error result
                result = {
                    "id": i,
                    "question": question,
                    "answer_text": answer_text,
                    "gold_answer": gold_answer,
                    "predicted_answer": "",
                    "raw_output": "",
                    "reasoning": "",
                    "is_correct": False,
                    "confidence": 0.0,
                    "elapsed_time": 0.0,
                    "error": str(e)
                }
                results.append(result)

        # Calculate final metrics
        final_accuracy = correct_predictions / total_questions
        logger.info(f"\n🎉 FINAL GSM8K RESULTS:")
        logger.info(f"📊 Dataset: {dataset_name}")
        logger.info(f"📊 Accuracy: {final_accuracy:.3f} ({correct_predictions}/{total_questions})")

        # Save results
        timestamp = int(time.time())
        results_file = f"gsm8k_{dataset_name}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "dataset": dataset_name,
                    "sample_size": len(eval_data),
                    "total_questions": total_questions,
                    "correct_predictions": correct_predictions,
                    "accuracy": final_accuracy,
                    "timestamp": timestamp,
                    "method": "a2a_pipeline_with_proven_colab_techniques",
                    "model": "unsloth/Qwen3-4B-Thinking-2507",
                    "generation_settings": {
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "top_k": 20,
                        "do_sample": True
                    }
                },
                "results": results
            }, f, indent=2)

        logger.info(f"💾 Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Stop carbon emissions tracking
        logger.info("🌱 Finalizing carbon emissions tracking...")
        try:
            emissions = tracker.stop()
            logger.info(f"🌱 Carbon Emissions: {emissions:.6f} kg CO2 equivalent")
            logger.info(f"🌱 Emissions data saved to: ./carbon_emissions/")
        except Exception as e:
            logger.error(f"Failed to finalize carbon tracking: {e}")

        # Clean up GPU memory sequentially
        logger.info("🧹 Cleaning up GPU memory...")
        if model_manager:
            model_manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    # Evaluate both datasets
    logger.info("🚀 Starting comprehensive GSM8K evaluation")

    # GSM8K main dataset (complete)
    logger.info("\n" + "="*50)
    logger.info("📊 Evaluating GSM8K Main Dataset")
    logger.info("="*50)
    run_gsm8k_evaluation(dataset_name="gsm8k", sample_size=None)  # Full dataset

    # Clean up between evaluations
    torch.cuda.empty_cache()
    gc.collect()

    # GSM8K Platinum dataset (complete)
    logger.info("\n" + "="*50)
    logger.info("📊 Evaluating GSM8K Platinum Dataset")
    logger.info("="*50)
    run_gsm8k_evaluation(dataset_name="gsm8k_platinum", sample_size=None)  # Full dataset