#!/usr/bin/env python3
"""
CommonsenseQA Evaluation Script for A2A Pipeline
Tests multiple choice question answering with A/B/C/D/E format
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

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
from agents.search import SearchAgent  # Import search functionality
from utils.logging import setup_logger, get_logger, PerformanceLogger
from utils.text_processing import extract_final_answer
from utils.metrics import compute_accuracy, compute_f1, compute_exact_match
import gc
import torch

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("commonsenseqa_evaluation")

def format_commonsenseqa_question(example: Dict[str, Any]) -> str:
    """
    Format a CommonsenseQA example into a multiple choice question

    Args:
        example: CommonsenseQA example with question and choices

    Returns:
        Formatted question string
    """
    question = example["question"]
    choices = example["choices"]

    # Format choices as A, B, C, D, E
    choice_lines = []
    for i, (label, text) in enumerate(zip(choices["label"], choices["text"])):
        choice_lines.append(f"{label}. {text}")

    formatted_question = f"{question}\n\n" + "\n".join(choice_lines)
    return formatted_question

def extract_letter_answer(output: str) -> str:
    """
    Extract single letter answer (A, B, C, D, E) from model output

    Args:
        output: Raw model output

    Returns:
        Single letter answer or empty string if not found
    """
    # Look for patterns like "A", "B", etc.
    import re

    # First try to find a standalone letter at the end
    matches = re.findall(r'\b([A-E])\b', output)
    if matches:
        return matches[-1]  # Return the last match

    # Try to find letter after common patterns
    patterns = [
        r'answer is ([A-E])',
        r'choose ([A-E])',
        r'option ([A-E])',
        r'select ([A-E])',
        r'^([A-E])$',  # Just the letter alone
        r'\n([A-E])\n',  # Letter on its own line
        r'([A-E])\.',  # Letter followed by period
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    return ""

def run_commonsenseqa_evaluation(sample_size: int = 100):
    """Run CommonsenseQA evaluation with carbon emission tracking"""

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="CommonsenseQA_A2A_Pipeline",
        output_dir="./carbon_emissions",
        output_file=f"commonsenseqa_emissions_{int(time.time())}.csv",
        log_level="INFO",
        measure_power_secs=10,  # Measure power every 10 seconds
        save_to_file=True,
        save_to_api=False,
        co2_signal_api_token=None,  # Optional: add API token for more accurate data
        tracking_mode="process"
    )

    # Start tracking emissions
    tracker.start()

    logger.info("🚀 Starting CommonsenseQA Evaluation with Carbon Tracking")
    logger.info(f"📊 Sample size: {sample_size}")

    # Load dataset
    logger.info("📚 Loading CommonsenseQA dataset...")
    try:
        dataset = load_dataset("commonsense_qa")
        validation_data = dataset["validation"]

        # Sample data for testing
        if sample_size and sample_size < len(validation_data):
            validation_data = validation_data.select(range(sample_size))

        logger.info(f"✅ Loaded {len(validation_data)} validation examples")

    except Exception as e:
        logger.error(f"Failed to load CommonsenseQA dataset: {e}")
        tracker.stop()  # Stop tracking on error
        return

    # Initialize components sequentially to avoid CUDA OOM
    logger.info("🧠 Initializing model components...")

    model_manager = None
    router_agent = None
    solver_agent = None
    judge_agent = None
    search_agent = None

    try:
        # Initialize model manager
        with PerformanceLogger("model_manager initialization"):
            model_manager = ModelManager()
            if not model_manager.load_model():
                raise RuntimeError("Failed to load model")
            logger.info("✅ ModelManager initialized")

        # Initialize router agent
        with PerformanceLogger("router initialization"):
            router_agent = RouterAgent()
            logger.info("✅ RouterAgent initialized")

        # Initialize solver agent with model manager
        with PerformanceLogger("solver initialization"):
            solver_agent = SolverAgent(model_client=model_manager)
            logger.info("✅ SolverAgent initialized")

        # Initialize judge agent
        with PerformanceLogger("judge initialization"):
            judge_agent = JudgeAgent()
            logger.info("✅ JudgeAgent initialized")

        # Initialize search agent for enhanced reasoning
        with PerformanceLogger("search initialization"):
            search_agent = SearchAgent()
            search_health = search_agent.health_check()
            logger.info(f"✅ SearchAgent initialized - Health: {search_health}")
            if search_health.get("tavily_client", False):
                logger.info("🔍 Tavily search API is available for enhanced reasoning")
            else:
                logger.warning("⚠️  Tavily search API not available - using local reasoning only")

        # Run evaluation
        results = []
        correct_predictions = 0
        total_questions = len(validation_data)

        logger.info(f"🎯 Starting evaluation of {total_questions} questions...")

        for i, example in enumerate(validation_data):
            question_id = example.get("id", f"q_{i}")
            formatted_question = format_commonsenseqa_question(example)
            gold_answer = example["answerKey"]

            logger.info(f"\n=== Question {i+1}/{total_questions} (ID: {question_id}) ===")
            logger.info(f"Question: {example['question']}")
            logger.info(f"Gold Answer: {gold_answer}")

            try:
                start_time = time.time()

                # Route the question (should be multiple_choice)
                routing_decision = router_agent.route(
                    query=formatted_question,
                    dataset="commonsenseqa"
                )

                logger.info(f"🎯 Routed as: {routing_decision.task_type}")

                # Use search for ALL CommonsenseQA questions to build proper RAG
                search_context = None
                if search_agent and search_health.get("tavily_client", False):
                    try:
                        logger.info(f"🔍 Using search for CommonsenseQA RAG enhancement")
                        # Use search for every question to gather relevant context
                        search_context = search_agent.retrieve_and_rerank(
                            query=example["question"],
                            top_k=3
                        )
                        if search_context.passages:
                            logger.info(f"🔍 Retrieved {len(search_context.passages)} passages for RAG context")
                        else:
                            logger.warning(f"🔍 No passages retrieved for question")
                    except Exception as search_error:
                        logger.warning(f"Search failed, continuing without external context: {search_error}")

                # Solve the question with optional search context
                context = search_agent.format_context_for_llm(search_context) if search_context and search_context.passages else None
                solver_response = solver_agent.solve(
                    question=formatted_question,
                    routing_decision=routing_decision,
                    context=context
                )

                # Extract letter answer
                predicted_answer = extract_letter_answer(solver_response.answer)
                if not predicted_answer:
                    predicted_answer = extract_letter_answer(solver_response.raw_output)

                # Judge the answer
                judge_result = judge_agent.judge_single(
                    prediction=predicted_answer,
                    gold_answer=gold_answer,
                    task_type="multiple_choice"
                )

                elapsed_time = time.time() - start_time

                # Check correctness
                is_correct = predicted_answer.upper() == gold_answer.upper()
                if is_correct:
                    correct_predictions += 1

                logger.info(f"🤖 Predicted: {predicted_answer}")
                logger.info(f"🎯 Gold: {gold_answer}")
                logger.info(f"{'✅' if is_correct else '❌'} Correct: {is_correct}")
                logger.info(f"⏱️  Time: {elapsed_time:.2f}s")

                # Store result with search context information
                result = {
                    "id": question_id,
                    "question": example["question"],
                    "choices": example["choices"],
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
                    "judge_result": {
                        "is_correct": judge_result.correct,
                        "score": judge_result.score,
                        "explanation": judge_result.explanation
                    },
                    "search_context": {
                        "used": search_context is not None and search_context.passages,
                        "num_passages": len(search_context.passages) if search_context and search_context.passages else 0,
                        "search_time": search_context.search_time if search_context else 0,
                        "reranked": search_context.reranked if search_context else False,
                        "passages_summary": [{"title": p.title, "url": p.url, "score": p.score} for p in search_context.passages] if search_context and search_context.passages else []
                    }
                }
                results.append(result)

                # Log running accuracy
                current_accuracy = correct_predictions / (i + 1)
                logger.info(f"📊 Running Accuracy: {current_accuracy:.3f} ({correct_predictions}/{i+1})")

            except Exception as e:
                logger.error(f"❌ Failed to process question {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Store error result
                result = {
                    "id": question_id,
                    "question": example["question"],
                    "choices": example["choices"],
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
        logger.info(f"\n🎉 FINAL RESULTS:")
        logger.info(f"📊 Accuracy: {final_accuracy:.3f} ({correct_predictions}/{total_questions})")

        # Save results
        timestamp = int(time.time())
        results_file = f"commonsenseqa_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "dataset": "commonsenseqa",
                    "sample_size": len(validation_data),
                    "total_questions": total_questions,
                    "correct_predictions": correct_predictions,
                    "accuracy": final_accuracy,
                    "timestamp": timestamp
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

        # Clean up GPU memory
        logger.info("🧹 Cleaning up GPU memory...")
        if model_manager:
            model_manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    run_commonsenseqa_evaluation(sample_size=100)  # Run proper evaluation with 100 samples