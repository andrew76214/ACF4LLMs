#!/usr/bin/env python3
"""
Fast CommonsenseQA Evaluation Script for A2A Pipeline
Tests multiple choice question answering with A/B/C/D/E format
Uses Tavily search without BGE reranking for speed
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
from utils.logging import setup_logger, get_logger, PerformanceLogger
from utils.text_processing import extract_final_answer
from utils.metrics import compute_accuracy, compute_f1, compute_exact_match
import gc
import torch

# Simple Tavily search without BGE
from tavily import TavilyClient

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("commonsenseqa_fast_evaluation")

def format_commonsenseqa_question(example: Dict[str, Any]) -> str:
    """Format a CommonsenseQA example into a multiple choice question"""
    question = example["question"]
    choices = example["choices"]

    # Format choices as A, B, C, D, E
    choice_lines = []
    for i, (label, text) in enumerate(zip(choices["label"], choices["text"])):
        choice_lines.append(f"{label}. {text}")

    formatted_question = f"{question}\n\n" + "\n".join(choice_lines)
    return formatted_question

def extract_letter_answer(output: str) -> str:
    """Extract single letter answer (A, B, C, D, E) from model output"""
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

def simple_tavily_search(query: str, top_k: int = 3) -> str:
    """Simple Tavily search without BGE reranking"""
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        results = client.search(
            query=query,
            search_depth="basic",
            max_results=top_k,
            include_answer=True,
            include_raw_content=False
        )

        if not results.get('results'):
            return ""

        # Format context from results
        context_parts = []
        for i, result in enumerate(results['results'][:top_k]):
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            url = result.get('url', '')

            context_parts.append(f"Source {i+1}: {title}\n{content[:500]}...\nURL: {url}")

        return "\n\n".join(context_parts)
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return ""

def run_fast_commonsenseqa_evaluation(sample_size: int = 50):
    """Run fast CommonsenseQA evaluation with carbon emission tracking"""

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="CommonsenseQA_A2A_Fast",
        output_dir="./carbon_emissions",
        output_file=f"commonsenseqa_fast_emissions_{int(time.time())}.csv",
        log_level="INFO",
        measure_power_secs=10,
        save_to_file=True,
        save_to_api=False,
        co2_signal_api_token=None,
        tracking_mode="process"
    )

    # Start tracking emissions
    tracker.start()

    logger.info(f"🚀 Starting FAST CommonsenseQA Evaluation with Carbon Tracking")
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
        tracker.stop()
        return

    # Initialize components sequentially to avoid CUDA OOM
    logger.info("🧠 Initializing model components...")

    model_manager = None
    router_agent = None
    solver_agent = None
    judge_agent = None

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

                # Fast Tavily search without BGE
                search_context = None
                try:
                    logger.info("🔍 Using fast Tavily search for RAG enhancement")
                    search_context = simple_tavily_search(example["question"], top_k=3)
                    if search_context:
                        logger.info(f"🔍 Retrieved search context ({len(search_context)} chars)")
                    else:
                        logger.warning("🔍 No search context retrieved")
                except Exception as search_error:
                    logger.warning(f"Search failed, continuing without context: {search_error}")

                # Solve the question with optional search context
                solver_response = solver_agent.solve(
                    question=formatted_question,
                    routing_decision=routing_decision,
                    context=search_context
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

                # Store result
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
                        "used": search_context is not None and len(search_context) > 0,
                        "length": len(search_context) if search_context else 0,
                        "fast_tavily": True
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
        results_file = f"commonsenseqa_fast_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "dataset": "commonsenseqa",
                    "sample_size": len(validation_data),
                    "total_questions": total_questions,
                    "correct_predictions": correct_predictions,
                    "accuracy": final_accuracy,
                    "timestamp": timestamp,
                    "method": "fast_tavily_no_reranking"
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
    run_fast_commonsenseqa_evaluation(sample_size=50)  # Run with 50 samples for speed