#!/usr/bin/env python3
"""
HotpotQA Evaluation Script for A2A Pipeline
Tests multi-hop reasoning with Tavily search and BGE reranking
Loads models sequentially to avoid CUDA OOM
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
from agents.search import SearchAgent
from utils.logging import setup_logger, get_logger, PerformanceLogger
from utils.text_processing import extract_final_answer
from utils.metrics import compute_accuracy, compute_f1, compute_exact_match
import gc
import torch

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("hotpotqa_evaluation")

def format_hotpotqa_question(example: Dict[str, Any]) -> str:
    """Format a HotpotQA example into a question"""
    question = example["question"]
    return question

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    import re
    import string

    # Remove articles
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # Fix whitespace
    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Compute exact match score after normalization"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def run_hotpotqa_evaluation(sample_size: int = 50):
    """Run HotpotQA evaluation with carbon emission tracking"""

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="HotpotQA_A2A_RAG",
        output_dir="./carbon_emissions",
        output_file=f"hotpotqa_emissions_{int(time.time())}.csv",
        log_level="INFO",
        measure_power_secs=10,
        save_to_file=True,
        save_to_api=False,
        co2_signal_api_token=None,
        tracking_mode="process"
    )

    # Start tracking emissions
    tracker.start()

    logger.info(f"🚀 Starting HotpotQA Evaluation with Tavily + BGE Reranking")
    logger.info(f"📊 Sample size: {sample_size}")

    # Load dataset
    logger.info("📚 Loading HotpotQA dataset...")
    try:
        dataset = load_dataset("hotpot_qa", "distractor")
        validation_data = dataset["validation"]

        # Sample data for testing
        if sample_size and sample_size < len(validation_data):
            validation_data = validation_data.select(range(sample_size))

        logger.info(f"✅ Loaded {len(validation_data)} validation examples")

    except Exception as e:
        logger.error(f"Failed to load HotpotQA dataset: {e}")
        tracker.stop()
        return

    # Initialize components SEQUENTIALLY to avoid CUDA OOM
    logger.info("🧠 Initializing model components sequentially...")

    model_manager = None
    router_agent = None
    solver_agent = None
    judge_agent = None
    search_agent = None

    try:
        # Step 1: Initialize model manager
        with PerformanceLogger("model_manager initialization"):
            model_manager = ModelManager()
            if not model_manager.load_model():
                raise RuntimeError("Failed to load model")
            logger.info("✅ ModelManager initialized")

        # Step 2: Initialize router agent
        with PerformanceLogger("router initialization"):
            router_agent = RouterAgent()
            logger.info("✅ RouterAgent initialized")

        # Step 3: Initialize solver agent with model manager
        with PerformanceLogger("solver initialization"):
            solver_agent = SolverAgent(model_client=model_manager)
            logger.info("✅ SolverAgent initialized")

        # Step 4: Initialize judge agent
        with PerformanceLogger("judge initialization"):
            judge_agent = JudgeAgent()
            logger.info("✅ JudgeAgent initialized")

        # Step 5: Initialize search agent with BGE models (SEQUENTIAL LOADING)
        with PerformanceLogger("search initialization"):
            logger.info("🔍 Initializing SearchAgent with BGE models (sequential loading)...")

            # Create search agent
            search_agent = SearchAgent()

            # Check health
            search_health = search_agent.health_check()
            logger.info(f"✅ SearchAgent initialized - Health: {search_health}")

            if not search_health.get("tavily_client", False):
                logger.warning("⚠️  Tavily client not available, continuing without search")
                search_agent = None
            else:
                logger.info("🔍 Tavily search API is available for enhanced reasoning")

        # Run evaluation
        results = []
        correct_predictions = 0
        total_questions = len(validation_data)

        logger.info(f"🎯 Starting evaluation of {total_questions} questions...")

        for i, example in enumerate(validation_data):
            question_id = example.get("id", f"q_{i}")
            formatted_question = format_hotpotqa_question(example)
            gold_answer = example["answer"]

            logger.info(f"\n=== Question {i+1}/{total_questions} (ID: {question_id}) ===")
            logger.info(f"Question: {formatted_question}")
            logger.info(f"Gold Answer: {gold_answer}")

            try:
                start_time = time.time()

                # Route the question (should be multihop for HotpotQA)
                routing_decision = router_agent.route(
                    query=formatted_question,
                    dataset="hotpotqa"
                )

                logger.info(f"🎯 Routed as: {routing_decision.task_type}")

                # Use search for ALL HotpotQA questions to build proper RAG
                search_context = None
                if search_agent and search_health.get("tavily_client", False):
                    try:
                        logger.info(f"🔍 Using search for HotpotQA multi-hop reasoning")
                        search_context = search_agent.retrieve_and_rerank(
                            query=formatted_question,
                            top_k=5  # More context for multi-hop reasoning
                        )
                        if search_context and search_context.passages:
                            logger.info(f"🔍 Retrieved {len(search_context.passages)} passages with reranking")
                        else:
                            logger.warning("🔍 No search context retrieved")
                    except Exception as search_error:
                        logger.warning(f"Search failed, continuing without context: {search_error}")

                # Solve the question with search context
                solver_response = solver_agent.solve(
                    question=formatted_question,
                    routing_decision=routing_decision,
                    context=search_context
                )

                # Extract final answer
                predicted_answer = extract_final_answer(solver_response.answer, routing_decision.task_type)
                if not predicted_answer:
                    predicted_answer = extract_final_answer(solver_response.raw_output, routing_decision.task_type)

                # Judge the answer
                judge_result = judge_agent.judge_single(
                    prediction=predicted_answer,
                    gold_answer=gold_answer,
                    task_type=routing_decision.task_type,
                    question_id=question_id
                )

                elapsed_time = time.time() - start_time

                # Check correctness using exact match
                is_correct = exact_match_score(predicted_answer, gold_answer)
                if is_correct:
                    correct_predictions += 1

                logger.info(f"🤖 Predicted: {predicted_answer}")
                logger.info(f"🎯 Gold: {gold_answer}")
                logger.info(f"{'✅' if is_correct else '❌'} Correct: {is_correct}")
                logger.info(f"⏱️  Time: {elapsed_time:.2f}s")

                # Store result
                result = {
                    "id": question_id,
                    "question": formatted_question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "raw_output": solver_response.raw_output,
                    "reasoning": solver_response.reasoning,
                    "is_correct": is_correct,
                    "confidence": solver_response.confidence,
                    "elapsed_time": elapsed_time,
                    "routing_decision": {
                        "task_type": routing_decision.task_type,
                        "use_search": routing_decision.use_search,
                        "prompt_style": routing_decision.prompt_style,
                        "decoding_strategy": routing_decision.decoding_strategy,
                        "temperature": routing_decision.temperature
                    },
                    "judge_result": {
                        "is_correct": judge_result.correct,
                        "score": judge_result.score,
                        "explanation": judge_result.explanation
                    },
                    "search_context": {
                        "used": search_context is not None,
                        "num_passages": len(search_context.passages) if search_context else 0,
                        "search_time": search_context.search_time if search_context else 0.0,
                        "reranked": search_context.reranked if search_context else False,
                        "passages": [
                            {
                                "title": p.title,
                                "url": p.url,
                                "score": p.score,
                                "summary": p.content[:200] + "..." if len(p.content) > 200 else p.content
                            } for p in search_context.passages
                        ] if search_context and search_context.passages else []
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
                    "question": formatted_question,
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
        logger.info(f"\n🎉 FINAL HOTPOTQA RESULTS:")
        logger.info(f"📊 Exact Match Accuracy: {final_accuracy:.3f} ({correct_predictions}/{total_questions})")

        # Save results
        timestamp = int(time.time())
        results_file = f"hotpotqa_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "dataset": "hotpotqa",
                    "sample_size": len(validation_data),
                    "total_questions": total_questions,
                    "correct_predictions": correct_predictions,
                    "accuracy": final_accuracy,
                    "timestamp": timestamp,
                    "method": "tavily_search_with_bge_reranking",
                    "search_enabled": search_agent is not None
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
        if search_agent:
            try:
                # Clear search models
                search_agent._cleanup_models()
            except:
                pass
        if model_manager:
            model_manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    run_hotpotqa_evaluation(sample_size=50)  # Run with 50 samples
