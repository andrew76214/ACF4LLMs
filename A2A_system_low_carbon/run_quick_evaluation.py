#!/usr/bin/env python3
"""
Quick Dataset Evaluation for A2A Pipeline
Runs smaller evaluations to demonstrate functionality
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dotenv import load_dotenv
load_dotenv()

import datasets
from datasets import load_dataset
from utils.logging import setup_logger, get_logger
from src.pipeline import A2APipeline

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=False)
logger = get_logger("quick_evaluation")

def evaluate_sample_questions():
    """Quick evaluation on sample questions"""
    logger.info("🚀 Running Quick A2A Pipeline Evaluation")

    # Initialize pipeline
    pipeline = A2APipeline(use_vllm_server=False)
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return

    # Test questions from different domains
    test_questions = [
        # Math questions
        {
            "question": "John has 15 apples. He gives 8 to his friend. How many apples does he have left?",
            "gold_answer": "7",
            "dataset": "gsm8k",
            "expected_correct": True
        },
        {
            "question": "What is 24 + 17?",
            "gold_answer": "41",
            "dataset": "gsm8k",
            "expected_correct": True
        },
        {
            "question": "If a rectangle has length 6 and width 4, what is its area?",
            "gold_answer": "24",
            "dataset": "math",
            "expected_correct": True
        },
        # QA questions (would require search)
        {
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
            "dataset": "qa",
            "expected_correct": True
        },
        {
            "question": "Who wrote the Harry Potter series?",
            "gold_answer": "J.K. Rowling",
            "dataset": "qa",
            "expected_correct": True
        }
    ]

    results = []
    correct = 0

    for i, test_case in enumerate(test_questions):
        question = test_case["question"]
        gold_answer = test_case["gold_answer"]
        dataset = test_case["dataset"]

        logger.info(f"\nQuestion {i+1}/{len(test_questions)}: {question}")
        logger.info(f"Expected answer: {gold_answer}")

        start_time = time.time()
        result = pipeline.solve_question(question, dataset=dataset)
        solve_time = time.time() - start_time

        if result and result.get("answer") and "error" not in result:
            predicted_answer = result["answer"]

            # Check correctness
            if dataset in ["gsm8k", "math"]:
                # Use math verification
                from utils.math_verify import extract_and_verify_answer
                verification = extract_and_verify_answer(predicted_answer, gold_answer)
                is_correct = verification["correct"]
            else:
                # Simple string matching for QA
                is_correct = gold_answer.lower().strip() in predicted_answer.lower().strip()

            logger.info(f"Predicted: {predicted_answer}")
            logger.info(f"Correct: {is_correct} ({solve_time:.2f}s)")

            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": is_correct,
                "solve_time": solve_time,
                "dataset": dataset,
                "used_search": result.get("used_search", False),
                "task_type": result.get("task_type", ""),
                "reasoning": result.get("reasoning", "")[:200] + "..." if result.get("reasoning") else ""
            })
        else:
            logger.error(f"Failed to solve question: {result.get('error', 'Unknown error')}")
            results.append({
                "question": question,
                "predicted_answer": "ERROR",
                "gold_answer": gold_answer,
                "correct": False,
                "solve_time": solve_time,
                "dataset": dataset,
                "error": result.get("error", "Unknown error")
            })

    accuracy = correct / len(test_questions)

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 QUICK EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Questions answered: {len(test_questions)}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"{'='*60}")

    # Show detailed results
    for i, result in enumerate(results):
        status = "✅" if result["correct"] else "❌"
        logger.info(f"{status} Q{i+1} ({result['dataset']}): {result['question'][:50]}...")
        logger.info(f"    → {result['predicted_answer']} (Expected: {result['gold_answer']})")
        if result.get("used_search"):
            logger.info(f"    🔍 Used search")
        if result.get("task_type"):
            logger.info(f"    📋 Task: {result['task_type']}")

    # Save results
    results_data = {
        "timestamp": time.time(),
        "total_questions": len(test_questions),
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }

    with open("quick_evaluation_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\n📁 Detailed results saved to: quick_evaluation_results.json")

    pipeline.cleanup()

    return results_data

if __name__ == "__main__":
    print("🔬 A2A Pipeline - Quick Evaluation Demo")
    print("Testing core functionality on sample questions")
    print("="*60)

    results = evaluate_sample_questions()

    print(f"\n✅ Quick evaluation completed!")
    print(f"   Accuracy: {results['accuracy']:.1%}")
    print(f"   See quick_evaluation_results.json for details")
