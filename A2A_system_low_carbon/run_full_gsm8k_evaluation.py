#!/usr/bin/env python3
"""
Complete GSM8K-Platinum Evaluation for A2A Pipeline
Evaluates on the FULL GSM8K-Platinum test dataset (all 1,209 examples)
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
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("full_gsm8k_evaluation")

def evaluate_full_gsm8k_platinum(pipeline: A2APipeline):
    """Evaluate on the COMPLETE GSM8K-Platinum test dataset"""
    logger.info("=== COMPLETE GSM8K-PLATINUM EVALUATION ===")

    try:
        # Load the FULL GSM8K-Platinum test dataset
        logger.info("Loading complete GSM8K-Platinum test dataset...")
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        logger.info(f"Loaded GSM8K-Platinum test set: {len(dataset)} total examples")

        # Use the COMPLETE dataset (all 1,209 examples)
        logger.info(f"Evaluating on FULL dataset: {len(dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        for i, example in enumerate(dataset):
            question = example["question"]
            # GSM8K answer format is: "Step 1...\\nStep 2...\\n#### 42"
            gold_answer_full = example["answer"]
            # Extract final number after ####
            gold_answer = gold_answer_full.split("####")[-1].strip() if "####" in gold_answer_full else gold_answer_full.strip()

            logger.info(f"Question {i+1}/{len(dataset)}: {question[:100]}...")
            logger.info(f"Expected answer: {gold_answer}")

            question_start_time = time.time()
            result = pipeline.solve_question(question, dataset="gsm8k")
            solve_time = time.time() - question_start_time

            if result and result.get("answer") and "error" not in result:
                predicted_answer = result["answer"]

                # Use math verification
                from utils.math_verify import extract_and_verify_answer
                verification = extract_and_verify_answer(predicted_answer, gold_answer)
                is_correct = verification["correct"]

                logger.info(f"Predicted: {predicted_answer}")
                logger.info(f"Correct: {is_correct} ({solve_time:.2f}s)")

                if is_correct:
                    correct += 1

            else:
                logger.error(f"Failed: {result.get('error', 'Unknown error')}")
                is_correct = False
                predicted_answer = result.get("error", "ERROR")

            results.append({
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": is_correct,
                "solve_time": solve_time
            })

            # Log progress every 50 questions
            if (i + 1) % 50 == 0:
                current_accuracy = correct / (i + 1)
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * len(dataset) / (i + 1)
                remaining_time = estimated_total_time - elapsed_time

                logger.info(f"Progress: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
                logger.info(f"Current accuracy: {current_accuracy:.3f} ({correct}/{i+1})")
                logger.info(f"Elapsed: {elapsed_time/3600:.2f}h, Est. remaining: {remaining_time/3600:.2f}h")

        total_time = time.time() - start_time
        accuracy = correct / len(dataset)

        logger.info(f"\n=== COMPLETE GSM8K-PLATINUM RESULTS ===")
        logger.info(f"Total questions: {len(dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per question: {total_time/len(dataset):.2f}s")

        return {
            "dataset": "GSM8K-Platinum-Complete",
            "num_examples": len(dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_hours": total_time / 3600,
            "avg_time_per_question": total_time / len(dataset),
            "detailed_results": results
        }

    except Exception as e:
        logger.error(f"GSM8K-Platinum evaluation failed: {e}")
        return {"error": str(e)}

def run_full_gsm8k_evaluation():
    """Run evaluation on complete GSM8K-Platinum dataset"""
    logger.info("🚀 COMPLETE GSM8K-PLATINUM EVALUATION - A2A Pipeline")
    logger.info("Evaluating on ALL 1,209 examples from GSM8K-Platinum test set")
    logger.info("=" * 80)

    # Initialize pipeline
    logger.info("Initializing A2A Pipeline...")
    pipeline = A2APipeline(use_vllm_server=False)

    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return

    # Run complete evaluation
    results = evaluate_full_gsm8k_platinum(pipeline)

    # Save results
    timestamp = int(time.time())
    results_file = f"full_gsm8k_platinum_evaluation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n📁 Complete results saved to: {results_file}")

    # Summary
    if "error" not in results:
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPLETE GSM8K-PLATINUM EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Dataset: {results['dataset']}")
        logger.info(f"Total Examples: {results['num_examples']}")
        logger.info(f"Correct: {results['correct']}")
        logger.info(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"Total Time: {results['total_time_hours']:.2f} hours")
        logger.info(f"Avg Time/Question: {results['avg_time_per_question']:.2f}s")
        logger.info("=" * 80)
        logger.info("✅ Complete GSM8K-Platinum evaluation finished!")
    else:
        logger.error(f"Evaluation failed: {results['error']}")

    pipeline.cleanup()
    return results

if __name__ == "__main__":
    print("🔬 A2A Pipeline - Complete GSM8K-Platinum Evaluation")
    print("Evaluating on ALL 1,209 examples from GSM8K-Platinum test dataset")
    print("=" * 80)

    results = run_full_gsm8k_evaluation()