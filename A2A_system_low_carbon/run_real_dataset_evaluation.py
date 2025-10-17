#!/usr/bin/env python3
"""
REAL Dataset Evaluation for A2A Pipeline
Evaluates on actual datasets: GSM8K, AIME, HotpotQA, SQuAD
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
logger = get_logger("real_dataset_evaluation")

def evaluate_gsm8k_real(pipeline: A2APipeline, num_examples: int = 20):
    """Evaluate on REAL GSM8K dataset"""
    logger.info(f"=== REAL GSM8K EVALUATION ({num_examples} examples) ===")

    try:
        # Load actual GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        logger.info(f"Loaded real GSM8K test set: {len(dataset)} total examples")

        # Take first N examples
        test_examples = dataset.select(range(min(num_examples, len(dataset))))
        logger.info(f"Evaluating on first {len(test_examples)} examples")

        results = []
        correct = 0

        for i, example in enumerate(test_examples):
            question = example["question"]
            # GSM8K answer format is: "Step 1...\nStep 2...\n#### 42"
            gold_answer_full = example["answer"]
            # Extract final number after ####
            gold_answer = gold_answer_full.split("####")[-1].strip() if "####" in gold_answer_full else gold_answer_full.strip()

            logger.info(f"\nQuestion {i+1}/{len(test_examples)}:")
            logger.info(f"Q: {question}")
            logger.info(f"Expected answer: {gold_answer}")

            start_time = time.time()
            result = pipeline.solve_question(question, dataset="gsm8k")
            solve_time = time.time() - start_time

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

        accuracy = correct / len(test_examples)
        logger.info(f"\n=== GSM8K RESULTS ===")
        logger.info(f"Correct: {correct}/{len(test_examples)} = {accuracy:.3f}")

        return {
            "dataset": "GSM8K",
            "num_examples": len(test_examples),
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

    except Exception as e:
        logger.error(f"GSM8K evaluation failed: {e}")
        return {"error": str(e)}

def evaluate_hotpotqa_real(pipeline: A2APipeline, num_examples: int = 10):
    """Evaluate on REAL HotpotQA dataset"""
    logger.info(f"=== REAL HOTPOTQA EVALUATION ({num_examples} examples) ===")

    try:
        # Load actual HotpotQA dataset
        dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        logger.info(f"Loaded real HotpotQA validation set: {len(dataset)} total examples")

        # Take first N examples
        test_examples = dataset.select(range(min(num_examples, len(dataset))))
        logger.info(f"Evaluating on first {len(test_examples)} examples")

        results = []
        answer_correct = 0

        for i, example in enumerate(test_examples):
            question = example["question"]
            gold_answer = example["answer"]
            gold_supporting_facts = example["supporting_facts"]

            logger.info(f"\nQuestion {i+1}/{len(test_examples)}:")
            logger.info(f"Q: {question}")
            logger.info(f"Expected answer: {gold_answer}")

            start_time = time.time()
            result = pipeline.solve_question(question, dataset="hotpotqa")
            solve_time = time.time() - start_time

            if result and result.get("answer") and "error" not in result:
                predicted_answer = result["answer"]

                # Simple string matching for answer correctness
                is_correct = gold_answer.lower().strip() in predicted_answer.lower().strip()

                logger.info(f"Predicted: {predicted_answer}")
                logger.info(f"Answer correct: {is_correct} ({solve_time:.2f}s)")

                if is_correct:
                    answer_correct += 1

            else:
                logger.error(f"Failed: {result.get('error', 'Unknown error')}")
                is_correct = False
                predicted_answer = result.get("error", "ERROR")

            results.append({
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "answer_correct": is_correct,
                "solve_time": solve_time
            })

        answer_em = answer_correct / len(test_examples)
        logger.info(f"\n=== HOTPOTQA RESULTS ===")
        logger.info(f"Answer EM: {answer_correct}/{len(test_examples)} = {answer_em:.3f}")

        return {
            "dataset": "HotpotQA",
            "num_examples": len(test_examples),
            "answer_correct": answer_correct,
            "answer_em": answer_em,
            "results": results
        }

    except Exception as e:
        logger.error(f"HotpotQA evaluation failed: {e}")
        return {"error": str(e)}

def evaluate_squad_real(pipeline: A2APipeline, num_examples: int = 15):
    """Evaluate on REAL SQuAD dataset"""
    logger.info(f"=== REAL SQUAD EVALUATION ({num_examples} examples) ===")

    try:
        # Load actual SQuAD v1.1 dataset
        dataset = load_dataset("squad", split="validation")
        logger.info(f"Loaded real SQuAD validation set: {len(dataset)} total examples")

        # Take first N examples
        test_examples = dataset.select(range(min(num_examples, len(dataset))))
        logger.info(f"Evaluating on first {len(test_examples)} examples")

        results = []
        correct = 0

        for i, example in enumerate(test_examples):
            question = example["question"]
            context = example["context"]
            # SQuAD has multiple possible answers
            gold_answers = [ans["text"] for ans in example["answers"]]
            primary_answer = gold_answers[0] if gold_answers else ""

            logger.info(f"\nQuestion {i+1}/{len(test_examples)}:")
            logger.info(f"Q: {question}")
            logger.info(f"Expected answers: {gold_answers}")

            start_time = time.time()
            result = pipeline.solve_question(question, dataset="squad", context=context)
            solve_time = time.time() - start_time

            if result and result.get("answer") and "error" not in result:
                predicted_answer = result["answer"]

                # Check if prediction matches any of the gold answers
                is_correct = any(gold_ans.lower().strip() in predicted_answer.lower().strip()
                               for gold_ans in gold_answers)

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
                "gold_answers": gold_answers,
                "correct": is_correct,
                "solve_time": solve_time
            })

        accuracy = correct / len(test_examples)
        logger.info(f"\n=== SQUAD RESULTS ===")
        logger.info(f"Correct: {correct}/{len(test_examples)} = {accuracy:.3f}")

        return {
            "dataset": "SQuAD",
            "num_examples": len(test_examples),
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

    except Exception as e:
        logger.error(f"SQuAD evaluation failed: {e}")
        return {"error": str(e)}

def run_real_evaluations():
    """Run evaluations on real datasets"""
    logger.info("🔬 REAL DATASET EVALUATION - A2A Pipeline")
    logger.info("=" * 60)

    # Initialize pipeline
    logger.info("Initializing A2A Pipeline...")
    pipeline = A2APipeline(use_vllm_server=False)

    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return

    all_results = {}

    # 1. Real GSM8K Evaluation
    logger.info("\n" + "="*50)
    logger.info("1. REAL GSM8K EVALUATION")
    logger.info("="*50)
    gsm8k_results = evaluate_gsm8k_real(pipeline, num_examples=20)
    all_results["gsm8k"] = gsm8k_results

    # 2. Real HotpotQA Evaluation
    logger.info("\n" + "="*50)
    logger.info("2. REAL HOTPOTQA EVALUATION")
    logger.info("="*50)
    hotpot_results = evaluate_hotpotqa_real(pipeline, num_examples=10)
    all_results["hotpotqa"] = hotpot_results

    # 3. Real SQuAD Evaluation
    logger.info("\n" + "="*50)
    logger.info("3. REAL SQUAD EVALUATION")
    logger.info("="*50)
    squad_results = evaluate_squad_real(pipeline, num_examples=15)
    all_results["squad"] = squad_results

    # Save results
    timestamp = int(time.time())
    results_file = f"real_dataset_evaluation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n📁 Results saved to: {results_file}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 REAL DATASET EVALUATION SUMMARY")
    logger.info("="*60)

    for dataset_name, results in all_results.items():
        if "error" in results:
            logger.error(f"{dataset_name.upper()}: FAILED - {results['error']}")
        else:
            if dataset_name == "gsm8k":
                logger.info(f"GSM8K: {results['accuracy']:.3f} accuracy ({results['correct']}/{results['num_examples']})")
            elif dataset_name == "hotpotqa":
                logger.info(f"HotpotQA: {results['answer_em']:.3f} answer EM ({results['answer_correct']}/{results['num_examples']})")
            elif dataset_name == "squad":
                logger.info(f"SQuAD: {results['accuracy']:.3f} accuracy ({results['correct']}/{results['num_examples']})")

    logger.info("="*60)
    logger.info("✅ Real dataset evaluation completed!")

    pipeline.cleanup()
    return all_results

if __name__ == "__main__":
    print("🔬 A2A Pipeline - REAL Dataset Evaluation")
    print("Evaluating on actual GSM8K, HotpotQA, SQuAD datasets")
    print("="*60)

    results = run_real_evaluations()