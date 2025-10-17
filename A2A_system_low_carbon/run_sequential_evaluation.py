#!/usr/bin/env python3
"""
Sequential Evaluation Workflow
Runs evaluations one at a time to manage VRAM usage efficiently
"""

import os
import sys
import json
import time
import subprocess
import gc
import torch
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

# Import CodeCarbon for emissions tracking
from codecarbon import EmissionsTracker

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("sequential_evaluation")

def clear_gpu_memory():
    """Clear GPU memory between evaluations"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(2)  # Give time for cleanup

def run_hotpotqa_evaluation_direct(pipeline: A2APipeline, num_examples=100):
    """Run HotpotQA evaluation directly (inline)"""
    logger.info("=== HOTPOTQA EVALUATION (SEQUENTIAL) ===")

    # Initialize CodeCarbon emissions tracker
    timestamp = int(time.time())
    emissions_output_dir = "sequential_hotpotqa_emissions"
    os.makedirs(emissions_output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="A2A_Sequential_HotpotQA",
        output_dir=emissions_output_dir,
        output_file=f"sequential_hotpotqa_{timestamp}.csv",
        save_to_file=True,
        save_to_api=False
    )

    try:
        # Start emissions tracking
        logger.info("🌱 Starting CodeCarbon emissions tracking...")
        tracker.start()

        # Load HotpotQA validation dataset
        logger.info("Loading HotpotQA validation dataset...")
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        logger.info(f"Loaded HotpotQA validation set: {len(dataset)} total examples")

        eval_dataset = dataset.select(range(num_examples))
        logger.info(f"Evaluating on {len(eval_dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        def normalize_answer(answer):
            """Normalize answer for comparison"""
            if not answer:
                return ""
            answer = str(answer).lower().strip()
            if answer in ["yes", "true", "1"]:
                return "yes"
            elif answer in ["no", "false", "0"]:
                return "no"
            return answer

        for i, example in enumerate(eval_dataset):
            question = example["question"]
            gold_answer = example["answer"]
            question_type = example["type"]
            level = example["level"]

            logger.info(f"Question {i+1}/{len(eval_dataset)}: {question}")
            logger.info(f"Type: {question_type}, Level: {level}")
            logger.info(f"Expected answer: {gold_answer}")

            question_start_time = time.time()
            result = pipeline.solve_question(question, dataset="multihop")
            solve_time = time.time() - question_start_time

            if result and result.get("answer") and "error" not in result:
                predicted_answer = result["answer"]
                pred_normalized = normalize_answer(predicted_answer)
                gold_normalized = normalize_answer(gold_answer)
                is_correct = pred_normalized == gold_normalized

                logger.info(f"Predicted: {predicted_answer}")
                logger.info(f"Correct: {is_correct} ({solve_time:.2f}s)")

                if is_correct:
                    correct += 1
            else:
                logger.error(f"Failed: {result.get('error', 'Unknown error')}")
                is_correct = False
                predicted_answer = result.get("error", "ERROR")

            results.append({
                "id": example["id"],
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": is_correct,
                "type": question_type,
                "level": level,
                "solve_time": solve_time
            })

            # Progress update every 25 questions
            if (i + 1) % 25 == 0:
                current_accuracy = correct / (i + 1)
                elapsed_time = time.time() - start_time
                logger.info(f"Progress: {i+1}/{len(eval_dataset)} ({current_accuracy:.3f} accuracy)")

        total_time = time.time() - start_time
        accuracy = correct / len(eval_dataset)

        # Stop emissions tracking
        logger.info("🌱 Stopping CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\n=== HOTPOTQA RESULTS ===")
        logger.info(f"Total questions: {len(eval_dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"🌱 TOTAL CO2 EMISSIONS: {final_emissions:.6f} kg CO2eq")

        # Check CSV file
        csv_file = f"{emissions_output_dir}/sequential_hotpotqa_{timestamp}.csv"
        csv_created = os.path.exists(csv_file)

        if csv_created:
            logger.info(f"✅ Emissions CSV created: {csv_file}")
        else:
            logger.error(f"❌ No emissions CSV found")

        results_data = {
            "dataset": "HotpotQA-Sequential",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_examples": len(eval_dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_minutes": total_time / 60,
            "carbon_footprint": {
                "total_emissions_kg_co2": final_emissions,
                "emissions_log_file": csv_file,
                "csv_file_created": csv_created
            },
            "detailed_results": results
        }

        # Save results
        results_file = f"sequential_hotpotqa_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"📁 Results saved to: {results_file}")
        return results_data

    except Exception as e:
        logger.error(f"HotpotQA evaluation failed: {e}")
        try:
            final_emissions = tracker.stop()
        except:
            final_emissions = 0.0
        return {"error": str(e), "partial_emissions": final_emissions}

def run_gsm8k_sample_evaluation_direct(pipeline: A2APipeline, num_examples=50):
    """Run GSM8K sample evaluation directly (inline)"""
    logger.info("=== GSM8K SAMPLE EVALUATION (SEQUENTIAL) ===")

    # Initialize CodeCarbon emissions tracker
    timestamp = int(time.time())
    emissions_output_dir = "sequential_gsm8k_emissions"
    os.makedirs(emissions_output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="A2A_Sequential_GSM8K",
        output_dir=emissions_output_dir,
        output_file=f"sequential_gsm8k_{timestamp}.csv",
        save_to_file=True,
        save_to_api=False
    )

    try:
        # Start emissions tracking
        logger.info("🌱 Starting CodeCarbon emissions tracking...")
        tracker.start()

        # Load GSM8K-Platinum dataset
        logger.info("Loading GSM8K-Platinum test dataset...")
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        logger.info(f"Loaded GSM8K-Platinum test set: {len(dataset)} total examples")

        eval_dataset = dataset.select(range(num_examples))
        logger.info(f"Evaluating on {len(eval_dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        for i, example in enumerate(eval_dataset):
            question = example["question"]
            gold_answer_full = example["answer"]
            gold_answer = gold_answer_full.split("####")[-1].strip() if "####" in gold_answer_full else gold_answer_full.strip()

            logger.info(f"Question {i+1}/{len(eval_dataset)}: {question[:100]}...")
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

            # Progress update every 25 questions
            if (i + 1) % 25 == 0:
                current_accuracy = correct / (i + 1)
                elapsed_time = time.time() - start_time
                logger.info(f"Progress: {i+1}/{len(eval_dataset)} ({current_accuracy:.3f} accuracy)")

        total_time = time.time() - start_time
        accuracy = correct / len(eval_dataset)

        # Stop emissions tracking
        logger.info("🌱 Stopping CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\n=== GSM8K SAMPLE RESULTS ===")
        logger.info(f"Total questions: {len(eval_dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"🌱 TOTAL CO2 EMISSIONS: {final_emissions:.6f} kg CO2eq")

        # Check CSV file
        csv_file = f"{emissions_output_dir}/sequential_gsm8k_{timestamp}.csv"
        csv_created = os.path.exists(csv_file)

        if csv_created:
            logger.info(f"✅ Emissions CSV created: {csv_file}")
        else:
            logger.error(f"❌ No emissions CSV found")

        results_data = {
            "dataset": "GSM8K-Platinum-Sample-Sequential",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_examples": len(eval_dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_minutes": total_time / 60,
            "carbon_footprint": {
                "total_emissions_kg_co2": final_emissions,
                "emissions_log_file": csv_file,
                "csv_file_created": csv_created
            },
            "detailed_results": results
        }

        # Save results
        results_file = f"sequential_gsm8k_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"📁 Results saved to: {results_file}")
        return results_data

    except Exception as e:
        logger.error(f"GSM8K evaluation failed: {e}")
        try:
            final_emissions = tracker.stop()
        except:
            final_emissions = 0.0
        return {"error": str(e), "partial_emissions": final_emissions}

def run_sequential_evaluation():
    """Run sequential evaluation workflow"""
    logger.info("🔬 SEQUENTIAL EVALUATION WORKFLOW - A2A Pipeline")
    logger.info("Running evaluations one at a time to manage VRAM efficiently")
    logger.info("=" * 80)

    all_results = {}

    try:
        # Step 1: Initialize pipeline once
        logger.info("Step 1: Initializing A2A Pipeline...")
        pipeline = A2APipeline(use_vllm_server=False)

        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return

        # Step 2: Run HotpotQA evaluation (100 examples)
        logger.info("\nStep 2: Running HotpotQA evaluation...")
        hotpotqa_results = run_hotpotqa_evaluation_direct(pipeline, num_examples=100)
        all_results["hotpotqa"] = hotpotqa_results

        # Clear memory between evaluations
        logger.info("Clearing GPU memory...")
        clear_gpu_memory()

        # Step 3: Run GSM8K sample evaluation (50 examples)
        logger.info("\nStep 3: Running GSM8K sample evaluation...")
        gsm8k_results = run_gsm8k_sample_evaluation_direct(pipeline, num_examples=50)
        all_results["gsm8k_sample"] = gsm8k_results

        # Cleanup pipeline
        logger.info("Cleaning up pipeline...")
        pipeline.cleanup()
        clear_gpu_memory()

        # Save combined results
        timestamp = int(time.time())
        combined_results = {
            "evaluation_type": "Sequential Multi-Dataset Evaluation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluations": all_results,
            "summary": {
                "hotpotqa_accuracy": all_results["hotpotqa"].get("accuracy", 0),
                "gsm8k_accuracy": all_results["gsm8k_sample"].get("accuracy", 0),
                "total_emissions": (
                    all_results["hotpotqa"].get("carbon_footprint", {}).get("total_emissions_kg_co2", 0) +
                    all_results["gsm8k_sample"].get("carbon_footprint", {}).get("total_emissions_kg_co2", 0)
                )
            }
        }

        combined_file = f"sequential_combined_results_{timestamp}.json"
        with open(combined_file, "w") as f:
            json.dump(combined_results, f, indent=2)

        logger.info(f"\n📁 Combined results saved to: {combined_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 SEQUENTIAL EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"HotpotQA Accuracy: {combined_results['summary']['hotpotqa_accuracy']:.3f}")
        logger.info(f"GSM8K Accuracy: {combined_results['summary']['gsm8k_accuracy']:.3f}")
        logger.info(f"🌱 Total CO2 Emissions: {combined_results['summary']['total_emissions']:.6f} kg CO2eq")
        logger.info("=" * 80)

        return combined_results

    except Exception as e:
        logger.error(f"Sequential evaluation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    results = run_sequential_evaluation()