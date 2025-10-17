#!/usr/bin/env python3
"""
Complete GSM8K-Platinum Evaluation for A2A Pipeline WITH CARBON EMISSIONS TRACKING
Evaluates on the FULL GSM8K-Platinum test dataset (all 1,209 examples) with CodeCarbon monitoring
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

# Import CodeCarbon for emissions tracking
from codecarbon import EmissionsTracker

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("full_gsm8k_evaluation_with_carbon")

def evaluate_full_gsm8k_platinum_with_carbon(pipeline: A2APipeline):
    """Evaluate on the COMPLETE GSM8K-Platinum test dataset with carbon emissions tracking"""
    logger.info("=== COMPLETE GSM8K-PLATINUM EVALUATION WITH CARBON TRACKING ===")

    # Initialize CodeCarbon emissions tracker
    timestamp = int(time.time())
    emissions_output_dir = "emissions_logs"
    os.makedirs(emissions_output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="A2A_Pipeline_GSM8K_Platinum_Full_Evaluation",
        experiment_id=f"gsm8k_full_eval_{timestamp}",
        output_dir=emissions_output_dir,
        output_file=f"gsm8k_full_emissions_{timestamp}.csv",
        log_level="INFO",
        gpu_ids=[0],  # Track GPU 0
        measure_power_secs=30,  # Measure every 30 seconds
        save_to_file=True,
        save_to_api=False,
        save_to_logger=True,
        logging_logger=logger
    )

    try:
        # Start emissions tracking
        logger.info("🌱 Starting CodeCarbon emissions tracking...")
        logger.info(f"📊 Emissions will be logged to: {emissions_output_dir}/gsm8k_full_emissions_{timestamp}.csv")
        tracker.start()

        # Load the FULL GSM8K-Platinum test dataset
        logger.info("Loading complete GSM8K-Platinum test dataset...")
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        logger.info(f"Loaded GSM8K-Platinum test set: {len(dataset)} total examples")

        # Use the COMPLETE dataset (all 1,209 examples)
        logger.info(f"Evaluating on FULL dataset: {len(dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        # Track carbon emissions for each batch of questions
        carbon_checkpoints = []

        for i, example in enumerate(dataset):
            question = example["question"]
            # GSM8K answer format is: "Step 1...\nStep 2...\n#### 42"
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

            # Log carbon emissions progress every 50 questions
            if (i + 1) % 50 == 0:
                current_accuracy = correct / (i + 1)
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * len(dataset) / (i + 1)
                remaining_time = estimated_total_time - elapsed_time

                # Get current emissions estimate
                current_emissions = getattr(tracker, '_total_energy', 0.0)

                carbon_checkpoints.append({
                    "questions_completed": i + 1,
                    "accuracy": current_accuracy,
                    "elapsed_time_hours": elapsed_time / 3600,
                    "estimated_total_time_hours": estimated_total_time / 3600,
                    "emissions_kg_co2": current_emissions
                })

                logger.info(f"Progress: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
                logger.info(f"Current accuracy: {current_accuracy:.3f} ({correct}/{i+1})")
                logger.info(f"Elapsed: {elapsed_time/3600:.2f}h, Est. remaining: {remaining_time/3600:.2f}h")
                logger.info(f"🌱 Current CO2 emissions: {current_emissions:.6f} kg CO2eq")

        total_time = time.time() - start_time
        accuracy = correct / len(dataset)

        # Stop emissions tracking and get final results
        logger.info("🌱 Stopping CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\n=== COMPLETE GSM8K-PLATINUM RESULTS WITH CARBON FOOTPRINT ===")
        logger.info(f"Total questions: {len(dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per question: {total_time/len(dataset):.2f}s")
        logger.info(f"🌱 TOTAL CO2 EMISSIONS: {final_emissions:.6f} kg CO2eq")
        logger.info(f"🌱 CO2 per question: {final_emissions/len(dataset):.8f} kg CO2eq")
        logger.info(f"🌱 CO2 per correct answer: {final_emissions/max(correct, 1):.8f} kg CO2eq")

        return {
            "dataset": "GSM8K-Platinum-Complete-With-Carbon",
            "num_examples": len(dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_hours": total_time / 3600,
            "avg_time_per_question": total_time / len(dataset),
            "carbon_footprint": {
                "total_emissions_kg_co2": final_emissions,
                "emissions_per_question_kg_co2": final_emissions / len(dataset),
                "emissions_per_correct_answer_kg_co2": final_emissions / max(correct, 1),
                "carbon_checkpoints": carbon_checkpoints,
                "emissions_log_file": f"{emissions_output_dir}/gsm8k_full_emissions_{timestamp}.csv"
            },
            "detailed_results": results
        }

    except Exception as e:
        logger.error(f"GSM8K-Platinum evaluation failed: {e}")
        try:
            # Still try to stop tracker and get partial emissions
            final_emissions = tracker.stop()
            logger.info(f"🌱 Partial emissions before failure: {final_emissions:.6f} kg CO2eq")
        except:
            final_emissions = 0.0

        return {
            "error": str(e),
            "partial_carbon_footprint": {
                "emissions_kg_co2": final_emissions
            }
        }

def run_full_gsm8k_evaluation_with_carbon():
    """Run evaluation on complete GSM8K-Platinum dataset with carbon emissions tracking"""
    logger.info("🚀 COMPLETE GSM8K-PLATINUM EVALUATION WITH CARBON TRACKING - A2A Pipeline")
    logger.info("Evaluating on ALL 1,209 examples from GSM8K-Platinum test set")
    logger.info("🌱 Including comprehensive carbon emissions monitoring with CodeCarbon")
    logger.info("=" * 80)

    # Initialize pipeline
    logger.info("Initializing A2A Pipeline...")
    pipeline = A2APipeline(use_vllm_server=False)

    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return

    # Run complete evaluation with carbon tracking
    results = evaluate_full_gsm8k_platinum_with_carbon(pipeline)

    # Save results with carbon footprint data
    timestamp = int(time.time())
    results_file = f"full_gsm8k_platinum_evaluation_with_carbon_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n📁 Complete results with carbon footprint saved to: {results_file}")

    # Summary with carbon footprint
    if "error" not in results:
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPLETE GSM8K-PLATINUM EVALUATION SUMMARY WITH CARBON FOOTPRINT")
        logger.info("=" * 80)
        logger.info(f"Dataset: {results['dataset']}")
        logger.info(f"Total Examples: {results['num_examples']}")
        logger.info(f"Correct: {results['correct']}")
        logger.info(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"Total Time: {results['total_time_hours']:.2f} hours")
        logger.info(f"Avg Time/Question: {results['avg_time_per_question']:.2f}s")
        logger.info("🌱 CARBON FOOTPRINT:")
        logger.info(f"🌱   Total CO2 Emissions: {results['carbon_footprint']['total_emissions_kg_co2']:.6f} kg CO2eq")
        logger.info(f"🌱   CO2 per Question: {results['carbon_footprint']['emissions_per_question_kg_co2']:.8f} kg CO2eq")
        logger.info(f"🌱   CO2 per Correct Answer: {results['carbon_footprint']['emissions_per_correct_answer_kg_co2']:.8f} kg CO2eq")
        logger.info(f"🌱   Emissions Log: {results['carbon_footprint']['emissions_log_file']}")
        logger.info("=" * 80)
        logger.info("✅ Complete GSM8K-Platinum evaluation with carbon tracking finished!")
    else:
        logger.error(f"Evaluation failed: {results['error']}")
        if 'partial_carbon_footprint' in results:
            logger.info(f"🌱 Partial emissions: {results['partial_carbon_footprint']['emissions_kg_co2']:.6f} kg CO2eq")

    pipeline.cleanup()
    return results

if __name__ == "__main__":
    print("🔬 A2A Pipeline - Complete GSM8K-Platinum Evaluation WITH CARBON TRACKING")
    print("Evaluating on ALL 1,209 examples from GSM8K-Platinum test dataset")
    print("🌱 Including comprehensive carbon emissions monitoring with CodeCarbon")
    print("=" * 80)

    results = run_full_gsm8k_evaluation_with_carbon()