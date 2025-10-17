#!/usr/bin/env python3
"""
Complete GSM8K-Platinum Evaluation with VERIFIED Carbon Emissions Tracking
Uses the working carbon tracking configuration from the test case
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

# Import CodeCarbon with VERIFIED working configuration
from codecarbon import EmissionsTracker

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("full_gsm8k_with_verified_carbon")

def evaluate_full_gsm8k_platinum_with_verified_carbon():
    """Evaluate on the COMPLETE GSM8K-Platinum dataset using VERIFIED carbon tracking configuration"""
    logger.info("=== COMPLETE GSM8K-PLATINUM EVALUATION WITH VERIFIED CARBON TRACKING ===")

    # Initialize CodeCarbon emissions tracker with VERIFIED configuration from test
    timestamp = int(time.time())
    emissions_output_dir = "verified_emissions_logs"
    os.makedirs(emissions_output_dir, exist_ok=True)

    # Use EXACTLY the same configuration that worked in the test
    tracker = EmissionsTracker(
        project_name="A2A_Pipeline_GSM8K_Full_Verified",
        output_dir=emissions_output_dir,
        output_file=f"gsm8k_full_verified_emissions_{timestamp}.csv",
        save_to_file=True,
        save_to_api=False
    )

    try:
        # Start emissions tracking
        logger.info("🌱 Starting VERIFIED CodeCarbon emissions tracking...")
        logger.info(f"📊 Emissions will be logged to: {emissions_output_dir}/gsm8k_full_verified_emissions_{timestamp}.csv")
        tracker.start()

        # Initialize pipeline
        logger.info("Initializing A2A Pipeline...")
        pipeline = A2APipeline(use_vllm_server=False)

        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return

        # Load the COMPLETE GSM8K-Platinum test dataset
        logger.info("Loading complete GSM8K-Platinum test dataset...")
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        logger.info(f"Loaded GSM8K-Platinum test set: {len(dataset)} total examples")

        # Use the COMPLETE dataset (all 1,209 examples)
        logger.info(f"Evaluating on FULL dataset: {len(dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        # Track carbon emissions for progress monitoring
        carbon_checkpoints = []

        for i, example in enumerate(dataset):
            question = example["question"]
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

                carbon_checkpoints.append({
                    "questions_completed": i + 1,
                    "accuracy": current_accuracy,
                    "elapsed_time_hours": elapsed_time / 3600,
                    "estimated_total_time_hours": estimated_total_time / 3600
                })

                logger.info(f"Progress: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
                logger.info(f"Current accuracy: {current_accuracy:.3f} ({correct}/{i+1})")
                logger.info(f"Elapsed: {elapsed_time/3600:.2f}h, Est. remaining: {remaining_time/3600:.2f}h")

        total_time = time.time() - start_time
        accuracy = correct / len(dataset)

        # Stop emissions tracking and get final results
        logger.info("🌱 Stopping VERIFIED CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\n=== COMPLETE GSM8K-PLATINUM RESULTS WITH VERIFIED CARBON FOOTPRINT ===")
        logger.info(f"Total questions: {len(dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per question: {total_time/len(dataset):.2f}s")
        logger.info(f"🌱 TOTAL CO2 EMISSIONS: {final_emissions:.6f} kg CO2eq")
        logger.info(f"🌱 CO2 per question: {final_emissions/len(dataset):.8f} kg CO2eq")
        logger.info(f"🌱 CO2 per correct answer: {final_emissions/max(correct, 1):.8f} kg CO2eq")

        # Check if CSV file was actually created
        csv_file = f"{emissions_output_dir}/gsm8k_full_verified_emissions_{timestamp}.csv"
        csv_created = os.path.exists(csv_file)
        csv_size = os.path.getsize(csv_file) if csv_created else 0

        if csv_created:
            logger.info(f"✅ SUCCESS: Emissions CSV file created at {csv_file} ({csv_size} bytes)")
        else:
            logger.error(f"❌ FAILED: No CSV file found at {csv_file}")

        results_data = {
            "dataset": "GSM8K-Platinum-Complete-With-VERIFIED-Carbon",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
                "emissions_log_file": csv_file,
                "csv_file_created": csv_created,
                "csv_file_size_bytes": csv_size,
                "tracking_verification": "VERIFIED working configuration from test_carbon_tracking.py"
            },
            "system_configuration": {
                "model": "unsloth/Qwen3-4B-Thinking-2507",
                "quantization": "fp8",
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.8,
                "self_consistency_samples": 10,
                "reasoning_enabled": True,
                "search_api": "Tavily",
                "embeddings": "BAAI/bge-m3",
                "reranker": "BAAI/bge-reranker-v2-m3",
                "math_verification": "Math-Verify (custom implementation)"
            },
            "detailed_results": results
        }

        # Save results
        results_file = f"gsm8k_platinum_complete_verified_carbon_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\n📁 Complete results with VERIFIED carbon footprint saved to: {results_file}")

        pipeline.cleanup()
        return results_data

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

if __name__ == "__main__":
    print("🔬 A2A Pipeline - Complete GSM8K-Platinum Evaluation WITH VERIFIED CARBON TRACKING")
    print("Evaluating on ALL 1,209 examples from GSM8K-Platinum test dataset")
    print("🌱 Using VERIFIED carbon emissions configuration from successful test")
    print("=" * 80)

    results = evaluate_full_gsm8k_platinum_with_verified_carbon()

    if "error" not in results:
        print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"Total Time: {results['total_time_hours']:.2f} hours")
        print(f"🌱 Total CO2 Emissions: {results['carbon_footprint']['total_emissions_kg_co2']:.6f} kg CO2eq")
        print(f"🌱 CSV File Created: {results['carbon_footprint']['csv_file_created']}")
        print(f"🌱 CSV File Size: {results['carbon_footprint']['csv_file_size_bytes']} bytes")
    else:
        print(f"\n❌ EVALUATION FAILED: {results['error']}")
        if 'partial_carbon_footprint' in results:
            print(f"🌱 Partial emissions: {results['partial_carbon_footprint']['emissions_kg_co2']:.6f} kg CO2eq")