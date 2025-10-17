#!/usr/bin/env python3
"""
Test CodeCarbon tracking with a small evaluation to verify it works properly
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
logger = get_logger("test_carbon_tracking")

def test_carbon_tracking_small():
    """Test carbon tracking with just 3 questions to verify it works"""
    logger.info("=== TESTING CARBON TRACKING WITH SMALL DATASET ===")

    # Initialize CodeCarbon emissions tracker with simple configuration
    timestamp = int(time.time())
    emissions_output_dir = "test_emissions_logs"
    os.makedirs(emissions_output_dir, exist_ok=True)

    # Use minimal configuration to avoid API issues
    tracker = EmissionsTracker(
        project_name="A2A_Pipeline_Test",
        output_dir=emissions_output_dir,
        output_file=f"test_emissions_{timestamp}.csv",
        save_to_file=True,
        save_to_api=False
    )

    try:
        # Start emissions tracking
        logger.info("🌱 Starting CodeCarbon emissions tracking...")
        logger.info(f"📊 Emissions will be logged to: {emissions_output_dir}/test_emissions_{timestamp}.csv")
        tracker.start()

        # Initialize pipeline
        logger.info("Initializing A2A Pipeline...")
        pipeline = A2APipeline(use_vllm_server=False)

        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return

        # Load just 3 examples from GSM8K-Platinum
        logger.info("Loading small GSM8K-Platinum test dataset...")
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        small_dataset = dataset.select(range(3))  # Just 3 questions
        logger.info(f"Testing with {len(small_dataset)} examples")

        results = []
        correct = 0
        start_time = time.time()

        for i, example in enumerate(small_dataset):
            question = example["question"]
            gold_answer_full = example["answer"]
            # Extract final number after ####
            gold_answer = gold_answer_full.split("####")[-1].strip() if "####" in gold_answer_full else gold_answer_full.strip()

            logger.info(f"Question {i+1}/{len(small_dataset)}: {question[:100]}...")
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

        total_time = time.time() - start_time
        accuracy = correct / len(small_dataset)

        # Stop emissions tracking and get final results
        logger.info("🌱 Stopping CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\\n=== TEST RESULTS ===")
        logger.info(f"Total questions: {len(small_dataset)}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"🌱 TOTAL CO2 EMISSIONS: {final_emissions:.6f} kg CO2eq")

        # Check if CSV file was actually created
        csv_file = f"{emissions_output_dir}/test_emissions_{timestamp}.csv"
        if os.path.exists(csv_file):
            logger.info(f"✅ SUCCESS: Emissions CSV file created at {csv_file}")
            with open(csv_file, 'r') as f:
                csv_content = f.read()
                logger.info(f"CSV file size: {len(csv_content)} bytes")
                logger.info(f"CSV preview: {csv_content[:200]}...")
        else:
            logger.error(f"❌ FAILED: No CSV file found at {csv_file}")

        # Save test results
        test_results = {
            "test_type": "Small Carbon Tracking Test",
            "dataset": "GSM8K-Platinum (3 examples)",
            "num_examples": len(small_dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_seconds": total_time,
            "carbon_emissions_kg_co2": final_emissions,
            "csv_file_created": os.path.exists(csv_file),
            "csv_file_path": csv_file,
            "detailed_results": results
        }

        results_file = f"test_carbon_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"📁 Test results saved to: {results_file}")

        pipeline.cleanup()
        return test_results

    except Exception as e:
        logger.error(f"Test failed: {e}")
        try:
            # Still try to stop tracker
            final_emissions = tracker.stop()
            logger.info(f"🌱 Partial emissions before failure: {final_emissions:.6f} kg CO2eq")
        except:
            final_emissions = 0.0
        return {"error": str(e), "partial_emissions": final_emissions}

if __name__ == "__main__":
    print("🧪 Testing CodeCarbon tracking with small dataset")
    print("=" * 60)

    results = test_carbon_tracking_small()

    if "error" not in results:
        print(f"\\n✅ TEST PASSED!")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"CO2 Emissions: {results['carbon_emissions_kg_co2']:.6f} kg")
        print(f"CSV Created: {results['csv_file_created']}")
    else:
        print(f"\\n❌ TEST FAILED: {results['error']}")