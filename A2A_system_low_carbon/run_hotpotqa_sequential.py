#!/usr/bin/env python3
"""
Memory-Efficient HotpotQA Evaluation with Sequential Agent Loading
Uses verified carbon tracking configuration
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
from sequential_pipeline import SequentialA2APipeline

# Import CodeCarbon for emissions tracking (using verified configuration)
from codecarbon import EmissionsTracker

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=True)
logger = get_logger("hotpotqa_sequential")

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

def run_hotpotqa_sequential_evaluation(num_examples=100):
    """Run HotpotQA evaluation using sequential memory-efficient pipeline"""
    logger.info("=== HOTPOTQA SEQUENTIAL MEMORY-EFFICIENT EVALUATION ===")

    # Initialize CodeCarbon emissions tracker (using verified configuration)
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
        logger.info(f"📊 Emissions will be logged to: {emissions_output_dir}/sequential_hotpotqa_{timestamp}.csv")
        tracker.start()

        # Initialize sequential pipeline (memory-efficient)
        logger.info("Initializing Sequential A2A Pipeline...")
        with SequentialA2APipeline(use_vllm_server=False) as pipeline:

            if not pipeline.initialize():
                logger.error("Failed to initialize sequential pipeline")
                return

            # Show initial memory status
            initial_memory = pipeline.get_memory_status()
            logger.info(f"Initial memory status: {initial_memory}")

            # Load HotpotQA validation dataset
            logger.info("Loading HotpotQA validation dataset...")
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            logger.info(f"Loaded HotpotQA validation set: {len(dataset)} total examples")

            # Use specified number of examples
            eval_dataset = dataset.select(range(num_examples))
            logger.info(f"Evaluating on {len(eval_dataset)} examples")

            results = []
            correct = 0
            start_time = time.time()

            for i, example in enumerate(eval_dataset):
                question = example["question"]
                gold_answer = example["answer"]
                question_type = example["type"]
                level = example["level"]

                logger.info(f"Question {i+1}/{len(eval_dataset)}: {question}")
                logger.info(f"Type: {question_type}, Level: {level}")
                logger.info(f"Expected answer: {gold_answer}")

                # Show memory status every 10 questions
                if i % 10 == 0:
                    mem_status = pipeline.get_memory_status()
                    logger.info(f"Memory status at question {i+1}: {mem_status}")

                question_start_time = time.time()
                result = pipeline.solve_question(question, dataset="hotpotqa")
                solve_time = time.time() - question_start_time

                if result and result.get("answer") and "error" not in result:
                    predicted_answer = result["answer"]
                    pred_normalized = normalize_answer(predicted_answer)
                    gold_normalized = normalize_answer(gold_answer)
                    is_correct = pred_normalized == gold_normalized

                    logger.info(f"Predicted: {predicted_answer}")
                    logger.info(f"Correct: {is_correct} ({solve_time:.2f}s)")
                    logger.info(f"Agents loaded: {result.get('loaded_agents', [])}")

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
                    "solve_time": solve_time,
                    "loaded_agents": result.get("loaded_agents", []),
                    "memory_usage": result.get("memory_usage", {}),
                    "retrieval_used": result.get("retrieval_used", False),
                    "passages_retrieved": result.get("passages_retrieved", 0)
                })

                # Progress update every 25 questions
                if (i + 1) % 25 == 0:
                    current_accuracy = correct / (i + 1)
                    elapsed_time = time.time() - start_time
                    logger.info(f"Progress: {i+1}/{len(eval_dataset)} ({current_accuracy:.3f} accuracy)")
                    logger.info(f"Elapsed time: {elapsed_time/60:.2f} minutes")

        total_time = time.time() - start_time
        accuracy = correct / len(eval_dataset)

        # Stop emissions tracking
        logger.info("🌱 Stopping CodeCarbon emissions tracking...")
        final_emissions = tracker.stop()

        logger.info(f"\\n=== SEQUENTIAL HOTPOTQA RESULTS ===")
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
            "dataset": "HotpotQA-Sequential-Memory-Efficient",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_examples": len(eval_dataset),
            "correct": correct,
            "accuracy": accuracy,
            "total_time_minutes": total_time / 60,
            "avg_time_per_question": total_time / len(eval_dataset),
            "sequential_evaluation": True,
            "memory_efficient": True,
            "carbon_footprint": {
                "total_emissions_kg_co2": final_emissions,
                "emissions_per_question_kg_co2": final_emissions / len(eval_dataset),
                "emissions_log_file": csv_file,
                "csv_file_created": csv_created,
                "tracking_verification": "VERIFIED working configuration"
            },
            "system_configuration": {
                "model": "unsloth/Qwen3-4B-Thinking-2507",
                "quantization": "fp8",
                "search_api": "Tavily",
                "embeddings": "BAAI/bge-m3",
                "reranker": "BAAI/bge-reranker-v2-m3",
                "agent_loading": "Sequential/On-demand",
                "memory_optimization": "Enabled"
            },
            "detailed_results": results
        }

        # Save results
        results_file = f"hotpotqa_sequential_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"📁 Results saved to: {results_file}")

        # Summary
        logger.info("\\n" + "=" * 60)
        logger.info("🎯 SEQUENTIAL HOTPOTQA EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Examples: {results_data['num_examples']}")
        logger.info(f"Accuracy: {results_data['accuracy']:.3f} ({results_data['accuracy']*100:.1f}%)")
        logger.info(f"Time: {results_data['total_time_minutes']:.2f} minutes")
        logger.info(f"🌱 CO2 Emissions: {results_data['carbon_footprint']['total_emissions_kg_co2']:.6f} kg CO2eq")
        logger.info(f"🌱 CSV Created: {results_data['carbon_footprint']['csv_file_created']}")
        logger.info(f"💾 Memory Optimization: {results_data['memory_efficient']}")
        logger.info("=" * 60)

        return results_data

    except Exception as e:
        logger.error(f"Sequential HotpotQA evaluation failed: {e}")
        try:
            final_emissions = tracker.stop()
        except:
            final_emissions = 0.0
        return {"error": str(e), "partial_emissions": final_emissions}

if __name__ == "__main__":
    print("🔬 Sequential HotpotQA Memory-Efficient Evaluation")
    print("Using on-demand agent loading to minimize GPU memory usage")
    print("=" * 60)

    results = run_hotpotqa_sequential_evaluation(num_examples=100)

    if "error" not in results:
        print(f"\\n✅ SEQUENTIAL EVALUATION COMPLETED!")
        print(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"🌱 CO2 Emissions: {results['carbon_footprint']['total_emissions_kg_co2']:.6f} kg CO2eq")
        print(f"💾 Memory Optimization: Enabled")
    else:
        print(f"\\n❌ EVALUATION FAILED: {results['error']}")