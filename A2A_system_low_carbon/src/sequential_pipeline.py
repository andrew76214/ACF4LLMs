"""
Sequential Memory-Efficient A2A Pipeline
Only loads agents when router determines they are needed
"""

import sys
import time
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.sequential_evaluator import SequentialEvaluator
from models.model_manager import ModelManager
from models.vllm_client import VLLMClient
from configs.config import config, validate_config, print_config
from utils.logging import setup_logger, get_logger
from utils.data_processing import load_dataset_by_name

logger = get_logger("sequential_a2a_pipeline")

class SequentialA2APipeline:
    """
    Memory-Efficient Sequential A2A Pipeline
    Only loads agents when the router determines they're needed
    """

    def __init__(self, use_vllm_server: bool = True):
        """
        Initialize Sequential A2A Pipeline

        Args:
            use_vllm_server: Whether to use vLLM server (True) or direct model loading (False)
        """
        self.use_vllm_server = use_vllm_server
        self.model_manager = None
        self.evaluator = None

        # Setup logging
        setup_logger(
            level=config.experiment.log_level,
            log_to_file=True,
            log_to_console=True
        )

        logger.info("Initializing Sequential A2A Pipeline")

    def initialize(self) -> bool:
        """
        Initialize sequential pipeline components

        Returns:
            True if initialization successful
        """
        try:
            # Validate configuration
            if not validate_config():
                logger.error("Configuration validation failed")
                return False

            print_config()

            # Initialize model (this is the main memory consumer)
            if self.use_vllm_server:
                # Start vLLM server
                self.model_manager = ModelManager()
                if not self.model_manager.start_vllm_server():
                    logger.error("Failed to start vLLM server")
                    return False

                # Create client
                model_client = VLLMClient()
                if not model_client.wait_for_server(timeout=60):
                    logger.error("vLLM server not responding")
                    return False
            else:
                # Direct model loading
                self.model_manager = ModelManager()
                if not self.model_manager.load_model():
                    logger.error("Failed to load model directly")
                    return False
                model_client = self.model_manager

            # Initialize sequential evaluator (only router loaded initially)
            self.evaluator = SequentialEvaluator(model_client)

            # Health check
            health = self.evaluator.health_check()
            logger.info(f"Sequential pipeline health check: {health}")

            if not health.get("pipeline_test", {}).get("success", False):
                logger.warning("Sequential pipeline health check failed - continuing anyway")

            logger.info("Sequential A2A Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Sequential pipeline initialization failed: {e}")
            return False

    def solve_question(self, question: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a single question using sequential agent loading

        Args:
            question: Question to solve
            dataset: Dataset type for routing

        Returns:
            Solution result with memory usage info
        """
        if not self.evaluator:
            return {"error": "Pipeline not initialized"}

        # Get memory status before
        mem_before = self.evaluator.get_memory_status()

        # Solve question
        result = self.evaluator.solve_question(question, dataset)

        # Get memory status after
        mem_after = self.evaluator.get_memory_status()

        # Add memory info to result
        result["memory_usage"] = {
            "before": mem_before,
            "after": mem_after
        }

        return result

    def evaluate_single_question(
        self,
        question: str,
        gold_answer: Any,
        dataset: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through sequential pipeline

        Args:
            question: Question to evaluate
            gold_answer: Ground truth answer
            dataset: Dataset name for routing
            context: Optional context
            **kwargs: Additional parameters

        Returns:
            Evaluation results with memory usage info
        """
        if not self.evaluator:
            return {"error": "Pipeline not initialized"}

        # Get memory status before
        mem_before = self.evaluator.get_memory_status()

        # Evaluate question
        result = self.evaluator.evaluate_single_question(
            question, gold_answer, dataset, context, **kwargs
        )

        # Get memory status after
        mem_after = self.evaluator.get_memory_status()

        # Add memory info to result
        result["memory_usage"] = {
            "before": mem_before,
            "after": mem_after
        }

        return result

    def evaluate_dataset(
        self,
        dataset_name: str,
        num_examples: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on a complete dataset using sequential approach

        Args:
            dataset_name: Name of dataset to evaluate
            num_examples: Number of examples to evaluate (None for all)
            save_results: Whether to save results to file

        Returns:
            Evaluation results
        """
        if not self.evaluator:
            return {"error": "Pipeline not initialized"}

        logger.info(f"Starting sequential evaluation on {dataset_name}")

        try:
            # Load dataset
            dataset = load_dataset_by_name(dataset_name, "test")
            if num_examples:
                dataset = dataset.select(range(min(num_examples, len(dataset))))

            logger.info(f"Evaluating {len(dataset)} examples from {dataset_name}")

            results = []
            correct = 0
            start_time = time.time()

            for i, example in enumerate(dataset):
                logger.info(f"Processing {i+1}/{len(dataset)}: {example['question'][:100]}...")

                # Get memory status periodically
                if i % 10 == 0:
                    mem_status = self.evaluator.get_memory_status()
                    logger.info(f"Memory status at question {i+1}: {mem_status}")

                result = self.evaluate_single_question(
                    example["question"],
                    example.get("answer", ""),
                    dataset_name
                )

                if "error" not in result and result.get("correct", False):
                    correct += 1

                results.append(result)

                # Log progress
                if (i + 1) % 25 == 0:
                    current_accuracy = correct / (i + 1)
                    elapsed_time = (time.time() - start_time) / 60
                    logger.info(f"Progress: {i+1}/{len(dataset)} ({current_accuracy:.3f} accuracy, {elapsed_time:.2f}min)")

            total_time = time.time() - start_time
            accuracy = correct / len(dataset)

            # Compile final results
            evaluation_results = {
                "dataset": dataset_name,
                "num_examples": len(dataset),
                "correct": correct,
                "accuracy": accuracy,
                "total_time_minutes": total_time / 60,
                "avg_time_per_question": total_time / len(dataset),
                "sequential_evaluation": True,
                "memory_efficient": True,
                "detailed_results": results
            }

            logger.info(f"Sequential evaluation complete:")
            logger.info(f"  Dataset: {dataset_name}")
            logger.info(f"  Examples: {len(dataset)}")
            logger.info(f"  Accuracy: {accuracy:.3f} ({correct}/{len(dataset)})")
            logger.info(f"  Time: {total_time/60:.2f} minutes")

            return evaluation_results

        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
            return {"error": str(e)}

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage status"""
        if not self.evaluator:
            return {"error": "Pipeline not initialized"}

        return self.evaluator.get_memory_status()

    def cleanup(self):
        """Clean up all resources and free memory"""
        logger.info("Cleaning up sequential pipeline...")

        if self.evaluator:
            self.evaluator.cleanup()

        if self.model_manager:
            self.model_manager.cleanup()

        logger.info("Sequential pipeline cleanup complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential A2A Pipeline")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to evaluate")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to evaluate")
    parser.add_argument("--use-server", action="store_true", help="Use vLLM server instead of direct loading")

    args = parser.parse_args()

    # Initialize and run pipeline
    with SequentialA2APipeline(use_vllm_server=args.use_server) as pipeline:
        if pipeline.initialize():
            results = pipeline.evaluate_dataset(args.dataset, args.num_examples)
            print(f"Results: {results}")
        else:
            print("Failed to initialize pipeline")