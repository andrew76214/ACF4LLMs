"""
Main A2A Pipeline Controller
Entry point for the Agent-to-Agent QA and Math Reasoning System
"""

import sys
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.evaluator import Evaluator
from models.model_manager import ModelManager
from models.vllm_client import VLLMClient
from configs.config import config, validate_config, print_config
from utils.logging import setup_logger, get_logger
from utils.data_processing import load_dataset_by_name

logger = get_logger("a2a_pipeline")

class A2APipeline:
    """
    Main A2A Pipeline for QA and Math reasoning
    Provides high-level interface for the complete system
    """

    def __init__(self, use_vllm_server: bool = True):
        """
        Initialize A2A Pipeline

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

        logger.info("Initializing A2A Pipeline")

    def initialize(self) -> bool:
        """
        Initialize all pipeline components

        Returns:
            True if initialization successful
        """
        try:
            # Validate configuration
            if not validate_config():
                logger.error("Configuration validation failed")
                return False

            print_config()

            # Initialize model
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

            # Initialize evaluator with model client
            self.evaluator = Evaluator(model_client)

            # Health check
            health = self.evaluator.health_check()
            logger.info(f"Pipeline health check: {health}")

            if not health.get("pipeline_test", {}).get("success", False):
                logger.warning("Pipeline health check failed - continuing anyway")

            logger.info("A2A Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def solve_question(
        self,
        question: str,
        dataset: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a single question

        Args:
            question: Question to solve
            dataset: Dataset name for optimal routing
            context: Optional context

        Returns:
            Solution result
        """
        if not self.evaluator:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            result = self.evaluator.evaluate_single_question(
                question=question,
                gold_answer="",  # Not needed for just solving
                dataset=dataset,
                context=context
            )

            return {
                "question": question,
                "answer": result.get("prediction", ""),
                "reasoning": result.get("solver", {}).get("reasoning", ""),
                "task_type": result.get("task_type", ""),
                "used_search": result.get("used_search", False),
                "confidence": result.get("solver", {}).get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0.0)
            }

        except Exception as e:
            logger.error(f"Question solving failed: {e}")
            return {
                "question": question,
                "answer": "",
                "error": str(e)
            }

    def evaluate_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        num_trials: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset

        Args:
            dataset_name: Name of dataset
            sample_size: Number of samples (None for full dataset)
            num_trials: Number of trials for variance analysis

        Returns:
            Evaluation results
        """
        if not self.evaluator:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        logger.info(f"Starting evaluation on {dataset_name}")

        # Use configured sample size if not specified
        if sample_size is None:
            sample_size = config.datasets.sample_sizes.get(dataset_name)

        return self.evaluator.evaluate_dataset(
            dataset_name=dataset_name,
            sample_size=sample_size,
            num_trials=num_trials
        )

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all configured datasets

        Returns:
            Comprehensive evaluation results
        """
        if not self.evaluator:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        return self.evaluator.run_comprehensive_evaluation()

    def interactive_mode(self):
        """Run pipeline in interactive mode"""
        print("\n=== A2A Pipeline Interactive Mode ===")
        print("Enter questions to solve, or 'quit' to exit")
        print("Commands:")
        print("  /dataset <name> - Set dataset context")
        print("  /help - Show this help")
        print("  /quit - Exit")
        print()

        current_dataset = None

        while True:
            try:
                user_input = input(f"[{current_dataset or 'auto'}] Question: ").strip()

                if user_input.lower() in ['quit', '/quit', 'exit']:
                    break
                elif user_input.startswith('/dataset '):
                    current_dataset = user_input[9:].strip()
                    print(f"Dataset context set to: {current_dataset}")
                    continue
                elif user_input == '/help':
                    print("Available datasets: gsm8k, aime, math, hotpotqa, squad_v1, squad_v2, nq")
                    continue
                elif not user_input:
                    continue

                # Solve the question
                print("Solving...")
                result = self.solve_question(user_input, current_dataset)

                # Display result
                print(f"\nAnswer: {result['answer']}")
                print(f"Task Type: {result['task_type']}")
                print(f"Used Search: {'Yes' if result.get('used_search') else 'No'}")
                print(f"Confidence: {result.get('confidence', 0):.3f}")
                print(f"Time: {result.get('processing_time', 0):.2f}s")

                if result.get('reasoning'):
                    print(f"\nReasoning:\n{result['reasoning'][:300]}...")

                print("-" * 50)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")

    def cleanup(self):
        """Cleanup resources"""
        if self.model_manager:
            if hasattr(self.model_manager, 'stop_vllm_server'):
                self.model_manager.stop_vllm_server()
            logger.info("Pipeline cleanup completed")

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="A2A Pipeline - Agent-to-Agent QA and Math Reasoning")
    parser.add_argument("--mode", choices=["interactive", "evaluate", "solve"], default="interactive",
                       help="Execution mode")
    parser.add_argument("--dataset", type=str, help="Dataset to evaluate")
    parser.add_argument("--question", type=str, help="Single question to solve")
    parser.add_argument("--sample-size", type=int, help="Sample size for evaluation")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials for math tasks")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation")
    parser.add_argument("--use-direct-model", action="store_true", help="Use direct model loading instead of vLLM server")
    parser.add_argument("--config-check", action="store_true", help="Check configuration and exit")

    args = parser.parse_args()

    # Configuration check
    if args.config_check:
        print("Checking configuration...")
        if validate_config():
            print("✓ Configuration is valid")
            print_config()
        else:
            print("✗ Configuration validation failed")
            sys.exit(1)
        return

    # Initialize pipeline
    pipeline = A2APipeline(use_vllm_server=not args.use_direct_model)

    try:
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            sys.exit(1)

        # Execute based on mode
        if args.mode == "interactive":
            pipeline.interactive_mode()

        elif args.mode == "solve":
            if not args.question:
                print("Error: --question required for solve mode")
                sys.exit(1)

            result = pipeline.solve_question(args.question, args.dataset)
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            if result.get('reasoning'):
                print(f"Reasoning: {result['reasoning'][:200]}...")

        elif args.mode == "evaluate":
            if args.comprehensive:
                results = pipeline.run_comprehensive_evaluation()
                print(results["report"])
            elif args.dataset:
                results = pipeline.evaluate_dataset(
                    args.dataset,
                    args.sample_size,
                    args.num_trials
                )
                print(f"Results for {args.dataset}:")
                print(f"Accuracy: {results.get('accuracy', 0):.3f}")
                if 'mean_accuracy' in results:
                    print(f"Mean±Std: {results['mean_accuracy']:.3f}±{results.get('std_accuracy', 0):.3f}")
            else:
                print("Error: --dataset or --comprehensive required for evaluate mode")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()