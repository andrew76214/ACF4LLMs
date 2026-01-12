"""Main benchmark runner that coordinates all evaluations."""

import os
import time
import json
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging

from src.evaluation.evaluators.lm_eval_evaluator import LMEvalEvaluator
from src.evaluation.evaluators.humaneval_evaluator import HumanEvalEvaluator
from src.evaluation.evaluators.latency_evaluator import LatencyEvaluator

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark evaluations for compressed models."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        """Initialize benchmark runner.

        Args:
            device: Device to run evaluations on
            cache_dir: Directory to cache results
        """
        self.device = device
        self.cache_dir = cache_dir or "data/cache/evaluations"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # lm-eval handles most benchmarks (gsm8k, commonsenseqa, truthfulqa, bigbench_hard,
        # mmlu, hellaswag, arc_easy, arc_challenge, winogrande)
        self.lm_eval_evaluator = LMEvalEvaluator(device=device)

        # Custom evaluators for specialized tasks
        self.custom_evaluators = {
            "humaneval": HumanEvalEvaluator(device=device),
            "latency": LatencyEvaluator(device=device),
        }

    def run_full_evaluation(
        self,
        checkpoint_path: str,
        benchmarks: List[str],
        use_proxy: bool = False,
        proxy_samples: int = 100,
        batch_size: int = 8,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run full evaluation suite on a model.

        Args:
            checkpoint_path: Path to model checkpoint
            benchmarks: List of benchmarks to run
            use_proxy: Whether to use proxy evaluation
            proxy_samples: Number of samples for proxy
            batch_size: Batch size for evaluation
            save_results: Whether to save results to cache

        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Starting full evaluation for {checkpoint_path}")
        start_time = time.time()

        results = {
            "checkpoint_path": checkpoint_path,
            "benchmarks": benchmarks,
            "is_proxy": use_proxy,
            "timestamp": datetime.now().isoformat(),
            "benchmark_scores": {},
            "performance_metrics": {},
        }

        # Load model once
        try:
            model, tokenizer = self._load_model(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": str(e)}

        # Separate benchmarks into lm-eval vs custom
        lm_eval_tasks = [
            b for b in benchmarks
            if LMEvalEvaluator.is_task_supported(b.lower())
        ]
        custom_tasks = [
            b for b in benchmarks
            if b.lower() in self.custom_evaluators
        ]
        unknown_tasks = [
            b for b in benchmarks
            if not LMEvalEvaluator.is_task_supported(b.lower())
            and b.lower() not in self.custom_evaluators
        ]

        if unknown_tasks:
            logger.warning(f"Unknown benchmarks (will be skipped): {unknown_tasks}")

        # Run lm-eval benchmarks (all in one batch for efficiency)
        if lm_eval_tasks:
            logger.info(f"Running lm-eval benchmarks: {lm_eval_tasks}")
            try:
                limit = proxy_samples if use_proxy else None
                lm_scores = self.lm_eval_evaluator.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=lm_eval_tasks,
                    limit=limit,
                    batch_size=batch_size,
                )
                results["benchmark_scores"].update(lm_scores)
            except Exception as e:
                logger.error(f"lm-eval evaluation failed: {e}")
                # Set failed benchmarks to 0
                for task in lm_eval_tasks:
                    results["benchmark_scores"][task.lower()] = 0.0

        # Run custom evaluators (humaneval)
        for benchmark in custom_tasks:
            benchmark_lower = benchmark.lower()
            if benchmark_lower in self.custom_evaluators:
                logger.info(f"Running custom {benchmark} evaluation...")
                evaluator = self.custom_evaluators[benchmark_lower]

                try:
                    if use_proxy:
                        score = evaluator.evaluate_proxy(
                            model, tokenizer, num_samples=proxy_samples, batch_size=batch_size
                        )
                    else:
                        score = evaluator.evaluate(model, tokenizer, batch_size=batch_size)

                    results["benchmark_scores"][benchmark_lower] = score
                    logger.info(f"{benchmark}: {score:.3f}")

                except Exception as e:
                    logger.error(f"Evaluation failed for {benchmark}: {e}")
                    results["benchmark_scores"][benchmark_lower] = 0.0

        # Calculate average accuracy
        if results["benchmark_scores"]:
            results["average_accuracy"] = sum(results["benchmark_scores"].values()) / len(
                results["benchmark_scores"]
            )
        else:
            results["average_accuracy"] = 0.0

        # Run performance evaluation
        logger.info("Running performance evaluation...")
        perf_results = self.custom_evaluators["latency"].evaluate_performance(
            model, tokenizer, batch_size=batch_size
        )
        results["performance_metrics"] = perf_results

        # Add key metrics to top level for convenience
        results["latency_ms"] = perf_results.get("mean_latency_ms", 0)
        results["throughput_tokens_per_sec"] = perf_results.get("throughput_tokens_per_sec", 0)
        results["memory_gb"] = perf_results.get("peak_memory_gb", 0)

        # Evaluation time
        results["evaluation_time_sec"] = time.time() - start_time

        # Save results if requested
        if save_results:
            self._save_results(results)

        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Evaluation completed in {results['evaluation_time_sec']:.1f}s")
        return results

    def run_proxy_evaluation(
        self,
        checkpoint_path: str,
        benchmarks: List[str],
        sample_ratio: float = 0.1,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Run fast proxy evaluation.

        Args:
            checkpoint_path: Model checkpoint path
            benchmarks: Benchmarks to evaluate
            sample_ratio: Ratio of dataset to use
            max_samples: Maximum samples per benchmark

        Returns:
            Dictionary with proxy results and confidence
        """
        num_samples = min(max_samples, int(10000 * sample_ratio))

        results = self.run_full_evaluation(
            checkpoint_path=checkpoint_path,
            benchmarks=benchmarks,
            use_proxy=True,
            proxy_samples=num_samples,
        )

        # Add confidence score based on sample size
        confidence = min(0.99, 0.5 + (num_samples / 2000))
        results["confidence"] = confidence
        results["sample_size"] = num_samples

        # Estimate full evaluation accuracy
        results["estimated_full_accuracy"] = results["average_accuracy"] * (1 + (1 - confidence) * 0.05)

        return results

    def compare_models(
        self,
        baseline_checkpoint: str,
        compressed_checkpoint: str,
        benchmarks: List[str],
    ) -> Dict[str, Any]:
        """Compare baseline and compressed models.

        Args:
            baseline_checkpoint: Baseline model path
            compressed_checkpoint: Compressed model path
            benchmarks: Benchmarks to compare on

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing baseline and compressed models...")

        # Evaluate baseline
        baseline_results = self.run_full_evaluation(baseline_checkpoint, benchmarks, use_proxy=True)

        # Evaluate compressed
        compressed_results = self.run_full_evaluation(compressed_checkpoint, benchmarks, use_proxy=True)

        # Calculate relative performance
        comparison = {
            "baseline_checkpoint": baseline_checkpoint,
            "compressed_checkpoint": compressed_checkpoint,
            "baseline_scores": baseline_results["benchmark_scores"],
            "compressed_scores": compressed_results["benchmark_scores"],
            "relative_performance": {},
            "accuracy_retention": compressed_results["average_accuracy"]
            / baseline_results["average_accuracy"]
            if baseline_results["average_accuracy"] > 0
            else 0,
        }

        # Per-benchmark comparison
        for benchmark in benchmarks:
            baseline_score = baseline_results["benchmark_scores"].get(benchmark, 0)
            compressed_score = compressed_results["benchmark_scores"].get(benchmark, 0)

            if baseline_score > 0:
                comparison["relative_performance"][benchmark] = compressed_score / baseline_score
            else:
                comparison["relative_performance"][benchmark] = 0

        # Performance comparison
        comparison["speedup"] = (
            baseline_results["latency_ms"] / compressed_results["latency_ms"]
            if compressed_results["latency_ms"] > 0
            else 0
        )
        comparison["memory_reduction"] = 1 - (
            compressed_results["memory_gb"] / baseline_results["memory_gb"]
            if baseline_results["memory_gb"] > 0
            else 0
        )

        return comparison

    def _load_model(self, checkpoint_path: str) -> Tuple[Any, Any]:
        """Load model and tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {checkpoint_path}")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()

        return model, tokenizer

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to cache.

        Args:
            results: Evaluation results to save
        """
        # Generate cache key
        checkpoint_name = os.path.basename(results["checkpoint_path"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(self.cache_dir, f"{checkpoint_name}_{timestamp}.json")

        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {cache_file}")

    def get_cached_results(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Get cached evaluation results if available.

        Args:
            checkpoint_path: Model checkpoint path

        Returns:
            Cached results or None
        """
        checkpoint_name = os.path.basename(checkpoint_path)
        import glob

        # Find most recent cache file
        pattern = os.path.join(self.cache_dir, f"{checkpoint_name}_*.json")
        cache_files = glob.glob(pattern)

        if not cache_files:
            return None

        # Get most recent
        latest_cache = max(cache_files, key=os.path.getctime)

        with open(latest_cache, "r") as f:
            results = json.load(f)

        logger.info(f"Loaded cached results from {latest_cache}")
        return results


class ProxyEvaluator:
    """Fast proxy evaluation for quick model assessment."""

    def __init__(self, runner: BenchmarkRunner):
        """Initialize proxy evaluator.

        Args:
            runner: Benchmark runner instance
        """
        self.runner = runner

    def evaluate_with_confidence(
        self,
        checkpoint_path: str,
        benchmarks: List[str],
        confidence_threshold: float = 0.95,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate with adaptive sampling until confidence threshold.

        Args:
            checkpoint_path: Model checkpoint
            benchmarks: Benchmarks to evaluate
            confidence_threshold: Required confidence level
            max_iterations: Maximum sampling iterations

        Returns:
            Dictionary with results and confidence
        """
        samples_per_iteration = 100
        total_samples = 0
        results = None

        for iteration in range(max_iterations):
            total_samples += samples_per_iteration

            # Run evaluation
            results = self.runner.run_proxy_evaluation(
                checkpoint_path=checkpoint_path,
                benchmarks=benchmarks,
                sample_ratio=total_samples / 10000,
                max_samples=total_samples,
            )

            # Check confidence
            if results["confidence"] >= confidence_threshold:
                logger.info(f"Reached confidence {results['confidence']:.2%} after {iteration + 1} iterations")
                break

            # Increase samples for next iteration
            samples_per_iteration *= 2

        return results

    def quick_assessment(self, checkpoint_path: str) -> Dict[str, str]:
        """Get quick assessment of model quality.

        Args:
            checkpoint_path: Model checkpoint

        Returns:
            Dictionary with quality assessment
        """
        # Run minimal evaluation
        results = self.runner.run_proxy_evaluation(
            checkpoint_path=checkpoint_path,
            benchmarks=["gsm8k", "commonsenseqa"],
            sample_ratio=0.05,
            max_samples=50,
        )

        accuracy = results["average_accuracy"]
        latency = results["latency_ms"]

        # Assess quality
        assessment = {
            "overall_quality": "unknown",
            "accuracy_level": "unknown",
            "performance_level": "unknown",
            "recommendation": "",
        }

        # Accuracy assessment
        if accuracy >= 0.9:
            assessment["accuracy_level"] = "excellent"
        elif accuracy >= 0.8:
            assessment["accuracy_level"] = "good"
        elif accuracy >= 0.7:
            assessment["accuracy_level"] = "acceptable"
        else:
            assessment["accuracy_level"] = "poor"

        # Performance assessment
        if latency < 50:
            assessment["performance_level"] = "excellent"
        elif latency < 100:
            assessment["performance_level"] = "good"
        elif latency < 200:
            assessment["performance_level"] = "acceptable"
        else:
            assessment["performance_level"] = "poor"

        # Overall assessment
        if assessment["accuracy_level"] in ["excellent", "good"] and assessment["performance_level"] in [
            "excellent",
            "good",
        ]:
            assessment["overall_quality"] = "excellent"
            assessment["recommendation"] = "Ready for production"
        elif assessment["accuracy_level"] in ["good", "acceptable"] and assessment["performance_level"] in [
            "good",
            "acceptable",
        ]:
            assessment["overall_quality"] = "good"
            assessment["recommendation"] = "Suitable for most applications"
        elif assessment["accuracy_level"] == "acceptable" or assessment["performance_level"] == "acceptable":
            assessment["overall_quality"] = "acceptable"
            assessment["recommendation"] = "Consider further optimization"
        else:
            assessment["overall_quality"] = "poor"
            assessment["recommendation"] = "Needs significant improvement"

        return assessment


def create_benchmark_runner(device: Optional[str] = None) -> BenchmarkRunner:
    """Create a benchmark runner with appropriate configuration.

    Args:
        device: Device to use (auto-detect if None)

    Returns:
        Configured benchmark runner
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return BenchmarkRunner(device=device)


__all__ = ["BenchmarkRunner", "ProxyEvaluator", "create_benchmark_runner"]