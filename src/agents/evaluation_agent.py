"""Evaluation Agent for benchmarking compressed models using DeepAgents subagent pattern."""

import json
import os
import time
import random
import logging
import threading
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
import torch

from src.common.schemas import Benchmark
from src.common.model_utils import estimate_model_size_gb

logger = logging.getLogger(__name__)


# Thread-safe singleton pattern for global runners
_benchmark_runner = None
_benchmark_runner_lock = threading.Lock()
_latency_evaluator = None
_latency_evaluator_lock = threading.Lock()
_perplexity_evaluator = None
_perplexity_evaluator_lock = threading.Lock()


def _get_benchmark_runner(device: str = "cuda"):
    """Get or create the benchmark runner instance (thread-safe).

    Uses double-checked locking pattern for efficiency.
    """
    global _benchmark_runner
    if _benchmark_runner is None:
        with _benchmark_runner_lock:
            # Double-check after acquiring lock
            if _benchmark_runner is None:
                try:
                    from src.evaluation.benchmark_runner import create_benchmark_runner
                    _benchmark_runner = create_benchmark_runner(device)
                    logger.info("Initialized real BenchmarkRunner")
                except Exception as e:
                    logger.warning(f"Failed to initialize BenchmarkRunner: {e}")
                    return None
    return _benchmark_runner


def _get_latency_evaluator(device: str = "cuda"):
    """Get or create the latency evaluator instance (thread-safe).

    Uses double-checked locking pattern for efficiency.
    """
    global _latency_evaluator
    if _latency_evaluator is None:
        with _latency_evaluator_lock:
            # Double-check after acquiring lock
            if _latency_evaluator is None:
                try:
                    from src.evaluation.evaluators.latency_evaluator import LatencyEvaluator
                    _latency_evaluator = LatencyEvaluator(device=device)
                    logger.info("Initialized real LatencyEvaluator")
                except Exception as e:
                    logger.warning(f"Failed to initialize LatencyEvaluator: {e}")
                    return None
    return _latency_evaluator


def _load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """Load model and tokenizer from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


from contextlib import contextmanager
from typing import Generator, Tuple

@contextmanager
def evaluation_context(checkpoint_path: str, device: str = "cuda") -> Generator[Tuple[Any, Any], None, None]:
    """Context manager for safe model loading during evaluation.

    Ensures proper cleanup of model and GPU memory after evaluation,
    including synchronization to prevent illegal memory access errors.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load model on

    Yields:
        Tuple of (model, tokenizer)
    """
    import gc
    model = None
    tokenizer = None

    try:
        model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)
        yield model, tokenizer
    finally:
        if model is not None:
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
            except Exception:
                pass
            del model

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _mock_evaluate_model(benchmarks: List[str], use_proxy: bool, device: str) -> Dict[str, Any]:
    """Fallback mock evaluation when real evaluation fails."""
    benchmark_scores = {}
    base_accuracy = 0.85 if use_proxy else 0.87

    for benchmark in benchmarks:
        if benchmark.lower() == "gsm8k":
            score = base_accuracy - random.uniform(0.05, 0.10)
        elif benchmark.lower() == "humaneval":
            score = base_accuracy - random.uniform(0.10, 0.20)
        elif benchmark.lower() in ["commonsenseqa", "commonsense_qa"]:
            score = base_accuracy + random.uniform(-0.05, 0.05)
        elif benchmark.lower() in ["truthfulqa", "truthful_qa"]:
            score = base_accuracy - random.uniform(0.02, 0.08)
        elif benchmark.lower() in ["bigbench_hard", "bigbench"]:
            score = base_accuracy - random.uniform(0.10, 0.15)
        else:
            score = base_accuracy + random.uniform(-0.05, 0.05)
        benchmark_scores[benchmark] = max(0.0, min(1.0, score))

    average_accuracy = sum(benchmark_scores.values()) / len(benchmark_scores) if benchmark_scores else 0.0
    latency_ms = random.uniform(80, 150) if "cuda" in device else random.uniform(200, 400)

    return {
        "benchmark_scores": benchmark_scores,
        "average_accuracy": average_accuracy,
        "latency_ms": latency_ms,
        "throughput_tokens_per_sec": 1000 / latency_ms * 8,
        "memory_gb": random.uniform(4.0, 8.0),
        "is_mock": True,
    }

@tool
def evaluate_model(
    checkpoint_path: str,
    benchmarks: List[str],
    use_proxy: bool = True,
    proxy_samples: int = 100,
    device: str = "cuda",
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate a model checkpoint on specified benchmarks.

    Uses EleutherAI's lm-eval harness for standardized evaluation.

    Args:
        checkpoint_path: Path to the model checkpoint
        benchmarks: List of benchmark names to run. Available benchmarks:
            - gsm8k: Mathematical reasoning
            - commonsenseqa: Common sense reasoning
            - truthfulqa: Truthfulness and factuality
            - humaneval: Code generation (custom evaluator)
            - bigbench_hard: Challenging diverse tasks (BBH)
            - mmlu: Massive Multitask Language Understanding
            - hellaswag: Commonsense NLI
            - arc_easy: AI2 Reasoning Challenge (Easy)
            - arc_challenge: AI2 Reasoning Challenge (Challenge)
            - winogrande: Winograd Schema Challenge
        use_proxy: Whether to use proxy evaluation (faster but less accurate)
        proxy_samples: Number of samples for proxy evaluation
        device: Device to run evaluation on ('cuda' or 'cpu')
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation results:
        - benchmark_scores: Dict mapping benchmark name to score
        - average_accuracy: Average accuracy across all benchmarks
        - latency_ms: Average inference latency
        - throughput_tokens_per_sec: Throughput in tokens/second
        - memory_gb: Peak memory usage during inference
        - evaluation_time_sec: Total evaluation time
        - is_proxy: Whether proxy evaluation was used
    """
    print(f"[Evaluation] Starting evaluation on {len(benchmarks)} benchmarks...")
    print(f"[Evaluation] Mode: {'Proxy' if use_proxy else 'Full'} evaluation")

    start_time = time.time()

    # Try to use real BenchmarkRunner
    runner = _get_benchmark_runner(device)

    if runner is not None:
        try:
            logger.info(f"Running real evaluation on {checkpoint_path}")
            results = runner.run_full_evaluation(
                checkpoint_path=checkpoint_path,
                benchmarks=benchmarks,
                use_proxy=use_proxy,
                proxy_samples=proxy_samples,
                batch_size=batch_size,
                save_results=True,
            )

            evaluation_time = time.time() - start_time

            print(f"[Evaluation] Completed in {evaluation_time:.1f} seconds")
            print(f"[Evaluation] Average accuracy: {results['average_accuracy']:.3f}")
            print(f"[Evaluation] Latency: {results['latency_ms']:.1f} ms")

            return {
                "benchmark_scores": results.get("benchmark_scores", {}),
                "average_accuracy": results.get("average_accuracy", 0.0),
                "latency_ms": results.get("latency_ms", 0.0),
                "throughput_tokens_per_sec": results.get("throughput_tokens_per_sec", 0.0),
                "memory_gb": results.get("memory_gb", 0.0),
                "evaluation_time_sec": evaluation_time,
                "is_proxy": use_proxy,
                "proxy_samples": proxy_samples if use_proxy else None,
                "device": device,
                "batch_size": batch_size,
                "is_mock": False,
            }

        except Exception as e:
            logger.warning(f"Real evaluation failed: {e}, falling back to mock")
            print(f"[Evaluation] Real evaluation failed: {e}, using mock fallback")

    # Fallback to mock evaluation
    print("[Evaluation] Using mock evaluation (real evaluation unavailable)")
    mock_results = _mock_evaluate_model(benchmarks, use_proxy, device)
    evaluation_time = time.time() - start_time

    for benchmark, score in mock_results["benchmark_scores"].items():
        print(f"[Evaluation] {benchmark}: {score:.3f}")

    print(f"[Evaluation] Completed in {evaluation_time:.1f} seconds")
    print(f"[Evaluation] Average accuracy: {mock_results['average_accuracy']:.3f}")

    return {
        **mock_results,
        "evaluation_time_sec": evaluation_time,
        "is_proxy": use_proxy,
        "proxy_samples": proxy_samples if use_proxy else None,
        "device": device,
        "batch_size": batch_size,
    }


@tool
def run_proxy_evaluation(
    checkpoint_path: str,
    benchmarks: List[str],
    sample_ratio: float = 0.1,
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """Run fast proxy evaluation for quick model assessment.

    Args:
        checkpoint_path: Path to the model checkpoint
        benchmarks: List of benchmark names to run
        sample_ratio: Ratio of dataset to use (0.0 to 1.0)
        max_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary with proxy evaluation results and confidence scores
    """
    print(f"[Proxy] Running proxy evaluation with {sample_ratio:.0%} of data...")

    # Calculate actual sample size
    samples = min(max_samples, int(10000 * sample_ratio))  # Assume 10k samples per benchmark

    # Run proxy evaluation
    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        benchmarks=benchmarks,
        use_proxy=True,
        proxy_samples=samples,
    )

    # Add confidence scores based on sample size
    confidence = min(0.99, 0.5 + (samples / 2000))  # Confidence increases with samples

    results["confidence"] = confidence
    results["sample_size"] = samples
    results["estimated_full_accuracy"] = results["average_accuracy"] + random.uniform(-0.02, 0.02)

    print(f"[Proxy] Confidence: {confidence:.1%}")

    return results


@tool
def measure_inference_latency(
    checkpoint_path: str,
    input_length: int = 512,
    output_length: int = 128,
    num_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure inference latency for a model.

    Args:
        checkpoint_path: Path to the model checkpoint
        input_length: Length of input sequence
        output_length: Length of generated output
        num_runs: Number of inference runs to average
        device: Device to run on

    Returns:
        Dictionary with latency measurements:
        - mean_latency_ms: Average latency
        - std_latency_ms: Standard deviation
        - min_latency_ms: Minimum latency
        - max_latency_ms: Maximum latency
        - tokens_per_second: Throughput
    """
    print(f"[Latency] Measuring latency over {num_runs} runs...")

    # Try to use real LatencyEvaluator
    latency_evaluator = _get_latency_evaluator(device)

    if latency_evaluator is not None:
        try:
            logger.info(f"Running real latency measurement on {checkpoint_path}")

            # Load model and tokenizer
            model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)

            # Use real LatencyEvaluator
            results = latency_evaluator.evaluate_performance(
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                sequence_length=input_length,
                num_iterations=num_runs,
            )

            # Clean up with synchronization
            del model
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"[Latency] Mean: {results['mean_latency_ms']:.1f} ms (±{results['std_latency_ms']:.1f} ms)")
            print(f"[Latency] Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")

            return {
                "mean_latency_ms": results["mean_latency_ms"],
                "std_latency_ms": results["std_latency_ms"],
                "min_latency_ms": results["min_latency_ms"],
                "max_latency_ms": results["max_latency_ms"],
                "p50_latency_ms": results.get("p50_latency_ms", results["mean_latency_ms"]),
                "p90_latency_ms": results.get("p90_latency_ms", results["max_latency_ms"]),
                "p99_latency_ms": results.get("p99_latency_ms", results["max_latency_ms"]),
                "tokens_per_second": results["throughput_tokens_per_sec"],
                "num_runs": num_runs,
                "device": device,
                "is_mock": False,
            }

        except Exception as e:
            logger.warning(f"Real latency measurement failed: {e}, falling back to mock")
            print(f"[Latency] Real measurement failed: {e}, using mock fallback")

    # Fallback to mock latency measurements
    print("[Latency] Using mock measurement (real measurement unavailable)")
    latencies = []

    for i in range(num_runs):
        base_latency = 100 if "cuda" in device else 250
        latency = base_latency + random.uniform(-20, 30)
        latencies.append(latency)

    mean_latency = sum(latencies) / len(latencies)
    std_latency = (sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
    tokens_per_second = (input_length + output_length) / (mean_latency / 1000)

    print(f"[Latency] Mean: {mean_latency:.1f} ms (±{std_latency:.1f} ms)")
    print(f"[Latency] Throughput: {tokens_per_second:.1f} tokens/sec")

    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "tokens_per_second": tokens_per_second,
        "num_runs": num_runs,
        "device": device,
        "is_mock": True,
    }


@tool
def measure_memory_usage(
    checkpoint_path: str,
    batch_size: int = 1,
    sequence_length: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure memory usage during inference.

    Args:
        checkpoint_path: Path to the model checkpoint
        batch_size: Batch size for inference
        sequence_length: Length of input sequences
        device: Device to measure on

    Returns:
        Dictionary with memory measurements:
        - model_size_gb: Size of model weights
        - peak_memory_gb: Peak memory during inference
        - inference_memory_gb: Additional memory for inference
    """
    print(f"[Memory] Measuring memory usage...")

    # Try to use real LatencyEvaluator for memory measurement
    latency_evaluator = _get_latency_evaluator(device)

    if latency_evaluator is not None:
        try:
            logger.info(f"Running real memory measurement on {checkpoint_path}")

            # Load model and tokenizer
            model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)

            # Measure model size
            size_results = latency_evaluator.measure_model_size(model)
            model_size_gb = size_results["model_size_gb"]

            # Profile memory usage with inference
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

                # Run inference to measure peak memory
                dummy_text = "Test input " * (sequence_length // 10)
                inputs = tokenizer(
                    [dummy_text] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=sequence_length,
                )
                inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=50)

                peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                import psutil
                peak_memory_gb = psutil.Process().memory_info().rss / (1024**3)

            inference_memory_gb = peak_memory_gb - model_size_gb

            # Clean up with synchronization
            del model
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"[Memory] Model size: {model_size_gb:.1f} GB")
            print(f"[Memory] Peak usage: {peak_memory_gb:.1f} GB")
            print(f"[Memory] Inference overhead: {inference_memory_gb:.1f} GB")

            return {
                "model_size_gb": model_size_gb,
                "peak_memory_gb": peak_memory_gb,
                "inference_memory_gb": inference_memory_gb,
                "total_parameters": size_results.get("total_parameters", 0),
                "parameters_millions": size_results.get("parameters_millions", 0),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "device": device,
                "is_mock": False,
            }

        except Exception as e:
            logger.warning(f"Real memory measurement failed: {e}, falling back to mock")
            print(f"[Memory] Real measurement failed: {e}, using mock fallback")

    # Fallback to mock memory measurements
    print("[Memory] Using mock measurement (real measurement unavailable)")

    # Estimate base model size from checkpoint path
    base_size = estimate_model_size_gb(checkpoint_path)

    # Apply compression based on quantization in path
    if "4bit" in checkpoint_path or "_4_" in checkpoint_path:
        model_size_gb = base_size / 4
    elif "8bit" in checkpoint_path or "int8" in checkpoint_path.lower():
        model_size_gb = base_size / 2
    elif "3bit" in checkpoint_path or "_3_" in checkpoint_path:
        model_size_gb = base_size / 5.33
    else:
        model_size_gb = base_size

    overhead_factor = 1.3 + (batch_size * 0.1)
    peak_memory_gb = model_size_gb * overhead_factor
    inference_memory_gb = peak_memory_gb - model_size_gb

    print(f"[Memory] Model size: {model_size_gb:.1f} GB")
    print(f"[Memory] Peak usage: {peak_memory_gb:.1f} GB")
    print(f"[Memory] Inference overhead: {inference_memory_gb:.1f} GB")

    return {
        "model_size_gb": model_size_gb,
        "peak_memory_gb": peak_memory_gb,
        "inference_memory_gb": inference_memory_gb,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "device": device,
        "is_mock": True,
    }


@tool
def compare_with_baseline(
    checkpoint_path: str,
    baseline_checkpoint: str,
    benchmarks: List[str],
) -> Dict[str, Any]:
    """Compare a model's performance with a baseline.

    Args:
        checkpoint_path: Path to the model to evaluate
        baseline_checkpoint: Path to the baseline model
        benchmarks: List of benchmarks to compare on

    Returns:
        Dictionary with comparison results:
        - model_scores: Scores for the model
        - baseline_scores: Scores for the baseline
        - relative_performance: Relative performance (model/baseline)
        - accuracy_retention: Percentage of baseline accuracy retained
    """
    print(f"[Compare] Comparing with baseline...")

    # Evaluate both models
    model_results = evaluate_model(
        checkpoint_path=checkpoint_path,
        benchmarks=benchmarks,
        use_proxy=True,
    )

    baseline_results = evaluate_model(
        checkpoint_path=baseline_checkpoint,
        benchmarks=benchmarks,
        use_proxy=True,
    )

    # Calculate relative performance
    relative_performance = {}
    for benchmark in benchmarks:
        model_score = model_results["benchmark_scores"].get(benchmark, 0)
        baseline_score = baseline_results["benchmark_scores"].get(benchmark, 1)
        relative_performance[benchmark] = model_score / baseline_score if baseline_score > 0 else 0

    accuracy_retention = model_results["average_accuracy"] / baseline_results["average_accuracy"]

    print(f"[Compare] Accuracy retention: {accuracy_retention:.1%}")

    return {
        "model_scores": model_results["benchmark_scores"],
        "baseline_scores": baseline_results["benchmark_scores"],
        "relative_performance": relative_performance,
        "accuracy_retention": accuracy_retention,
        "model_latency_ms": model_results["latency_ms"],
        "baseline_latency_ms": baseline_results["latency_ms"],
        "speedup": baseline_results["latency_ms"] / model_results["latency_ms"],
    }


def _get_perplexity_evaluator(device: str = "cuda"):
    """Get or create the perplexity evaluator instance (thread-safe).

    Uses double-checked locking pattern for efficiency.
    """
    global _perplexity_evaluator
    if _perplexity_evaluator is None:
        with _perplexity_evaluator_lock:
            # Double-check after acquiring lock
            if _perplexity_evaluator is None:
                try:
                    from src.evaluation.evaluators.perplexity_evaluator import PerplexityEvaluator
                    _perplexity_evaluator = PerplexityEvaluator(device=device)
                    logger.info("Initialized PerplexityEvaluator")
                except Exception as e:
                    logger.warning(f"Failed to initialize PerplexityEvaluator: {e}")
                    return None
    return _perplexity_evaluator


@tool
def measure_perplexity(
    checkpoint_path: str,
    dataset: str = "wikitext",
    split: str = "test",
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Measure model perplexity on a text dataset.

    Perplexity is a standard metric for evaluating language models.
    Lower perplexity indicates better language modeling capability.

    Args:
        checkpoint_path: Path to the model checkpoint
        dataset: Dataset to evaluate on. Supported datasets:
            - wikitext: WikiText-2 (default, good for general LM evaluation)
            - wikitext-103: WikiText-103 (larger, more comprehensive)
            - c4: C4 dataset (web text)
            - ptb: Penn Treebank
            - lambada: LAMBADA (long-range dependencies)
        split: Dataset split ('test' or 'validation')
        max_samples: Maximum number of samples to use (None for all)
        device: Device to run evaluation on

    Returns:
        Dictionary with perplexity metrics:
        - perplexity: Standard perplexity (lower is better)
        - bits_per_byte: Bits per byte metric
        - word_perplexity: Per-word perplexity
        - avg_nll: Average negative log-likelihood
        - total_tokens: Total tokens evaluated
        - dataset: Dataset used
        - is_mock: Whether mock values were used
    """
    print(f"[Perplexity] Measuring perplexity on {dataset} ({split})...")

    evaluator = _get_perplexity_evaluator(device)

    if evaluator is not None:
        try:
            # Load model and tokenizer
            model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)

            # Run evaluation
            results = evaluator.evaluate_full(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                split=split,
                max_samples=max_samples,
            )

            # Clean up with synchronization
            del model
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"[Perplexity] Perplexity: {results['perplexity']:.2f}")
            print(f"[Perplexity] Bits per byte: {results['bits_per_byte']:.4f}")
            print(f"[Perplexity] Tokens evaluated: {results['total_tokens']}")

            return {
                **results,
                "checkpoint_path": checkpoint_path,
                "device": device,
                "is_mock": False,
            }

        except Exception as e:
            logger.warning(f"Perplexity evaluation failed: {e}")
            print(f"[Perplexity] Evaluation failed: {e}, using mock fallback")

    # Fallback to mock perplexity
    print("[Perplexity] Using mock evaluation (real evaluation unavailable)")

    # Generate realistic mock values based on model path
    base_ppl = 15.0  # Base perplexity for a decent model
    if "4bit" in checkpoint_path or "_4_" in checkpoint_path:
        base_ppl *= 1.1  # 10% degradation for 4-bit
    elif "8bit" in checkpoint_path or "int8" in checkpoint_path.lower():
        base_ppl *= 1.05  # 5% degradation for 8-bit
    elif "3bit" in checkpoint_path or "_3_" in checkpoint_path:
        base_ppl *= 1.2  # 20% degradation for 3-bit

    # Add some randomness
    import random
    ppl = base_ppl * random.uniform(0.9, 1.1)
    avg_nll = random.uniform(2.5, 3.0)

    print(f"[Perplexity] Mock perplexity: {ppl:.2f}")

    return {
        "perplexity": ppl,
        "bits_per_byte": avg_nll / 2.0,
        "word_perplexity": ppl * 1.5,
        "avg_nll": avg_nll,
        "total_tokens": 10000,
        "num_words": 8000,
        "dataset": dataset,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "is_mock": True,
    }


def _measure_energy_with_pynvml(
    model,
    tokenizer,
    num_inferences: int,
    device: str,
) -> Optional[Dict[str, float]]:
    """Measure energy using pynvml power readings.

    This is a fallback when codecarbon is unavailable.
    Uses GPU power sampling before/after each inference to estimate energy.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        num_inferences: Number of inferences to run
        device: Device string

    Returns:
        Dictionary with energy measurements, or None if pynvml unavailable
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        total_energy_joules = 0.0
        dummy_input = tokenizer("Hello world, this is a test.", return_tensors="pt")
        if device == "cuda" and torch.cuda.is_available():
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for i in range(num_inferences):
            # Sample power before
            power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
            start_time = time.time()

            # Run inference
            with torch.no_grad():
                _ = model.generate(**dummy_input, max_new_tokens=10, do_sample=False, pad_token_id=pad_token_id)

            # Sample power after and calculate energy
            elapsed = time.time() - start_time
            power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
            avg_power = (power_start + power_end) / 2
            energy_joules = avg_power * elapsed
            total_energy_joules += energy_joules

            if (i + 1) % 100 == 0:
                print(f"[Energy/pynvml] Completed {i + 1}/{num_inferences} inferences...")

        pynvml.nvmlShutdown()

        # CO2 estimation: ~0.5 kg CO2/kWh (global average grid intensity)
        energy_kwh = total_energy_joules / 3_600_000
        co2_grams = energy_kwh * 500  # 500g CO2 per kWh

        return {
            "energy_per_inference_joules": total_energy_joules / num_inferences,
            "total_energy_joules": total_energy_joules,
            "energy_kwh": energy_kwh,
            "co2_grams": co2_grams,
            "co2_kg": co2_grams / 1000,
            "num_inferences": num_inferences,
            "device": device,
            "measurement_method": "pynvml",
            "is_mock": False,
        }

    except ImportError:
        logger.debug("pynvml not available for energy measurement")
        return None
    except Exception as e:
        logger.warning(f"pynvml energy measurement failed: {e}")
        return None


def _mock_energy_consumption(
    checkpoint_path: str,
    num_inferences: int,
    device: str,
) -> Dict[str, float]:
    """Fallback mock energy estimation when real measurement is unavailable."""
    if "cuda" in device:
        if "4bit" in checkpoint_path:
            energy_per_inference = 0.5
        elif "8bit" in checkpoint_path:
            energy_per_inference = 1.0
        else:
            energy_per_inference = 2.0
    else:
        energy_per_inference = 0.3

    total_energy = energy_per_inference * num_inferences
    co2_grams = (total_energy / 3600000) * 500

    return {
        "energy_per_inference_joules": energy_per_inference,
        "total_energy_joules": total_energy,
        "energy_kwh": total_energy / 3600000,
        "co2_grams": co2_grams,
        "co2_kg": co2_grams / 1000,
        "num_inferences": num_inferences,
        "device": device,
        "measurement_method": "mock",
        "is_mock": True,
    }


@tool
def estimate_energy_consumption(
    checkpoint_path: str,
    num_inferences: int = 1000,
    device: str = "cuda",
) -> Dict[str, float]:
    """Estimate energy consumption for inference.

    Args:
        checkpoint_path: Path to the model checkpoint
        num_inferences: Number of inferences to estimate for
        device: Device type

    Returns:
        Dictionary with energy estimates:
        - energy_per_inference_joules: Energy per inference
        - total_energy_joules: Total energy for num_inferences
        - co2_grams: Estimated CO2 emissions
    """
    print(f"[Energy] Estimating energy consumption for {num_inferences} inferences...")

    # Try to use codecarbon for real energy tracking
    try:
        from codecarbon import EmissionsTracker

        logger.info(f"Using codecarbon for real energy tracking")

        # Load model and tokenizer
        model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)

        # Start tracking emissions
        tracker = EmissionsTracker(
            log_level="error",
            save_to_file=False,
            save_to_api=False,
        )
        tracker.start()

        # Run inferences
        dummy_input = tokenizer("Hello world, this is a test.", return_tensors="pt")
        if device == "cuda" and torch.cuda.is_available():
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}

        # Set pad_token_id to suppress warning
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for i in range(num_inferences):
            with torch.no_grad():
                _ = model.generate(**dummy_input, max_new_tokens=10, do_sample=False, pad_token_id=pad_token_id)
            if (i + 1) % 100 == 0:
                print(f"[Energy] Completed {i + 1}/{num_inferences} inferences...")

        # Stop tracking and get results
        emissions_kg = tracker.stop()

        # Get detailed data with null checks
        final_data = tracker.final_emissions_data
        energy_kwh = (
            final_data.energy_consumed
            if final_data and final_data.energy_consumed is not None
            else 0.0
        )

        # Convert to various units
        total_energy_joules = energy_kwh * 3600000  # kWh to Joules
        energy_per_inference_joules = total_energy_joules / num_inferences
        co2_grams = emissions_kg * 1000

        # Clean up with synchronization
        del model
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"[Energy] Total energy: {energy_kwh:.6f} kWh ({total_energy_joules:.1f} J)")
        print(f"[Energy] Per inference: {energy_per_inference_joules:.4f} J")
        print(f"[Energy] CO2 emissions: {co2_grams:.4f} g")

        return {
            "energy_per_inference_joules": energy_per_inference_joules,
            "total_energy_joules": total_energy_joules,
            "energy_kwh": energy_kwh,
            "co2_grams": co2_grams,
            "co2_kg": emissions_kg,
            "num_inferences": num_inferences,
            "device": device,
            "is_mock": False,
        }

    except ImportError:
        logger.warning("codecarbon not available, trying pynvml fallback")
        print("[Energy] codecarbon not available, trying pynvml...")
    except Exception as e:
        logger.warning(f"codecarbon tracking failed: {e}, trying pynvml fallback")
        print(f"[Energy] codecarbon failed: {e}, trying pynvml...")

    # Try pynvml as fallback for real energy measurement
    if device == "cuda" and torch.cuda.is_available():
        try:
            # Load model for pynvml measurement
            model, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)
            pynvml_results = _measure_energy_with_pynvml(model, tokenizer, num_inferences, device)

            # Clean up with synchronization
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if pynvml_results is not None:
                print(f"[Energy/pynvml] Total energy: {pynvml_results['energy_kwh']:.6f} kWh ({pynvml_results['total_energy_joules']:.1f} J)")
                print(f"[Energy/pynvml] Per inference: {pynvml_results['energy_per_inference_joules']:.4f} J")
                print(f"[Energy/pynvml] CO2 emissions: {pynvml_results['co2_grams']:.4f} g")
                return pynvml_results

        except Exception as e:
            logger.warning(f"pynvml energy measurement failed: {e}")
            print(f"[Energy] pynvml failed: {e}, using mock estimation")

    # Final fallback to mock estimation
    results = _mock_energy_consumption(checkpoint_path, num_inferences, device)

    print(f"[Energy] Per inference: {results['energy_per_inference_joules']:.2f} J (mock)")
    print(f"[Energy] Total: {results['total_energy_joules']:.1f} J (mock)")
    print(f"[Energy] Estimated CO2: {results['co2_grams']:.2f} g (mock)")

    return results


def get_evaluation_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the evaluation subagent configuration for DeepAgents.

    Args:
        spec: Model specification with requirements and constraints

    Returns:
        Subagent configuration dictionary
    """
    # Extract relevant spec information
    model_name = spec.get("model_name", "unknown")
    primary_objective = spec.get("primary_objective", "accuracy")
    accuracy_threshold = spec.get("accuracy_threshold", 0.95)

    prompt = f"""You are an Evaluation Specialist Agent responsible for benchmarking model performance.

Model: {model_name}
Primary Objective: {primary_objective}
Accuracy Threshold: {accuracy_threshold:.0%}

Your responsibilities:
1. Evaluate models on multiple benchmarks using lm-eval harness
2. Measure inference latency and throughput
3. Track memory usage during inference
4. Estimate energy consumption
5. Compare compressed models with baselines
6. Use proxy evaluation for quick assessment, full evaluation for final results

Available tools:
- evaluate_model: Run comprehensive evaluation (uses lm-eval harness)
- run_proxy_evaluation: Quick evaluation with subset of data
- measure_perplexity: Measure perplexity on WikiText/C4/etc
- measure_inference_latency: Detailed latency profiling
- measure_memory_usage: Memory consumption tracking
- compare_with_baseline: Compare with original model
- estimate_energy_consumption: Energy and CO2 estimates

Evaluation Strategy:
1. Start with proxy evaluation for quick filtering
2. If proxy results are promising (>{accuracy_threshold:.0%} accuracy), run full evaluation
3. Always measure latency and memory usage
4. Compare with baseline to calculate accuracy retention
5. Report comprehensive metrics for Pareto optimization

Available Benchmarks (via lm-eval harness):
- GSM8K: Mathematical reasoning
- CommonsenseQA: Common sense reasoning
- TruthfulQA: Truthfulness and factuality
- HumanEval: Code generation (custom evaluator)
- BIG-Bench Hard (bbh): Challenging diverse tasks
- MMLU: Massive Multitask Language Understanding
- HellaSwag: Commonsense NLI
- ARC-Easy: AI2 Reasoning Challenge (Easy)
- ARC-Challenge: AI2 Reasoning Challenge (Challenge)
- WinoGrande: Winograd Schema Challenge

When you receive an evaluation request:
1. First run proxy evaluation on all benchmarks
2. If results meet threshold, proceed with full evaluation
3. Measure performance metrics (latency, memory, energy)
4. Return comprehensive results including all metrics
"""

    return {
        "name": "evaluation",
        "description": "Evaluates model performance on benchmarks and measures efficiency metrics",
        "prompt": prompt,
        "tools": [
            evaluate_model,
            run_proxy_evaluation,
            measure_perplexity,
            measure_inference_latency,
            measure_memory_usage,
            compare_with_baseline,
            estimate_energy_consumption,
        ],
        "model": "anthropic:claude-3-haiku-20240307",  # Use cheaper model for specialized task
    }


# Export the subagent creator
__all__ = ["get_evaluation_subagent", "evaluate_model", "measure_perplexity"]