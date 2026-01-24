"""Latency and performance evaluator."""

import torch
import time
import psutil
import numpy as np
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class LatencyEvaluator:
    """Evaluator for model latency and performance metrics."""

    def __init__(self, device: str = "cuda"):
        """Initialize latency evaluator.

        Args:
            device: Device to run evaluation on
        """
        self.device = device

    def evaluate_performance(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int = 1,
        sequence_length: int = 512,
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        """Evaluate model performance metrics.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size for inference
            sequence_length: Input sequence length
            num_iterations: Number of iterations for averaging

        Returns:
            Dictionary with performance metrics
        """
        # Warm up
        self._warmup(model, tokenizer, batch_size, sequence_length)

        # Measure latency
        latencies = []
        memory_usage = []

        for _ in range(num_iterations):
            # Create dummy input
            dummy_text = "The quick brown fox jumps over the lazy dog. " * (sequence_length // 10)
            inputs = tokenizer(
                [dummy_text] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=sequence_length,
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
                torch.cuda.synchronize()

            # Measure memory before
            if self.device == "cuda":
                mem_before = torch.cuda.memory_allocated() / (1024**3)
            else:
                mem_before = psutil.Process().memory_info().rss / (1024**3)

            # Measure inference time
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                )

            if self.device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Measure memory after
            if self.device == "cuda":
                mem_after = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
            else:
                mem_after = psutil.Process().memory_info().rss / (1024**3)

            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(mem_after - mem_before)

        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p90_latency = np.percentile(latencies, 90)
        p99_latency = np.percentile(latencies, 99)

        # Calculate throughput
        tokens_generated = 50 * batch_size  # max_new_tokens * batch_size
        throughput = tokens_generated / (mean_latency / 1000)  # tokens per second

        # Memory statistics
        peak_memory = max(memory_usage) if memory_usage else 0

        results = {
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "p50_latency_ms": p50_latency,
            "p90_latency_ms": p90_latency,
            "p99_latency_ms": p99_latency,
            "throughput_tokens_per_sec": throughput,
            "peak_memory_gb": peak_memory,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_iterations": num_iterations,
            "device": self.device,
        }

        logger.info(f"Performance: {mean_latency:.1f}ms latency, {throughput:.1f} tokens/s")

        return results

    def measure_model_size(self, model: Any) -> Dict[str, float]:
        """Measure model size and parameter count.

        Args:
            model: Model to measure

        Returns:
            Dictionary with size metrics
        """
        total_params = 0
        trainable_params = 0
        total_size_bytes = 0

        for name, param in model.named_parameters():
            params = param.numel()
            total_params += params

            if param.requires_grad:
                trainable_params += params

            # Calculate size in bytes
            if param.dtype == torch.float32:
                bytes_per_param = 4
            elif param.dtype == torch.float16:
                bytes_per_param = 2
            elif param.dtype == torch.int8:
                bytes_per_param = 1
            else:
                bytes_per_param = 4  # Default

            total_size_bytes += params * bytes_per_param

        total_size_gb = total_size_bytes / (1024**3)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_gb": total_size_gb,
            "parameters_millions": total_params / 1e6,
        }

    def profile_memory_usage(
        self, model: Any, tokenizer: Any, batch_sizes: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Any]:
        """Profile memory usage for different batch sizes.

        Args:
            model: Model to profile
            tokenizer: Tokenizer
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with memory profiling results
        """
        results = {}

        for batch_size in batch_sizes:
            try:
                # Create input
                dummy_text = "Test input " * 50
                inputs = tokenizer(
                    [dummy_text] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    torch.cuda.reset_peak_memory_stats()

                # Run inference
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=50)

                # Get memory usage
                if self.device == "cuda":
                    memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                else:
                    memory_gb = psutil.Process().memory_info().rss / (1024**3)

                results[f"batch_{batch_size}"] = memory_gb

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f"batch_{batch_size}"] = "OOM"
                    break
                else:
                    raise e

        # Find maximum batch size
        max_batch = 0
        for batch_size in batch_sizes:
            if f"batch_{batch_size}" in results and results[f"batch_{batch_size}"] != "OOM":
                max_batch = batch_size

        results["max_batch_size"] = max_batch

        return results

    def _warmup(
        self, model: Any, tokenizer: Any, batch_size: int, sequence_length: int, iterations: int = 3
    ):
        """Warm up model for accurate measurements.

        Args:
            model: Model to warm up
            tokenizer: Tokenizer
            batch_size: Batch size
            sequence_length: Sequence length
            iterations: Number of warmup iterations
        """
        dummy_text = "Warmup text " * (sequence_length // 10)
        inputs = tokenizer(
            [dummy_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sequence_length,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        for _ in range(iterations):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)

        if self.device == "cuda":
            torch.cuda.synchronize()

    def compare_latency(
        self, baseline_model: Any, compressed_model: Any, tokenizer: Any
    ) -> Dict[str, Any]:
        """Compare latency between two models.

        Args:
            baseline_model: Baseline model
            compressed_model: Compressed model
            tokenizer: Tokenizer

        Returns:
            Dictionary with comparison results
        """
        # Evaluate both models
        baseline_perf = self.evaluate_performance(baseline_model, tokenizer)
        compressed_perf = self.evaluate_performance(compressed_model, tokenizer)

        # Calculate improvements (with zero-division protection)
        compressed_latency = compressed_perf["mean_latency_ms"]
        baseline_memory = baseline_perf["peak_memory_gb"]
        baseline_throughput = baseline_perf["throughput_tokens_per_sec"]

        speedup = (
            baseline_perf["mean_latency_ms"] / compressed_latency
            if compressed_latency > 0 else 0.0
        )
        memory_reduction = (
            1 - (compressed_perf["peak_memory_gb"] / baseline_memory)
            if baseline_memory > 0 else 0.0
        )
        throughput_improvement = (
            compressed_perf["throughput_tokens_per_sec"] / baseline_throughput
            if baseline_throughput > 0 else 0.0
        )

        return {
            "baseline_latency_ms": baseline_perf["mean_latency_ms"],
            "compressed_latency_ms": compressed_perf["mean_latency_ms"],
            "speedup": speedup,
            "baseline_memory_gb": baseline_perf["peak_memory_gb"],
            "compressed_memory_gb": compressed_perf["peak_memory_gb"],
            "memory_reduction": memory_reduction,
            "baseline_throughput": baseline_perf["throughput_tokens_per_sec"],
            "compressed_throughput": compressed_perf["throughput_tokens_per_sec"],
            "throughput_improvement": throughput_improvement,
        }