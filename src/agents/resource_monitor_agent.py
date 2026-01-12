"""Resource Monitor Agent for GPU/VRAM monitoring and resource management."""

import os
import psutil
import time
import json
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


def _estimate_model_size_gb(model_name: str) -> float:
    """Estimate model size in GB from model name."""
    import re

    MODEL_SIZE_DATABASE = {
        "gpt2": 0.5, "gpt2-medium": 1.5, "gpt2-large": 3.0, "gpt2-xl": 6.0,
        "facebook/opt-125m": 0.25, "facebook/opt-350m": 0.7, "facebook/opt-1.3b": 2.6,
        "facebook/opt-2.7b": 5.4, "facebook/opt-6.7b": 13.4, "facebook/opt-13b": 26.0,
        "meta-llama/Meta-Llama-3-8B": 16.0, "meta-llama/Llama-2-7b-hf": 13.5,
        "mistralai/Mistral-7B-v0.1": 13.5, "Qwen/Qwen-7B": 14.0,
    }

    if model_name in MODEL_SIZE_DATABASE:
        return MODEL_SIZE_DATABASE[model_name]

    model_lower = model_name.lower()
    for key, size in MODEL_SIZE_DATABASE.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return size

    patterns = [
        (r'(\d+\.?\d*)B', lambda x: float(x) * 2),
        (r'(\d+\.?\d*)b', lambda x: float(x) * 2),
        (r'(\d+)M', lambda x: float(x) / 1000 * 2),
        (r'(\d+)m', lambda x: float(x) / 1000 * 2),
    ]

    for pattern, converter in patterns:
        match = re.search(pattern, model_name)
        if match:
            return converter(match.group(1))

    return 1.0


@tool
def check_gpu_availability() -> Dict[str, Any]:
    """Check available GPUs and their current status.

    Returns:
        Dictionary with GPU information
    """
    print("[GPU] Checking GPU availability...")

    gpu_info = {
        "cuda_available": False,
        "num_gpus": 0,
        "gpus": [],
        "total_vram_gb": 0,
        "available_vram_gb": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["num_gpus"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory / (1024**3)

                # Get current memory usage
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                available = total_vram - allocated

                gpu_data = {
                    "id": i,
                    "name": props.name,
                    "total_vram_gb": total_vram,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "available_gb": available,
                    "utilization": (allocated / total_vram * 100) if total_vram > 0 else 0,
                    "compute_capability": f"{props.major}.{props.minor}",
                }

                gpu_info["gpus"].append(gpu_data)
                gpu_info["total_vram_gb"] += total_vram
                gpu_info["available_vram_gb"] += available

                print(f"[GPU] GPU {i}: {props.name} - {available:.1f}/{total_vram:.1f} GB available")

        # Try nvidia-ml-py for more detailed info
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to W

                if i < len(gpu_info["gpus"]):
                    gpu_info["gpus"][i].update({
                        "gpu_utilization": util.gpu,
                        "memory_utilization": util.memory,
                        "temperature_c": temp,
                        "power_w": power,
                    })

            pynvml.nvmlShutdown()
        except:
            pass  # nvidia-ml-py not available

    except ImportError:
        print("[GPU] CUDA not available, checking CPU resources only")
        gpu_info["cuda_available"] = False

    # Add CPU info as fallback
    cpu_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "ram_percent": psutil.virtual_memory().percent,
    }

    gpu_info["cpu_info"] = cpu_info

    return gpu_info


@tool
def monitor_resource_usage(
    duration_seconds: int = 60,
    interval_seconds: int = 5,
) -> Dict[str, Any]:
    """Monitor resource usage over time.

    Args:
        duration_seconds: How long to monitor
        interval_seconds: Sampling interval

    Returns:
        Dictionary with resource usage statistics
    """
    print(f"[Monitor] Monitoring resources for {duration_seconds}s...")

    samples = []
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        sample = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_gb": psutil.virtual_memory().used / (1024**3),
            "ram_percent": psutil.virtual_memory().percent,
        }

        # Add GPU metrics if available
        try:
            import torch

            if torch.cuda.is_available():
                sample["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
                sample["gpu_utilization"] = 0  # Would need nvidia-ml-py for real util

                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sample["gpu_utilization"] = util.gpu
                    pynvml.nvmlShutdown()
                except:
                    pass
        except:
            pass

        samples.append(sample)
        time.sleep(interval_seconds)

    # Calculate statistics
    if samples:
        cpu_values = [s["cpu_percent"] for s in samples]
        ram_values = [s["ram_gb"] for s in samples]

        stats = {
            "duration_seconds": time.time() - start_time,
            "num_samples": len(samples),
            "cpu": {
                "mean": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "ram_gb": {
                "mean": sum(ram_values) / len(ram_values),
                "max": max(ram_values),
                "min": min(ram_values),
            },
            "samples": samples,
        }

        if "gpu_memory_gb" in samples[0]:
            gpu_values = [s["gpu_memory_gb"] for s in samples]
            stats["gpu_memory_gb"] = {
                "mean": sum(gpu_values) / len(gpu_values),
                "max": max(gpu_values),
                "min": min(gpu_values),
            }

        print(f"[Monitor] Collected {len(samples)} samples")
        print(f"[Monitor] CPU: {stats['cpu']['mean']:.1f}% avg")
        print(f"[Monitor] RAM: {stats['ram_gb']['mean']:.1f} GB avg")

        return stats

    return {"error": "No samples collected"}


@tool
def estimate_compression_resources(
    model_name: str,
    compression_method: str,
    model_size_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate resource requirements for compression.

    Args:
        model_name: Model to compress
        compression_method: Compression method to use
        model_size_gb: Model size in GB (auto-estimated if not provided)

    Returns:
        Dictionary with resource estimates
    """
    if model_size_gb is None:
        model_size_gb = _estimate_model_size_gb(model_name)
    print(f"[Estimate] Estimating resources for {compression_method} on {model_size_gb:.1f}GB model")

    # Base estimates
    estimates = {
        "model_name": model_name,
        "model_size_gb": model_size_gb,
        "compression_method": compression_method,
    }

    # Method-specific multipliers
    method_requirements = {
        "quantization": {
            "autoround": {"vram_multiplier": 2.5, "time_hours": 0.5, "cpu_intensive": False},
            "gptq": {"vram_multiplier": 2.0, "time_hours": 0.3, "cpu_intensive": False},
            "awq": {"vram_multiplier": 2.0, "time_hours": 0.4, "cpu_intensive": False},
            "int8": {"vram_multiplier": 1.5, "time_hours": 0.1, "cpu_intensive": False},
        },
        "pruning": {
            "structured": {"vram_multiplier": 1.8, "time_hours": 0.3, "cpu_intensive": True},
            "unstructured": {"vram_multiplier": 1.5, "time_hours": 0.5, "cpu_intensive": True},
            "sparsegpt": {"vram_multiplier": 2.0, "time_hours": 1.0, "cpu_intensive": False},
        },
        "distillation": {
            "standard": {"vram_multiplier": 3.0, "time_hours": 2.0, "cpu_intensive": False},
            "progressive": {"vram_multiplier": 2.5, "time_hours": 4.0, "cpu_intensive": False},
        },
        "finetuning": {
            "lora": {"vram_multiplier": 1.5, "time_hours": 1.0, "cpu_intensive": False},
            "qlora": {"vram_multiplier": 0.8, "time_hours": 1.5, "cpu_intensive": False},
            "full": {"vram_multiplier": 4.0, "time_hours": 3.0, "cpu_intensive": False},
        },
    }

    # Find matching method
    for category, methods in method_requirements.items():
        for method, reqs in methods.items():
            if method in compression_method.lower():
                estimates["vram_required_gb"] = model_size_gb * reqs["vram_multiplier"]
                estimates["vram_recommended_gb"] = estimates["vram_required_gb"] * 1.2
                estimates["estimated_time_hours"] = reqs["time_hours"] * (model_size_gb / 16)
                estimates["cpu_intensive"] = reqs["cpu_intensive"]
                estimates["method_category"] = category
                break

    # Default if no match
    if "vram_required_gb" not in estimates:
        estimates["vram_required_gb"] = model_size_gb * 2.0
        estimates["vram_recommended_gb"] = model_size_gb * 2.5
        estimates["estimated_time_hours"] = 1.0
        estimates["cpu_intensive"] = False
        estimates["method_category"] = "unknown"

    # Check current resources
    current = check_gpu_availability()
    estimates["current_vram_available_gb"] = current.get("available_vram_gb", 0)
    estimates["sufficient_vram"] = estimates["current_vram_available_gb"] >= estimates["vram_required_gb"]

    if not estimates["sufficient_vram"]:
        estimates["vram_deficit_gb"] = estimates["vram_required_gb"] - estimates["current_vram_available_gb"]
        print(f"[Estimate] ⚠️  Insufficient VRAM: need {estimates['vram_deficit_gb']:.1f} GB more")
    else:
        print(f"[Estimate] ✅ Sufficient VRAM available")

    return estimates


@tool
def preflight_check(
    model_name: str,
    compression_strategy: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pre-flight checks before compression.

    Args:
        model_name: Model to compress
        compression_strategy: Strategy configuration

    Returns:
        Dictionary with preflight check results
    """
    print("[Preflight] Running pre-flight checks...")

    checks = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "strategy": compression_strategy,
        "checks_passed": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # Check GPU availability
    gpu_info = check_gpu_availability()
    checks["gpu_available"] = gpu_info["cuda_available"]
    checks["num_gpus"] = gpu_info["num_gpus"]
    checks["total_vram_gb"] = gpu_info.get("total_vram_gb", 0)

    if not gpu_info["cuda_available"]:
        checks["errors"].append("No CUDA-capable GPU detected")
        checks["checks_passed"] = False

    # Check VRAM requirements
    method = compression_strategy.get("method", "quantization")
    model_size = compression_strategy.get("model_size_gb") or _estimate_model_size_gb(model_name)

    resource_est = estimate_compression_resources(model_name, method, model_size)
    checks["vram_required_gb"] = resource_est["vram_required_gb"]
    checks["vram_available_gb"] = resource_est["current_vram_available_gb"]

    if not resource_est["sufficient_vram"]:
        checks["errors"].append(
            f"Insufficient VRAM: {resource_est['vram_required_gb']:.1f} GB required, "
            f"{resource_est['current_vram_available_gb']:.1f} GB available"
        )
        checks["checks_passed"] = False
        checks["recommendations"].append("Consider using CPU-based methods or smaller batch sizes")

    # Check disk space
    disk_usage = psutil.disk_usage("/")
    free_gb = disk_usage.free / (1024**3)
    required_disk_gb = model_size * 3  # Original + compressed + workspace

    checks["disk_free_gb"] = free_gb
    checks["disk_required_gb"] = required_disk_gb

    if free_gb < required_disk_gb:
        checks["warnings"].append(
            f"Low disk space: {free_gb:.1f} GB free, {required_disk_gb:.1f} GB recommended"
        )

    # Check CPU/RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    checks["ram_total_gb"] = ram_gb

    if ram_gb < 32:
        checks["warnings"].append(f"Limited RAM: {ram_gb:.1f} GB may cause slowdowns")

    # Method-specific checks
    if "quantization" in method.lower():
        if compression_strategy.get("bits", 8) < 4:
            checks["warnings"].append("Very low bit quantization (<4) may significantly impact accuracy")

    if "pruning" in method.lower():
        if compression_strategy.get("sparsity", 0) > 0.8:
            checks["warnings"].append("High sparsity (>80%) usually requires fine-tuning")

    # Temperature check for GPUs
    if gpu_info["cuda_available"] and gpu_info["gpus"]:
        for gpu in gpu_info["gpus"]:
            if "temperature_c" in gpu and gpu["temperature_c"] > 80:
                checks["warnings"].append(
                    f"GPU {gpu['id']} running hot ({gpu['temperature_c']}°C), may throttle"
                )

    # Summary
    checks["ready_to_proceed"] = checks["checks_passed"] and len(checks["errors"]) == 0

    print(f"[Preflight] Checks passed: {checks['checks_passed']}")
    print(f"[Preflight] Warnings: {len(checks['warnings'])}")
    print(f"[Preflight] Errors: {len(checks['errors'])}")

    return checks


@tool
def optimize_batch_size(
    model_size_gb: float,
    available_vram_gb: float,
    sequence_length: int = 2048,
) -> Dict[str, Any]:
    """Calculate optimal batch size based on available resources.

    Args:
        model_size_gb: Model size in GB
        available_vram_gb: Available VRAM in GB
        sequence_length: Maximum sequence length

    Returns:
        Dictionary with batch size recommendations
    """
    print(f"[Optimize] Calculating optimal batch size for {available_vram_gb:.1f} GB VRAM")

    # Rough estimation of memory per batch
    # Model weights + activations + gradients
    base_memory = model_size_gb
    activation_memory_per_sample = (sequence_length * 4096 * 2) / (1024**3)  # Rough estimate

    # Calculate max batch size
    available_for_batch = available_vram_gb - base_memory - 2  # 2GB safety margin

    if available_for_batch <= 0:
        max_batch_size = 1
        recommended_batch_size = 1
        warning = "Very limited VRAM, consider CPU offloading"
    else:
        max_batch_size = int(available_for_batch / activation_memory_per_sample)
        recommended_batch_size = max(1, max_batch_size // 2)  # Conservative
        warning = None

    # Gradient accumulation if batch size too small
    gradient_accumulation_steps = 1
    if recommended_batch_size < 4:
        gradient_accumulation_steps = 4 // recommended_batch_size

    recommendations = {
        "model_size_gb": model_size_gb,
        "available_vram_gb": available_vram_gb,
        "sequence_length": sequence_length,
        "max_batch_size": max_batch_size,
        "recommended_batch_size": recommended_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": recommended_batch_size * gradient_accumulation_steps,
        "estimated_memory_usage_gb": base_memory + recommended_batch_size * activation_memory_per_sample,
        "warning": warning,
    }

    print(f"[Optimize] Recommended batch size: {recommended_batch_size}")
    if gradient_accumulation_steps > 1:
        print(f"[Optimize] With gradient accumulation: {recommendations['effective_batch_size']}")

    return recommendations


@tool
def clear_gpu_cache() -> Dict[str, Any]:
    """Clear GPU memory cache to free up VRAM.

    Returns:
        Dictionary with cleared memory info
    """
    print("[Cache] Clearing GPU cache...")

    before_free = 0
    after_free = 0

    try:
        import torch

        if torch.cuda.is_available():
            # Get before stats
            before_allocated = torch.cuda.memory_allocated() / (1024**3)
            before_reserved = torch.cuda.memory_reserved() / (1024**3)

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Get after stats
            after_allocated = torch.cuda.memory_allocated() / (1024**3)
            after_reserved = torch.cuda.memory_reserved() / (1024**3)

            freed_gb = (before_reserved - after_reserved)

            print(f"[Cache] Freed {freed_gb:.2f} GB of GPU memory")

            return {
                "success": True,
                "before_allocated_gb": before_allocated,
                "before_reserved_gb": before_reserved,
                "after_allocated_gb": after_allocated,
                "after_reserved_gb": after_reserved,
                "freed_gb": freed_gb,
            }
    except Exception as e:
        print(f"[Cache] Error clearing cache: {e}")

    return {"success": False, "error": str(e) if 'e' in locals() else "No GPU available"}


def get_resource_monitor_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the resource monitor subagent configuration.

    Args:
        spec: Model specification

    Returns:
        Subagent configuration dictionary
    """
    model_name = spec.get("model_name", "unknown")
    model_size = spec.get("model_size_gb") or _estimate_model_size_gb(model_name)

    prompt = f"""You are a Resource Monitor Agent responsible for system resource management.

Model: {model_name}
Model Size: {model_size:.1f} GB

Your responsibilities:
1. Monitor GPU/CPU/RAM usage
2. Run pre-flight checks before compression
3. Estimate resource requirements
4. Optimize batch sizes for available resources
5. Clear GPU cache when needed
6. Prevent out-of-memory errors

Available tools:
- check_gpu_availability: Get current GPU status
- monitor_resource_usage: Track resources over time
- estimate_compression_resources: Estimate requirements
- preflight_check: Verify readiness for compression
- optimize_batch_size: Calculate optimal batch size
- clear_gpu_cache: Free up GPU memory

Monitoring Strategy:
1. Always run preflight checks before compression
2. Monitor resources during long operations
3. Clear cache if VRAM is tight
4. Adjust batch sizes based on available memory
5. Warn about potential bottlenecks

Resource Guidelines:
- Minimum VRAM: Model size × 1.5
- Recommended VRAM: Model size × 2.5
- For training: Model size × 4
- Keep 2GB VRAM buffer for stability

When you receive a monitoring request:
1. Check current resource availability
2. Estimate requirements for the operation
3. Run preflight checks
4. Optimize settings if needed
5. Monitor during execution
"""

    return {
        "name": "resource_monitor",
        "description": "Monitors and manages system resources for compression tasks",
        "prompt": prompt,
        "tools": [
            check_gpu_availability,
            monitor_resource_usage,
            estimate_compression_resources,
            preflight_check,
            optimize_batch_size,
            clear_gpu_cache,
        ],
        "model": "anthropic:claude-3-haiku-20240307",
    }


__all__ = ["get_resource_monitor_subagent", "check_gpu_availability", "preflight_check"]