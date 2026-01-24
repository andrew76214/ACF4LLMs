"""GPU memory management and CUDA error handling utilities.

This module provides centralized utilities for:
- Thorough GPU memory cleanup with synchronization
- Memory availability checks before operations
- CUDA error type detection and recovery
- Context managers for safe GPU operations
"""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Tuple

import torch

logger = logging.getLogger(__name__)


def robust_gpu_cleanup(force_gc: bool = True, reset_peak_stats: bool = False) -> Dict[str, Any]:
    """Perform thorough GPU memory cleanup with synchronization.

    This function ensures complete GPU memory cleanup between operations,
    preventing memory fragmentation and leaks between episodes.

    Args:
        force_gc: Whether to run Python garbage collection
        reset_peak_stats: Whether to reset peak memory statistics

    Returns:
        Dictionary with memory statistics before and after cleanup:
        - success: Whether cleanup completed successfully
        - cuda_available: Whether CUDA is available
        - before_allocated_gb: Allocated memory before cleanup (GB)
        - before_reserved_gb: Reserved memory before cleanup (GB)
        - after_allocated_gb: Allocated memory after cleanup (GB)
        - after_reserved_gb: Reserved memory after cleanup (GB)
        - freed_gb: Amount of memory freed (GB)
    """
    result: Dict[str, Any] = {"success": False, "cuda_available": torch.cuda.is_available()}

    if not torch.cuda.is_available():
        return result

    try:
        # Record before stats
        before_allocated = torch.cuda.memory_allocated() / (1024**3)
        before_reserved = torch.cuda.memory_reserved() / (1024**3)

        # Python GC first to release Python references
        if force_gc:
            gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure operations complete
        torch.cuda.synchronize()

        # Reset peak stats if requested
        if reset_peak_stats:
            torch.cuda.reset_peak_memory_stats()

        # Record after stats
        after_allocated = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)

        result.update({
            "success": True,
            "before_allocated_gb": before_allocated,
            "before_reserved_gb": before_reserved,
            "after_allocated_gb": after_allocated,
            "after_reserved_gb": after_reserved,
            "freed_gb": before_reserved - after_reserved,
        })

        if result["freed_gb"] > 0.1:
            logger.info(f"GPU cleanup freed {result['freed_gb']:.2f} GB")

    except Exception as e:
        logger.error(f"GPU cleanup failed: {e}")
        result["error"] = str(e)

    return result


def check_gpu_memory_available(required_gb: float, device: int = 0) -> Tuple[bool, float]:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Required memory in gigabytes
        device: CUDA device index

    Returns:
        Tuple of (is_available, available_gb)
    """
    if not torch.cuda.is_available():
        return False, 0.0

    try:
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        available = total - allocated

        return available >= required_gb, available
    except Exception as e:
        logger.error(f"Failed to check GPU memory: {e}")
        return False, 0.0


def get_gpu_memory_stats(device: int = 0) -> Dict[str, float]:
    """Get detailed GPU memory statistics.

    Args:
        device: CUDA device index

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "total_gb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0,
        }

    try:
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        return {
            "cuda_available": True,
            "device_name": props.name,
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - allocated,
            "fragmentation_ratio": (reserved - allocated) / reserved if reserved > 0 else 0.0,
        }
    except Exception as e:
        logger.error(f"Failed to get GPU memory stats: {e}")
        return {"cuda_available": False, "error": str(e)}


def handle_cuda_error(error: Exception) -> Tuple[bool, str]:
    """Identify CUDA error type and determine if recoverable.

    Args:
        error: The exception to analyze

    Returns:
        Tuple of (is_recoverable, error_category):
        - is_recoverable: True if the error can potentially be recovered from
        - error_category: Type of error (oom, illegal_memory_access, device_assert, cuda_generic, unknown)
    """
    error_str = str(error).lower()

    if "out of memory" in error_str:
        return True, "oom"
    elif "illegal memory access" in error_str:
        return False, "illegal_memory_access"
    elif "device-side assert" in error_str:
        return False, "device_assert"
    elif "cublas" in error_str:
        return True, "cublas"
    elif "cudnn" in error_str:
        return True, "cudnn"
    elif "cuda" in error_str:
        return True, "cuda_generic"
    else:
        return True, "unknown"


@contextmanager
def gpu_memory_context(
    cleanup_on_exit: bool = True,
    cleanup_on_enter: bool = False,
    synchronize: bool = True,
) -> Generator[None, None, None]:
    """Context manager for GPU operations with automatic cleanup.

    Ensures proper cleanup of GPU memory when operations complete or fail.

    Args:
        cleanup_on_exit: Whether to clean up GPU memory on exit
        cleanup_on_enter: Whether to clean up GPU memory before entering
        synchronize: Whether to synchronize CUDA after cleanup

    Yields:
        None

    Example:
        with gpu_memory_context():
            model = load_model()
            result = model.generate(...)
        # Memory automatically cleaned up here
    """
    if cleanup_on_enter and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()

    try:
        yield
    finally:
        if cleanup_on_exit and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()


class CUDAErrorRecovery:
    """Utility class for recovering from CUDA errors."""

    @staticmethod
    def attempt_oom_recovery() -> bool:
        """Attempt to recover from OOM by aggressive cleanup.

        Performs multiple cleanup passes to maximize freed memory.

        Returns:
            True if recovery likely succeeded (freed significant memory)
        """
        if not torch.cuda.is_available():
            return False

        try:
            # Multiple cleanup passes
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()

            torch.cuda.synchronize()

            # Check if we freed memory
            available = (torch.cuda.get_device_properties(0).total_memory -
                        torch.cuda.memory_allocated()) / (1024**3)

            success = available > 0.5  # At least 0.5GB free
            if success:
                logger.info(f"OOM recovery successful, {available:.2f} GB now available")
            else:
                logger.warning(f"OOM recovery may not have freed enough memory, only {available:.2f} GB available")

            return success
        except Exception as e:
            logger.error(f"OOM recovery failed: {e}")
            return False

    @staticmethod
    def reset_cuda_state() -> bool:
        """Reset CUDA state after fatal error.

        Note: Full reset may require process restart for some errors.

        Returns:
            True if reset completed without errors
        """
        if not torch.cuda.is_available():
            return False

        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Reset accumulated error state
            torch.cuda.reset_accumulated_memory_stats()

            logger.info("CUDA state reset completed")
            return True
        except Exception as e:
            logger.error(f"CUDA state reset failed: {e}")
            return False

    @staticmethod
    def check_cuda_health() -> Dict[str, Any]:
        """Check CUDA device health and detect potential issues.

        Returns:
            Dictionary with health information:
            - cuda_available: Whether CUDA is available
            - healthy: Whether the device appears healthy
            - issues: List of detected issues
            - recommendations: List of recommendations
        """
        health: Dict[str, Any] = {
            "cuda_available": False,
            "healthy": False,
            "issues": [],
            "recommendations": [],
        }

        if not torch.cuda.is_available():
            health["issues"].append("CUDA not available")
            return health

        health["cuda_available"] = True

        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = props.total_memory / (1024**3)

                # Check fragmentation (high ratio of reserved but unallocated memory)
                if reserved > 0 and (reserved - allocated) / reserved > 0.3:
                    health["issues"].append(f"GPU {i}: High memory fragmentation ({(reserved - allocated):.2f} GB reserved but unallocated)")
                    health["recommendations"].append("Run robust_gpu_cleanup() to reduce fragmentation")

                # Check available memory
                if total - allocated < 1.0:
                    health["issues"].append(f"GPU {i}: Low available memory ({(total - allocated):.2f} GB)")
                    health["recommendations"].append("Consider using smaller batch sizes or model offloading")

                # Check utilization
                utilization = allocated / total if total > 0 else 0
                if utilization > 0.95:
                    health["issues"].append(f"GPU {i}: Very high memory utilization ({utilization:.0%})")

            # Test CUDA responsiveness with a simple operation
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            torch.cuda.synchronize()

            health["healthy"] = len(health["issues"]) == 0

        except Exception as e:
            health["issues"].append(f"Health check failed: {e}")
            health["healthy"] = False

        return health


def estimate_operation_vram(
    model_size_gb: float,
    operation: str = "inference",
    batch_size: int = 1,
    sequence_length: int = 2048,
) -> float:
    """Estimate VRAM required for an operation.

    Args:
        model_size_gb: Model size in GB
        operation: Type of operation (inference, quantization, training)
        batch_size: Batch size for the operation
        sequence_length: Sequence length for transformers

    Returns:
        Estimated VRAM requirement in GB
    """
    # Base multipliers for different operations
    multipliers = {
        "inference": 1.3,  # Model + activations + KV cache
        "quantization": 2.5,  # Model + calibration + output
        "training": 4.0,  # Model + optimizer + gradients
        "evaluation": 1.5,  # Model + activations
        "lora": 1.5,  # Quantized base + adapter gradients
        "qlora": 0.5,  # 4-bit base + adapter gradients
    }

    base_multiplier = multipliers.get(operation, 2.0)

    # Adjust for batch size and sequence length
    batch_factor = 1 + (batch_size - 1) * 0.1
    seq_factor = sequence_length / 2048

    estimated = model_size_gb * base_multiplier * batch_factor * seq_factor

    # Add safety margin
    estimated *= 1.1

    return estimated


__all__ = [
    "robust_gpu_cleanup",
    "check_gpu_memory_available",
    "get_gpu_memory_stats",
    "handle_cuda_error",
    "gpu_memory_context",
    "CUDAErrorRecovery",
    "estimate_operation_vram",
]
