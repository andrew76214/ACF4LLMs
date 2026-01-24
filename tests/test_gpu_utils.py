"""Tests for GPU memory management utilities."""

import gc
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.common.gpu_utils import (
    robust_gpu_cleanup,
    check_gpu_memory_available,
    get_gpu_memory_stats,
    handle_cuda_error,
    gpu_memory_context,
    CUDAErrorRecovery,
    estimate_operation_vram,
)


class TestRobustGpuCleanup(unittest.TestCase):
    """Tests for robust_gpu_cleanup function."""

    def test_cleanup_when_cuda_unavailable(self):
        """Test cleanup gracefully handles no CUDA."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = robust_gpu_cleanup()
            self.assertFalse(result['success'])
            self.assertFalse(result['cuda_available'])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cleanup_with_cuda(self):
        """Test cleanup works with CUDA available."""
        # Allocate some memory
        tensor = torch.zeros(1000, 1000, device='cuda')
        del tensor

        result = robust_gpu_cleanup(force_gc=True, reset_peak_stats=True)

        self.assertTrue(result['success'])
        self.assertTrue(result['cuda_available'])
        self.assertIn('before_allocated_gb', result)
        self.assertIn('after_allocated_gb', result)
        self.assertIn('freed_gb', result)

    def test_cleanup_handles_exception(self):
        """Test cleanup handles exceptions gracefully."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'memory_allocated', side_effect=RuntimeError("Test error")):
                result = robust_gpu_cleanup()
                self.assertFalse(result['success'])
                self.assertIn('error', result)


class TestCheckGpuMemoryAvailable(unittest.TestCase):
    """Tests for check_gpu_memory_available function."""

    def test_check_when_cuda_unavailable(self):
        """Test returns False when CUDA unavailable."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            is_available, available_gb = check_gpu_memory_available(1.0)
            self.assertFalse(is_available)
            self.assertEqual(available_gb, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_check_with_cuda(self):
        """Test correctly reports available memory."""
        is_available, available_gb = check_gpu_memory_available(0.001)  # Very small requirement
        self.assertTrue(is_available)
        self.assertGreater(available_gb, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_check_with_large_requirement(self):
        """Test correctly reports insufficient memory for huge requirement."""
        is_available, available_gb = check_gpu_memory_available(10000.0)  # 10 TB
        self.assertFalse(is_available)
        self.assertGreater(available_gb, 0.0)


class TestGetGpuMemoryStats(unittest.TestCase):
    """Tests for get_gpu_memory_stats function."""

    def test_stats_when_cuda_unavailable(self):
        """Test returns proper dict when CUDA unavailable."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            stats = get_gpu_memory_stats()
            self.assertFalse(stats['cuda_available'])
            self.assertEqual(stats['total_gb'], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_stats_with_cuda(self):
        """Test returns proper stats with CUDA."""
        stats = get_gpu_memory_stats()
        self.assertTrue(stats['cuda_available'])
        self.assertGreater(stats['total_gb'], 0.0)
        self.assertIn('device_name', stats)
        self.assertIn('fragmentation_ratio', stats)


class TestHandleCudaError(unittest.TestCase):
    """Tests for handle_cuda_error function."""

    def test_oom_error(self):
        """Test correctly identifies OOM error."""
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        is_recoverable, error_type = handle_cuda_error(error)
        self.assertTrue(is_recoverable)
        self.assertEqual(error_type, "oom")

    def test_illegal_memory_access(self):
        """Test correctly identifies illegal memory access."""
        error = RuntimeError("CUDA error: an illegal memory access was encountered")
        is_recoverable, error_type = handle_cuda_error(error)
        self.assertFalse(is_recoverable)
        self.assertEqual(error_type, "illegal_memory_access")

    def test_device_side_assert(self):
        """Test correctly identifies device-side assert."""
        error = RuntimeError("CUDA error: device-side assert triggered")
        is_recoverable, error_type = handle_cuda_error(error)
        self.assertFalse(is_recoverable)
        self.assertEqual(error_type, "device_assert")

    def test_generic_cuda_error(self):
        """Test correctly identifies generic CUDA error."""
        error = RuntimeError("CUDA error: some cuda problem")
        is_recoverable, error_type = handle_cuda_error(error)
        self.assertTrue(is_recoverable)
        self.assertEqual(error_type, "cuda_generic")

    def test_non_cuda_error(self):
        """Test correctly handles non-CUDA error."""
        error = ValueError("Some other error")
        is_recoverable, error_type = handle_cuda_error(error)
        self.assertTrue(is_recoverable)
        self.assertEqual(error_type, "unknown")


class TestGpuMemoryContext(unittest.TestCase):
    """Tests for gpu_memory_context context manager."""

    def test_context_cleans_up_on_exit(self):
        """Test cleanup is called on exit."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'empty_cache') as mock_empty:
                with patch.object(torch.cuda, 'synchronize') as mock_sync:
                    with gpu_memory_context():
                        pass

                    mock_empty.assert_called()
                    mock_sync.assert_called()

    def test_context_cleans_up_on_exception(self):
        """Test cleanup is called even on exception."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'empty_cache') as mock_empty:
                with patch.object(torch.cuda, 'synchronize') as mock_sync:
                    try:
                        with gpu_memory_context():
                            raise ValueError("Test error")
                    except ValueError:
                        pass

                    mock_empty.assert_called()
                    mock_sync.assert_called()

    def test_context_with_cleanup_on_enter(self):
        """Test cleanup can be enabled on enter."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'empty_cache') as mock_empty:
                with gpu_memory_context(cleanup_on_enter=True):
                    pass

                # Should be called at least twice (enter and exit)
                self.assertGreaterEqual(mock_empty.call_count, 2)


class TestCUDAErrorRecovery(unittest.TestCase):
    """Tests for CUDAErrorRecovery class."""

    def test_oom_recovery_when_cuda_unavailable(self):
        """Test OOM recovery returns False when CUDA unavailable."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = CUDAErrorRecovery.attempt_oom_recovery()
            self.assertFalse(result)

    def test_reset_cuda_state_when_cuda_unavailable(self):
        """Test reset returns False when CUDA unavailable."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = CUDAErrorRecovery.reset_cuda_state()
            self.assertFalse(result)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_oom_recovery_with_cuda(self):
        """Test OOM recovery with CUDA available."""
        result = CUDAErrorRecovery.attempt_oom_recovery()
        # Should succeed if CUDA is available
        self.assertIsInstance(result, bool)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_reset_cuda_state_with_cuda(self):
        """Test reset works with CUDA available."""
        result = CUDAErrorRecovery.reset_cuda_state()
        self.assertTrue(result)

    def test_check_cuda_health_when_unavailable(self):
        """Test health check when CUDA unavailable."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            health = CUDAErrorRecovery.check_cuda_health()
            self.assertFalse(health['cuda_available'])
            self.assertFalse(health['healthy'])
            self.assertIn("CUDA not available", health['issues'])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_check_cuda_health_with_cuda(self):
        """Test health check with CUDA available."""
        health = CUDAErrorRecovery.check_cuda_health()
        self.assertTrue(health['cuda_available'])
        self.assertIn('issues', health)
        self.assertIn('recommendations', health)


class TestEstimateOperationVram(unittest.TestCase):
    """Tests for estimate_operation_vram function."""

    def test_inference_estimate(self):
        """Test inference VRAM estimation."""
        estimate = estimate_operation_vram(7.0, "inference")
        # Should be around 7 * 1.3 * 1.1 = ~10 GB
        self.assertGreater(estimate, 9.0)
        self.assertLess(estimate, 12.0)

    def test_quantization_estimate(self):
        """Test quantization VRAM estimation."""
        estimate = estimate_operation_vram(7.0, "quantization")
        # Should be around 7 * 2.5 * 1.1 = ~19 GB
        self.assertGreater(estimate, 17.0)
        self.assertLess(estimate, 22.0)

    def test_training_estimate(self):
        """Test training VRAM estimation."""
        estimate = estimate_operation_vram(7.0, "training")
        # Should be around 7 * 4.0 * 1.1 = ~31 GB
        self.assertGreater(estimate, 28.0)
        self.assertLess(estimate, 35.0)

    def test_qlora_estimate(self):
        """Test QLoRA VRAM estimation (should be lower)."""
        estimate = estimate_operation_vram(7.0, "qlora")
        # Should be around 7 * 0.5 * 1.1 = ~3.9 GB
        self.assertGreater(estimate, 3.0)
        self.assertLess(estimate, 5.0)

    def test_batch_size_increases_estimate(self):
        """Test that larger batch size increases estimate."""
        estimate_batch1 = estimate_operation_vram(7.0, "inference", batch_size=1)
        estimate_batch8 = estimate_operation_vram(7.0, "inference", batch_size=8)
        self.assertGreater(estimate_batch8, estimate_batch1)

    def test_sequence_length_affects_estimate(self):
        """Test that sequence length affects estimate."""
        estimate_short = estimate_operation_vram(7.0, "inference", sequence_length=512)
        estimate_long = estimate_operation_vram(7.0, "inference", sequence_length=4096)
        self.assertGreater(estimate_long, estimate_short)


if __name__ == '__main__':
    unittest.main()
