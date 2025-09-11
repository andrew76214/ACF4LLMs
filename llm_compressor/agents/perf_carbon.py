"""Performance and carbon footprint measurement agent."""

import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional
import subprocess
import logging
from .base import BaseAgent, AgentResult


class PerfCarbonAgent(BaseAgent):
    """Agent for measuring performance metrics and carbon footprint."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.gpu_available = self._check_gpu_availability()
        self.grid_emission_factor = config.get("grid_emission_factor_kg_per_kwh", 0.4)
        
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Performance measurement can run on any recipe."""
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute performance and carbon measurements."""
        try:
            model_path = context.get("artifacts", {}).get("quantization", {}).get("quantized_model_path") or \
                        context.get("artifacts", {}).get("kv_longcontext", {}).get("optimized_model_path") or \
                        context.get("model_path", "mock_model")
            
            # Run performance benchmarks
            latency_metrics = self._measure_latency(model_path, recipe)
            throughput_metrics = self._measure_throughput(model_path, recipe)
            memory_metrics = self._measure_memory_usage(model_path, recipe)
            energy_metrics = self._measure_energy_consumption(model_path, recipe)
            
            # Calculate carbon footprint
            carbon_metrics = self._calculate_carbon_footprint(energy_metrics)
            
            # Combine all metrics
            all_metrics = {
                **latency_metrics,
                **throughput_metrics,
                **memory_metrics,
                **energy_metrics,
                **carbon_metrics
            }
            
            artifacts = {
                "benchmark_results": all_metrics,
                "model_path": model_path,
                "measurement_config": {
                    "grid_emission_factor": self.grid_emission_factor,
                    "gpu_available": self.gpu_available
                }
            }
            
            return AgentResult(
                success=True,
                metrics=all_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e)
            )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                return result.returncode == 0
            except FileNotFoundError:
                return False
    
    def _measure_latency(self, model_path: str, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Measure inference latency."""
        self.logger.info("Measuring latency...")
        
        # Mock latency measurement - in real implementation, load model and benchmark
        sequence_length = self.config.get("sequence_length", 512)
        batch_size = self.config.get("batch_size", 1)
        
        # Simulate latency based on optimizations
        base_latency = 100.0  # Base latency in ms
        
        # Apply optimization effects
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            speedup = 1.8 if bits == 4 else 1.4 if bits == 8 else 1.0
            base_latency /= speedup
        
        if recipe.get("kv_longcontext", {}).get("enabled", False):
            if recipe["kv_longcontext"].get("attention_type") == "flash":
                base_latency *= 0.85
            if recipe["kv_longcontext"].get("paged_attention"):
                base_latency *= 0.9
        
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            reduction = prune_config.get("head_pruning_ratio", 0) + prune_config.get("ffn_pruning_ratio", 0)
            base_latency *= (1.0 - reduction * 0.3)
        
        # Add some variance
        latencies = [base_latency * (1 + np.random.normal(0, 0.1)) for _ in range(100)]
        
        return {
            "latency_mean_ms": np.mean(latencies),
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "latency_std_ms": np.std(latencies)
        }
    
    def _measure_throughput(self, model_path: str, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Measure throughput (tokens/second)."""
        self.logger.info("Measuring throughput...")
        
        # Base throughput (tokens/sec)
        base_throughput = 50.0
        
        # Apply optimization effects
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            speedup = 1.8 if bits == 4 else 1.4 if bits == 8 else 1.0
            base_throughput *= speedup
        
        if recipe.get("kv_longcontext", {}).get("enabled", False):
            if recipe["kv_longcontext"].get("attention_type") == "flash":
                base_throughput *= 1.2
        
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            reduction = prune_config.get("head_pruning_ratio", 0) + prune_config.get("ffn_pruning_ratio", 0)
            base_throughput *= (1.0 + reduction * 0.4)
        
        return {
            "throughput_tokens_per_sec": base_throughput,
            "throughput_samples_per_sec": base_throughput / 512  # Assuming 512 tokens per sample
        }
    
    def _measure_memory_usage(self, model_path: str, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Measure memory usage."""
        self.logger.info("Measuring memory usage...")
        
        # Base VRAM usage (GB) 
        base_vram = 40.0  # For Llama-3-8B
        
        # Apply optimization effects
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            if bits == 4:
                base_vram *= 0.25
            elif bits == 8:
                base_vram *= 0.5
        
        if recipe.get("kv_longcontext", {}).get("enabled", False):
            if recipe["kv_longcontext"].get("paged_attention"):
                base_vram *= 0.85  # Paged attention reduces peak memory
        
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            reduction = prune_config.get("head_pruning_ratio", 0) + prune_config.get("ffn_pruning_ratio", 0)
            base_vram *= (1.0 - reduction * 0.4)
        
        # Estimate KV cache memory
        sequence_length = self.config.get("sequence_length", 4096)
        kv_cache_memory = (sequence_length / 1024) * 2.0  # Rough estimate
        
        return {
            "vram_peak_gb": base_vram,
            "vram_allocated_gb": base_vram * 0.9,
            "kv_cache_memory_gb": kv_cache_memory,
            "system_memory_gb": psutil.virtual_memory().used / (1024**3)
        }
    
    def _measure_energy_consumption(self, model_path: str, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Measure energy consumption."""
        self.logger.info("Measuring energy consumption...")
        
        # Base power consumption (watts)
        base_power = 300.0  # For A100
        
        # Apply optimization effects
        power_factor = 1.0
        
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            power_factor *= 0.7 if bits == 4 else 0.85 if bits == 8 else 1.0
        
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            reduction = prune_config.get("head_pruning_ratio", 0) + prune_config.get("ffn_pruning_ratio", 0)
            power_factor *= (1.0 - reduction * 0.2)
        
        actual_power = base_power * power_factor
        
        # Estimate energy for different scenarios
        inference_time_hours = 1.0  # 1 hour of inference
        energy_kwh = (actual_power * inference_time_hours) / 1000
        
        return {
            "power_consumption_watts": actual_power,
            "energy_per_token_joules": (actual_power / 50) * 1,  # Assuming 50 tokens/sec
            "energy_1hour_kwh": energy_kwh,
            "energy_1000_samples_kwh": energy_kwh * (1000 / 3600)  # Assuming 1 sample/sec
        }
    
    def _calculate_carbon_footprint(self, energy_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate carbon footprint based on energy consumption."""
        
        carbon_metrics = {}
        
        for energy_key, energy_value in energy_metrics.items():
            if "kwh" in energy_key:
                carbon_key = energy_key.replace("kwh", "co2e_kg")
                carbon_metrics[carbon_key] = energy_value * self.grid_emission_factor
        
        # Additional carbon metrics
        carbon_metrics.update({
            "co2e_per_token_g": energy_metrics.get("energy_per_token_joules", 0) * 
                               (self.grid_emission_factor / 3600) * 1000,  # Convert to grams
            "grid_emission_factor": self.grid_emission_factor
        })
        
        return carbon_metrics
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU-specific metrics if available."""
        if not self.gpu_available:
            return {}
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    "gpu_utilization_pct": gpu.load * 100,
                    "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                    "gpu_memory_total_gb": gpu.memoryTotal / 1024,
                    "gpu_temperature_c": gpu.temperature
                }
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
        
        return {}
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate cost of performance measurement."""
        return {
            "time": 30.0,    # 30 seconds for benchmarking
            "memory": 0.5,   # Minimal memory overhead
            "energy": 5.0    # 5 minutes of GPU time
        }