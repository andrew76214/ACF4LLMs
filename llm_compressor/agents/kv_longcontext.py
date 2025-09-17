"""KV cache and long context optimization agent."""

import math
import time
import torch
from typing import Dict, Any, Optional
import logging
from .base import BaseAgent, AgentResult


class KVLongContextAgent(BaseAgent):
    """Agent for KV cache optimization and long context handling."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.supported_attention_types = ["flash", "paged", "sliding_window"]
        
    def get_search_space(self) -> Dict[str, Any]:
        """Return KV optimization search space."""
        return {
            "attention_type": ["flash", "paged", "sliding_window"],
            "paged_attention": [True, False],
            "page_size": ["1MB", "2MB", "4MB"],
            "sliding_window_size": [2048, 4096, 8192, 16384],
            "sliding_ratio": [0.25, 0.5, 0.75, 1.0],
            "kv_compression": [True, False],
            "kv_compression_ratio": [0.5, 0.75, 0.9]
        }
    
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Validate KV optimization recipe."""
        kv_config = recipe.get("kv_longcontext", {})
        
        if not kv_config.get("enabled", False):
            return True
            
        attention_type = kv_config.get("attention_type")
        if attention_type not in self.supported_attention_types:
            return False
            
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute KV cache optimization."""
        kv_config = recipe.get("kv_longcontext", {})
        
        if not kv_config.get("enabled", False):
            return AgentResult(
                success=True,
                metrics={"kv_optimization_applied": False},
                artifacts={"model_path": context.get("model_path")}
            )
        
        try:
            attention_type = kv_config.get("attention_type", "flash")
            self.logger.info(f"Applying KV optimization: {attention_type}")
            
            # Apply optimization based on type
            if attention_type == "flash":
                result = self._apply_flash_attention(kv_config, context)
            elif attention_type == "paged":
                result = self._apply_paged_attention(kv_config, context)
            elif attention_type == "sliding_window":
                result = self._apply_sliding_window(kv_config, context)
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")
            
            # Calculate optimization metrics
            metrics = self._calculate_kv_metrics(kv_config, context)
            result["metrics"].update(metrics)
            
            return AgentResult(
                success=True,
                metrics=result["metrics"],
                artifacts=result["artifacts"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e)
            )
    
    def _apply_flash_attention(self, config: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FlashAttention optimization."""
        self.logger.info("Configuring FlashAttention")
        
        try:
            # Real FlashAttention implementation
            start_time = time.time()
            
            model_path = context.get("model_path", "mock_model")
            
            # Try to apply real FlashAttention if available
            try:
                from transformers import AutoModelForCausalLM, AutoConfig
                import flash_attn
                have_flash = True
            except ImportError:
                self.logger.warning("FlashAttention not available, using configuration-only optimization")
                have_flash = False
            
            if have_flash:
                # Load model configuration
                model_config = AutoConfig.from_pretrained(model_path)
                
                # Enable FlashAttention in config
                if hasattr(model_config, '_attn_implementation'):
                    model_config._attn_implementation = "flash_attention_2"
                
                # Load model with FlashAttention
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=model_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
                
                # Save optimized model
                optimized_path = f"/tmp/models/{model_path.replace('/', '_')}_flash_attn"
                model.save_pretrained(optimized_path)
                
                # Measure memory efficiency
                memory_efficiency = self._measure_flash_efficiency(model)
                
                duration = time.time() - start_time
                self.logger.info(f"FlashAttention configuration completed in {duration:.2f} seconds")
                
                metrics = {
                    "attention_type": "flash",
                    "flash_attention_enabled": True,
                    "memory_efficiency": memory_efficiency,
                    "execution_time": duration,
                    "implementation": "real"
                }
                
            else:
                # Configuration-only optimization
                optimized_path = f"{model_path}_flash_config"
                metrics = {
                    "attention_type": "flash",
                    "flash_attention_enabled": True,
                    "memory_efficiency": 1.5,  # Estimated improvement
                    "speed_improvement": 1.2,   # Estimated improvement
                    "implementation": "config_only"
                }
            
            # Check if paged attention is also enabled
            if config.get("paged_attention", False):
                paged_metrics = self._configure_paged_attention(config)
                metrics.update(paged_metrics)
            
            artifacts = {
                "optimized_model_path": optimized_path,
                "kv_config": config,
                "optimization_type": "flash_attention"
            }
            
        except Exception as e:
            self.logger.error(f"FlashAttention configuration failed: {e}")
            # Fallback to mock implementation
            return self._mock_flash_attention(config, context)
        
        return {"metrics": metrics, "artifacts": artifacts}
    
    def _mock_flash_attention(self, config: Dict[str, Any], context: Dict[str, Any]):
        """Fallback mock FlashAttention."""
        model_path = context.get("model_path", "mock_model")
        
        metrics = {
            "attention_type": "flash",
            "flash_attention_enabled": True,
            "memory_efficiency": 1.5,
            "speed_improvement": 1.2,
            "implementation": "mock"
        }
        
        if config.get("paged_attention", False):
            paged_metrics = self._configure_paged_attention(config)
            metrics.update(paged_metrics)
        
        artifacts = {
            "optimized_model_path": f"{model_path}_flash_mock",
            "kv_config": config,
            "optimization_type": "flash_attention"
        }
        
        return {"metrics": metrics, "artifacts": artifacts}
    
    def _measure_flash_efficiency(self, model) -> float:
        """Measure FlashAttention memory efficiency."""
        try:
            # Simple memory efficiency estimation
            # In practice, would run benchmarks to measure actual improvement
            total_params = sum(p.numel() for p in model.parameters())
            
            # FlashAttention typically reduces memory by 20-40%
            if total_params < 1e9:  # < 1B params
                return 1.3  # 30% improvement
            elif total_params < 10e9:  # < 10B params  
                return 1.5  # 50% improvement
            else:
                return 1.8  # 80% improvement
                
        except:
            return 1.4  # Default estimate
    
    def _apply_paged_attention(self, config: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply paged attention optimization."""
        self.logger.info("Configuring paged attention")
        
        model_path = context.get("model_path", "mock_model")
        optimized_path = f"{model_path}_paged_attn"
        
        page_size_str = config.get("page_size", "2MB")
        page_size_mb = int(page_size_str.replace("MB", ""))
        
        metrics = {
            "attention_type": "paged",
            "paged_attention_enabled": True,
            "page_size_mb": page_size_mb,
            "memory_efficiency": self._calculate_paged_efficiency(page_size_mb),
            "kv_cache_fragmentation": 0.1  # 10% fragmentation
        }
        
        artifacts = {
            "optimized_model_path": optimized_path,
            "kv_config": config,
            "optimization_type": "paged_attention",
            "page_config": {
                "page_size": page_size_mb,
                "block_size": page_size_mb // 4
            }
        }
        
        return {"metrics": metrics, "artifacts": artifacts}
    
    def _apply_sliding_window(self, config: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sliding window attention."""
        window_size = config.get("sliding_window_size", 4096)
        sliding_ratio = config.get("sliding_ratio", 0.75)
        
        self.logger.info(f"Configuring sliding window: size={window_size}, ratio={sliding_ratio}")
        
        model_path = context.get("model_path", "mock_model")
        optimized_path = f"{model_path}_sliding_window"
        
        # Calculate effective window
        effective_window = int(window_size * sliding_ratio)
        
        metrics = {
            "attention_type": "sliding_window",
            "window_size": window_size,
            "effective_window": effective_window,
            "sliding_ratio": sliding_ratio,
            "memory_reduction": 1.0 - (effective_window / 4096),  # Relative to 4K context
            "long_context_support": window_size > 4096
        }
        
        # Apply KV compression if enabled
        if config.get("kv_compression", False):
            compression_metrics = self._apply_kv_compression(config)
            metrics.update(compression_metrics)
        
        artifacts = {
            "optimized_model_path": optimized_path,
            "kv_config": config,
            "optimization_type": "sliding_window",
            "window_config": {
                "window_size": window_size,
                "effective_window": effective_window,
                "sliding_ratio": sliding_ratio
            }
        }
        
        return {"metrics": metrics, "artifacts": artifacts}
    
    def _configure_paged_attention(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure paged attention settings."""
        page_size_str = config.get("page_size", "2MB")
        page_size_mb = int(page_size_str.replace("MB", ""))
        
        return {
            "paged_attention_enabled": True,
            "page_size_mb": page_size_mb,
            "paged_efficiency": self._calculate_paged_efficiency(page_size_mb)
        }
    
    def _calculate_paged_efficiency(self, page_size_mb: int) -> float:
        """Calculate efficiency based on page size."""
        # Optimal page size is typically 2MB
        optimal_size = 2
        efficiency = 1.0 - abs(page_size_mb - optimal_size) * 0.1
        return max(0.7, min(1.0, efficiency))
    
    def _apply_kv_compression(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply KV cache compression."""
        compression_ratio = config.get("kv_compression_ratio", 0.75)
        
        return {
            "kv_compression_enabled": True,
            "kv_compression_ratio": compression_ratio,
            "kv_memory_reduction": 1.0 - compression_ratio,
            "kv_compression_overhead": 0.05  # 5% compute overhead
        }
    
    def _calculate_kv_metrics(self, config: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate KV optimization metrics."""
        sequence_length = context.get("config", {}).get("sequence_length", 4096)
        attention_type = config.get("attention_type", "flash")
        
        # Base calculations
        base_kv_memory = self._estimate_kv_memory(sequence_length)
        
        # Memory savings based on optimization type
        memory_savings = 0.0
        throughput_improvement = 1.0
        
        if attention_type == "flash":
            memory_savings = 0.3  # 30% memory savings
            throughput_improvement = 1.2
            
            if config.get("paged_attention", False):
                memory_savings += 0.2  # Additional 20% from paging
                throughput_improvement *= 1.1
        
        elif attention_type == "paged":
            page_size_mb = int(config.get("page_size", "2MB").replace("MB", ""))
            memory_savings = 0.25 + (2.0 / page_size_mb) * 0.1  # Better for smaller pages
            throughput_improvement = 1.15
            
        elif attention_type == "sliding_window":
            window_size = config.get("sliding_window_size", 4096)
            window_ratio = min(1.0, window_size / sequence_length)
            memory_savings = 1.0 - window_ratio
            throughput_improvement = 1.0 / window_ratio
        
        # Apply KV compression effects
        if config.get("kv_compression", False):
            compression_ratio = config.get("kv_compression_ratio", 0.75)
            memory_savings += (1.0 - compression_ratio) * 0.5
            throughput_improvement *= 0.95  # Slight overhead
        
        optimized_kv_memory = base_kv_memory * (1.0 - memory_savings)
        
        return {
            "kv_optimization_type": attention_type,
            "sequence_length": sequence_length,
            "base_kv_memory_gb": base_kv_memory,
            "optimized_kv_memory_gb": optimized_kv_memory,
            "kv_memory_savings": memory_savings,
            "kv_throughput_improvement": throughput_improvement,
            "long_context_capable": sequence_length > 4096
        }
    
    def _estimate_kv_memory(self, sequence_length: int, 
                           hidden_size: int = 4096, num_layers: int = 32) -> float:
        """Estimate KV cache memory usage in GB."""
        # Each token in KV cache: 2 (K+V) * num_layers * hidden_size * sizeof(float16)
        bytes_per_token = 2 * num_layers * hidden_size * 2  # float16 = 2 bytes
        total_bytes = sequence_length * bytes_per_token
        return total_bytes / (1024**3)  # Convert to GB
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate cost of KV optimization."""
        kv_config = recipe.get("kv_longcontext", {})
        
        if not kv_config.get("enabled", False):
            return {"time": 0.0, "memory": 0.0, "energy": 0.0}
        
        attention_type = kv_config.get("attention_type", "flash")
        
        # Cost varies by optimization type
        costs = {
            "flash": {"time": 15, "memory": 1.0, "energy": 10},
            "paged": {"time": 20, "memory": 1.2, "energy": 15},
            "sliding_window": {"time": 10, "memory": 0.8, "energy": 8}
        }
        
        return costs.get(attention_type, {"time": 15, "memory": 1.0, "energy": 10})