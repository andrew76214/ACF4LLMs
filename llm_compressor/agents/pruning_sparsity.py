"""Pruning and sparsity agent for model compression."""

import numpy as np
from typing import Dict, Any, Optional
import logging
from .base import BaseAgent, AgentResult


class PruningSparsityAgent(BaseAgent):
    """Agent for model pruning and structured sparsity."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.supported_methods = ["unstructured", "structured", "n_m_sparsity"]
        
    def get_search_space(self) -> Dict[str, Any]:
        """Return pruning search space."""
        return {
            "head_pruning_ratio": [0.0, 0.1, 0.2, 0.3],
            "ffn_pruning_ratio": [0.0, 0.15, 0.25, 0.4], 
            "structured_sparsity": ["none", "2:4", "1:2", "4:8"],
            "pruning_method": ["magnitude", "gradient", "fisher"],
            "gradual_pruning": [True, False],
            "layer_wise_pruning": [True, False]
        }
    
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Validate pruning recipe."""
        prune_config = recipe.get("pruning_sparsity", {})
        
        if not prune_config.get("enabled", False):
            return True
            
        # Check ratio bounds
        head_ratio = prune_config.get("head_pruning_ratio", 0.0)
        ffn_ratio = prune_config.get("ffn_pruning_ratio", 0.0)
        
        if not (0.0 <= head_ratio <= 0.5):
            return False
        if not (0.0 <= ffn_ratio <= 0.6):
            return False
            
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute pruning and sparsity optimization."""
        prune_config = recipe.get("pruning_sparsity", {})
        
        if not prune_config.get("enabled", False):
            return AgentResult(
                success=True,
                metrics={"pruning_applied": False},
                artifacts={"model_path": context.get("model_path")}
            )
        
        try:
            model_path = self._get_input_model_path(context)
            
            # Apply different pruning techniques
            pruning_results = {}
            
            # Head pruning
            if prune_config.get("head_pruning_ratio", 0.0) > 0:
                head_results = self._apply_head_pruning(model_path, prune_config)
                pruning_results["head_pruning"] = head_results
            
            # FFN pruning
            if prune_config.get("ffn_pruning_ratio", 0.0) > 0:
                ffn_results = self._apply_ffn_pruning(model_path, prune_config)
                pruning_results["ffn_pruning"] = ffn_results
            
            # Structured sparsity
            if prune_config.get("structured_sparsity", "none") != "none":
                sparsity_results = self._apply_structured_sparsity(model_path, prune_config)
                pruning_results["structured_sparsity"] = sparsity_results
            
            # Calculate overall metrics
            overall_metrics = self._calculate_pruning_metrics(pruning_results, prune_config)
            
            # Generate pruned model path
            pruned_model_path = f"{model_path}_pruned"
            
            artifacts = {
                "pruned_model_path": pruned_model_path,
                "pruning_config": prune_config,
                "pruning_results": pruning_results
            }
            
            return AgentResult(
                success=True,
                metrics=overall_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e)
            )
    
    def _get_input_model_path(self, context: Dict[str, Any]) -> str:
        """Get the input model path from previous agents."""
        artifacts = context.get("artifacts", {})
        
        # Check for quantized model first
        if "quantization" in artifacts:
            quantized_path = artifacts["quantization"].get("quantized_model_path")
            if quantized_path:
                return quantized_path
        
        return context.get("model_path", "mock_model")
    
    def _apply_head_pruning(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply attention head pruning."""
        head_ratio = config.get("head_pruning_ratio", 0.0)
        method = config.get("pruning_method", "magnitude")
        
        self.logger.info(f"Applying head pruning: {head_ratio*100:.1f}% ratio using {method}")
        
        # Mock head pruning implementation
        # In real implementation, would analyze attention patterns and prune heads
        
        num_layers = 32  # Typical for Llama-3-8B
        heads_per_layer = 32
        total_heads = num_layers * heads_per_layer
        
        heads_to_prune = int(total_heads * head_ratio)
        
        # Simulate head importance scoring
        head_importance = np.random.rand(total_heads)
        pruned_head_indices = np.argsort(head_importance)[:heads_to_prune]
        
        return {
            "pruning_ratio": head_ratio,
            "total_heads": total_heads,
            "pruned_heads": heads_to_prune,
            "remaining_heads": total_heads - heads_to_prune,
            "pruned_indices": pruned_head_indices.tolist(),
            "method": method,
            "compression_ratio": 1.0 - head_ratio
        }
    
    def _apply_ffn_pruning(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feed-forward network pruning."""
        ffn_ratio = config.get("ffn_pruning_ratio", 0.0)
        method = config.get("pruning_method", "magnitude")
        
        self.logger.info(f"Applying FFN pruning: {ffn_ratio*100:.1f}% ratio using {method}")
        
        # Mock FFN pruning implementation
        num_layers = 32
        ffn_size = 11008  # Typical intermediate size for Llama-3-8B
        
        neurons_per_layer = ffn_size
        total_neurons = num_layers * neurons_per_layer
        neurons_to_prune = int(total_neurons * ffn_ratio)
        
        # Simulate neuron importance scoring
        neuron_importance = np.random.rand(total_neurons)
        pruned_neuron_indices = np.argsort(neuron_importance)[:neurons_to_prune]
        
        return {
            "pruning_ratio": ffn_ratio,
            "total_neurons": total_neurons,
            "pruned_neurons": neurons_to_prune,
            "remaining_neurons": total_neurons - neurons_to_prune,
            "neurons_per_layer": neurons_per_layer,
            "method": method,
            "compression_ratio": 1.0 - ffn_ratio
        }
    
    def _apply_structured_sparsity(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply structured sparsity (N:M patterns)."""
        sparsity_pattern = config.get("structured_sparsity", "2:4")
        
        if sparsity_pattern == "none":
            return {"enabled": False}
        
        self.logger.info(f"Applying structured sparsity: {sparsity_pattern}")
        
        # Parse N:M pattern
        if ":" in sparsity_pattern:
            n, m = map(int, sparsity_pattern.split(":"))
            sparsity_ratio = 1.0 - (n / m)
        else:
            n, m = 2, 4
            sparsity_ratio = 0.5
        
        # Mock structured sparsity implementation
        total_weights = 7000000000  # Approximate for Llama-3-8B
        sparse_weights = int(total_weights * sparsity_ratio)
        
        return {
            "pattern": sparsity_pattern,
            "n": n,
            "m": m,
            "sparsity_ratio": sparsity_ratio,
            "total_weights": total_weights,
            "sparse_weights": sparse_weights,
            "compression_ratio": sparsity_ratio,
            "hardware_optimized": True  # N:M patterns are hardware friendly
        }
    
    def _calculate_pruning_metrics(self, pruning_results: Dict[str, Any], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pruning metrics."""
        
        metrics = {"pruning_applied": True}
        
        # Aggregate compression ratios
        total_compression = 1.0
        compression_sources = []
        
        # Head pruning compression
        if "head_pruning" in pruning_results:
            head_compression = pruning_results["head_pruning"]["compression_ratio"]
            total_compression *= head_compression
            compression_sources.append("heads")
            metrics["head_pruning_ratio"] = pruning_results["head_pruning"]["pruning_ratio"]
        
        # FFN pruning compression  
        if "ffn_pruning" in pruning_results:
            ffn_compression = pruning_results["ffn_pruning"]["compression_ratio"]
            total_compression *= ffn_compression
            compression_sources.append("ffn")
            metrics["ffn_pruning_ratio"] = pruning_results["ffn_pruning"]["pruning_ratio"]
        
        # Structured sparsity compression
        if "structured_sparsity" in pruning_results and pruning_results["structured_sparsity"].get("enabled", True):
            sparsity_compression = pruning_results["structured_sparsity"]["compression_ratio"]
            total_compression *= sparsity_compression
            compression_sources.append("sparsity")
            metrics["structured_sparsity_ratio"] = pruning_results["structured_sparsity"]["sparsity_ratio"]
        
        # Overall metrics
        metrics["total_compression_ratio"] = total_compression
        metrics["model_size_reduction"] = 1.0 - total_compression
        metrics["compression_sources"] = compression_sources
        
        # Estimate performance impact
        metrics.update(self._estimate_performance_impact(pruning_results, config))
        
        return metrics
    
    def _estimate_performance_impact(self, pruning_results: Dict[str, Any],
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate performance impact of pruning."""
        
        # Base estimates for different pruning types
        speedup_factor = 1.0
        accuracy_impact = 0.0
        memory_reduction = 0.0
        
        # Head pruning impact
        if "head_pruning" in pruning_results:
            head_ratio = pruning_results["head_pruning"]["pruning_ratio"]
            speedup_factor *= (1.0 + head_ratio * 0.2)  # Modest speedup
            accuracy_impact += head_ratio * 0.15  # Accuracy drop
            memory_reduction += head_ratio * 0.1  # Memory savings
        
        # FFN pruning impact
        if "ffn_pruning" in pruning_results:
            ffn_ratio = pruning_results["ffn_pruning"]["pruning_ratio"]
            speedup_factor *= (1.0 + ffn_ratio * 0.4)  # Better speedup from FFN pruning
            accuracy_impact += ffn_ratio * 0.2  # Accuracy drop
            memory_reduction += ffn_ratio * 0.3  # Better memory savings
        
        # Structured sparsity impact
        if "structured_sparsity" in pruning_results and pruning_results["structured_sparsity"].get("enabled", True):
            sparsity_ratio = pruning_results["structured_sparsity"]["sparsity_ratio"]
            speedup_factor *= (1.0 + sparsity_ratio * 0.3)  # Hardware-accelerated speedup
            accuracy_impact += sparsity_ratio * 0.1  # Lower accuracy impact for structured
            memory_reduction += sparsity_ratio * 0.4
        
        return {
            "estimated_speedup": speedup_factor,
            "estimated_accuracy_drop": accuracy_impact,
            "estimated_memory_reduction": memory_reduction,
            "pruning_efficiency": (speedup_factor - 1.0) / max(accuracy_impact, 0.001)  # Speedup per accuracy drop
        }
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate pruning cost."""
        prune_config = recipe.get("pruning_sparsity", {})
        
        if not prune_config.get("enabled", False):
            return {"time": 0.0, "memory": 0.0, "energy": 0.0}
        
        # Base cost for analysis and pruning
        base_time = 60  # 1 hour base
        
        # Additional cost based on pruning types
        cost_multiplier = 1.0
        
        if prune_config.get("head_pruning_ratio", 0.0) > 0:
            cost_multiplier += 0.5  # Head analysis is expensive
        
        if prune_config.get("ffn_pruning_ratio", 0.0) > 0:
            cost_multiplier += 0.3  # FFN analysis
        
        if prune_config.get("structured_sparsity", "none") != "none":
            cost_multiplier += 0.2  # Structured sparsity setup
        
        if prune_config.get("gradual_pruning", False):
            cost_multiplier += 0.5  # Iterative pruning takes longer
        
        total_time = base_time * cost_multiplier
        
        return {
            "time": total_time,
            "memory": 3.0,  # High memory for model analysis
            "energy": total_time * 0.8  # GPU intensive analysis
        }