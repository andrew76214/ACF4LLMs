"""Recipe planner agent for generating optimization strategies."""

import itertools
from typing import Dict, Any, List
import random
import logging
from .base import BaseAgent, AgentResult


class RecipePlannerAgent(BaseAgent):
    """Agent for planning optimization recipes based on constraints and objectives."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.baseline_recipes = self._load_baseline_recipes()
        
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Recipe planner always validates (generates recipes)."""
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Generate optimization recipes."""
        try:
            config = context.get("config", {})
            
            # Generate different types of recipes
            recipes = []
            
            # Add baseline recipes
            recipes.extend(self._generate_baseline_recipes(config))
            
            # Add search-based recipes if search agent is enabled
            if config.get("agents", {}).get("search", {}).get("enabled", True):
                recipes.extend(self._generate_search_recipes(config))
            
            # Add custom recipes from config
            custom_recipes = config.get("custom_recipes", [])
            recipes.extend(custom_recipes)
            
            # Filter recipes based on constraints
            filtered_recipes = self._filter_recipes(recipes, config)
            
            self.logger.info(f"Generated {len(filtered_recipes)} valid recipes")
            
            return AgentResult(
                success=True,
                metrics={"recipes_generated": len(filtered_recipes)},
                artifacts={"recipes": filtered_recipes}
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e)
            )
    
    def _load_baseline_recipes(self) -> List[Dict[str, Any]]:
        """Load predefined baseline recipes."""
        return [
            {
                "name": "quantization_only",
                "description": "AWQ 4-bit quantization only",
                "pipeline": ["quantization", "perf_carbon", "eval_safety"],
                "quantization": {
                    "enabled": True,
                    "method": "awq",
                    "bits": 4,
                    "group_size": 128
                },
                "kv_longcontext": {"enabled": False},
                "pruning_sparsity": {"enabled": False},
                "distillation": {"enabled": False}
            },
            {
                "name": "kv_optimization_only", 
                "description": "KV cache optimization with FlashAttention",
                "pipeline": ["kv_longcontext", "perf_carbon", "eval_safety"],
                "quantization": {"enabled": False},
                "kv_longcontext": {
                    "enabled": True,
                    "attention_type": "flash",
                    "paged_attention": True,
                    "page_size": "2MB"
                },
                "pruning_sparsity": {"enabled": False},
                "distillation": {"enabled": False}
            },
            {
                "name": "quantization_plus_kv",
                "description": "AWQ quantization + KV optimization",
                "pipeline": ["quantization", "kv_longcontext", "perf_carbon", "eval_safety"],
                "quantization": {
                    "enabled": True,
                    "method": "awq",
                    "bits": 4,
                    "group_size": 128
                },
                "kv_longcontext": {
                    "enabled": True,
                    "attention_type": "flash",
                    "paged_attention": True,
                    "page_size": "2MB"
                },
                "pruning_sparsity": {"enabled": False},
                "distillation": {"enabled": False}
            }
        ]
    
    def _generate_baseline_recipes(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate baseline recipes with IDs."""
        recipes = []
        
        for i, recipe in enumerate(self.baseline_recipes):
            recipe_copy = recipe.copy()
            recipe_copy["id"] = f"baseline_{recipe['name']}_{i}"
            recipe_copy["type"] = "baseline"
            recipes.append(recipe_copy)
        
        return recipes
    
    def _generate_search_recipes(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recipes for search-based optimization."""
        recipes = []
        
        # Get search configuration
        search_config = config.get("search", {})
        num_candidates = search_config.get("num_candidates", 20)
        
        # Define search space
        quantization_options = [
            {"enabled": False},
            {"enabled": True, "method": "awq", "bits": 4, "group_size": 128},
            {"enabled": True, "method": "awq", "bits": 8, "group_size": 64},
            {"enabled": True, "method": "gptq", "bits": 4, "group_size": 128},
            {"enabled": True, "method": "bnb", "bits": 4}
        ]
        
        kv_options = [
            {"enabled": False},
            {"enabled": True, "attention_type": "flash", "paged_attention": False},
            {"enabled": True, "attention_type": "flash", "paged_attention": True, "page_size": "1MB"},
            {"enabled": True, "attention_type": "flash", "paged_attention": True, "page_size": "2MB"}
        ]
        
        pruning_options = [
            {"enabled": False},
            {"enabled": True, "head_pruning_ratio": 0.1, "ffn_pruning_ratio": 0.15},
            {"enabled": True, "head_pruning_ratio": 0.2, "ffn_pruning_ratio": 0.25},
            {"enabled": True, "structured_sparsity": "2:4"}
        ]
        
        distillation_options = [
            {"enabled": False},
            {"enabled": True, "temperature": 3.0, "alpha": 0.3, "method": "lora"},
            {"enabled": True, "temperature": 4.0, "alpha": 0.5, "method": "qlora"}
        ]
        
        # Generate combinations
        all_combinations = list(itertools.product(
            quantization_options, kv_options, pruning_options, distillation_options
        ))
        
        # Sample random combinations
        sampled_combinations = random.sample(
            all_combinations, 
            min(num_candidates, len(all_combinations))
        )
        
        for i, (quant, kv, prune, distill) in enumerate(sampled_combinations):
            # Build pipeline based on enabled components
            pipeline = []
            if quant["enabled"]:
                pipeline.append("quantization")
            if kv["enabled"]:
                pipeline.append("kv_longcontext")
            if prune["enabled"]:
                pipeline.append("pruning_sparsity")
            if distill["enabled"]:
                pipeline.append("distillation")
            
            # Always add performance and evaluation
            pipeline.extend(["perf_carbon", "eval_safety"])
            
            recipe = {
                "id": f"search_{i}",
                "name": f"search_candidate_{i}",
                "type": "search",
                "description": f"Search-generated recipe {i}",
                "pipeline": pipeline,
                "quantization": quant,
                "kv_longcontext": kv,
                "pruning_sparsity": prune,
                "distillation": distill
            }
            
            recipes.append(recipe)
        
        return recipes
    
    def _filter_recipes(self, recipes: List[Dict[str, Any]], 
                       config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter recipes based on constraints."""
        filtered = []
        constraints = config.get("constraints", {})
        
        max_accuracy_drop = constraints.get("max_accuracy_drop", 0.05)  # 5%
        max_latency_ms = constraints.get("max_latency_ms", 150)
        max_vram_gb = constraints.get("max_vram_gb", 80)
        
        for recipe in recipes:
            if self._satisfies_constraints(recipe, constraints):
                filtered.append(recipe)
            else:
                self.logger.debug(f"Recipe {recipe.get('name')} filtered out by constraints")
        
        return filtered
    
    def _satisfies_constraints(self, recipe: Dict[str, Any], 
                              constraints: Dict[str, Any]) -> bool:
        """Check if recipe satisfies constraints."""
        
        # Estimate recipe impact
        estimated_accuracy_drop = 0.0
        estimated_latency_factor = 1.0
        estimated_vram_factor = 1.0
        
        # Quantization impact
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            method = recipe["quantization"].get("method", "awq")
            
            accuracy_impact = {
                ("awq", 4): 0.005,
                ("awq", 8): 0.002,
                ("gptq", 4): 0.008,
                ("gptq", 8): 0.003,
                ("bnb", 4): 0.015,
                ("bnb", 8): 0.008
            }.get((method, bits), 0.01)
            
            estimated_accuracy_drop += accuracy_impact
            estimated_latency_factor *= 0.7 if bits == 4 else 0.85
            estimated_vram_factor *= 0.25 if bits == 4 else 0.5
        
        # Pruning impact
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            head_ratio = prune_config.get("head_pruning_ratio", 0.0)
            ffn_ratio = prune_config.get("ffn_pruning_ratio", 0.0)
            
            estimated_accuracy_drop += (head_ratio + ffn_ratio) * 0.1
            estimated_latency_factor *= (1.0 - (head_ratio + ffn_ratio) * 0.3)
            estimated_vram_factor *= (1.0 - (head_ratio + ffn_ratio) * 0.4)
        
        # Distillation impact
        if recipe.get("distillation", {}).get("enabled", False):
            estimated_accuracy_drop += 0.01  # Small accuracy drop from distillation
        
        # Check constraints
        max_accuracy_drop = constraints.get("max_accuracy_drop", 0.05)
        max_latency_ms = constraints.get("max_latency_ms", 150)
        max_vram_gb = constraints.get("max_vram_gb", 80)
        
        baseline_latency = constraints.get("baseline_latency_ms", 100)
        baseline_vram = constraints.get("baseline_vram_gb", 40)
        
        estimated_latency = baseline_latency * estimated_latency_factor
        estimated_vram = baseline_vram * estimated_vram_factor
        
        if estimated_accuracy_drop > max_accuracy_drop:
            return False
        if estimated_latency > max_latency_ms:
            return False  
        if estimated_vram > max_vram_gb:
            return False
        
        return True
    
    def get_search_space(self) -> Dict[str, Any]:
        """Get the search space for recipe generation."""
        return {
            "quantization_method": ["awq", "gptq", "bnb"],
            "quantization_bits": [4, 8],
            "kv_attention_type": ["flash", "paged"],
            "pruning_head_ratio": [0.0, 0.1, 0.2, 0.3],
            "pruning_ffn_ratio": [0.0, 0.15, 0.25, 0.4],
            "distillation_temperature": [2.0, 3.0, 4.0],
            "distillation_alpha": [0.1, 0.3, 0.5, 0.7]
        }