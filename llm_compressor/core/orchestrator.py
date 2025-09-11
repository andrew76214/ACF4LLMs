"""Main orchestrator for multi-agent LLM compression pipeline."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import time
from pathlib import Path

from .registry import Registry
from .pareto import ParetoFrontier
from .metrics import MetricsCollector
from ..agents.base import BaseAgent, AgentResult


class Orchestrator:
    """Orchestrates the multi-agent compression pipeline."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.registry = Registry(self.config.get("registry", {}))
        self.pareto = ParetoFrontier(self.config.get("pareto", {}))
        self.metrics = MetricsCollector(self.config.get("metrics", {}))
        self.agents = self._initialize_agents()
        self.logger = logging.getLogger("orchestrator")
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all agents based on configuration."""
        agents = {}
        
        # Import agent classes dynamically
        from ..agents.data_curation import DataCurationAgent
        from ..agents.recipe_planner import RecipePlannerAgent
        from ..agents.quantization import QuantizationAgent
        from ..agents.pruning_sparsity import PruningSparsityAgent
        from ..agents.distillation import DistillationAgent
        from ..agents.kv_longcontext import KVLongContextAgent
        from ..agents.rag import RAGAgent
        from ..agents.perf_carbon import PerfCarbonAgent
        from ..agents.search import SearchAgent
        from ..agents.eval_safety import EvalSafetyAgent
        
        agent_classes = {
            "data_curation": DataCurationAgent,
            "recipe_planner": RecipePlannerAgent,
            "quantization": QuantizationAgent,
            "pruning_sparsity": PruningSparsityAgent,
            "distillation": DistillationAgent,
            "kv_longcontext": KVLongContextAgent,
            "rag": RAGAgent,
            "perf_carbon": PerfCarbonAgent,
            "search": SearchAgent,
            "eval_safety": EvalSafetyAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            if agent_name in self.config.get("agents", {}):
                agent_config = self.config["agents"][agent_name]
                agents[agent_name] = agent_class(agent_name, agent_config)
                self.logger.info(f"Initialized agent: {agent_name}")
        
        return agents
    
    def generate_recipes(self) -> List[Dict[str, Any]]:
        """Generate optimization recipes using the recipe planner."""
        if "recipe_planner" not in self.agents:
            raise ValueError("RecipePlannerAgent not configured")
        
        planner = self.agents["recipe_planner"]
        context = {"config": self.config}
        
        result = planner.run({}, context)
        if not result.success:
            raise RuntimeError(f"Recipe generation failed: {result.error}")
        
        recipes = result.artifacts.get("recipes", [])
        self.logger.info(f"Generated {len(recipes)} recipes")
        return recipes
    
    def execute_recipe(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single recipe through the agent pipeline."""
        recipe_id = recipe.get("id", f"recipe_{int(time.time())}")
        self.logger.info(f"Executing recipe: {recipe_id}")
        
        context = {
            "config": self.config,
            "recipe_id": recipe_id,
            "artifacts": {}
        }
        
        results = {}
        pipeline = recipe.get("pipeline", [])
        
        # Execute agents in sequence
        for agent_name in pipeline:
            if agent_name not in self.agents:
                self.logger.warning(f"Agent {agent_name} not available, skipping")
                continue
            
            agent = self.agents[agent_name]
            agent_result = agent.run(recipe, context)
            results[agent_name] = agent_result
            
            if not agent_result.success:
                self.logger.error(f"Agent {agent_name} failed: {agent_result.error}")
                return {
                    "recipe_id": recipe_id,
                    "success": False,
                    "error": f"Agent {agent_name} failed",
                    "results": results
                }
            
            # Update context with agent artifacts
            context["artifacts"][agent_name] = agent_result.artifacts
        
        # Collect final metrics
        final_metrics = self._collect_final_metrics(results, context)
        
        # Register experiment
        experiment_data = {
            "recipe": recipe,
            "results": results,
            "metrics": final_metrics,
            "timestamp": time.time()
        }
        self.registry.register_experiment(recipe_id, experiment_data)
        
        return {
            "recipe_id": recipe_id,
            "success": True,
            "metrics": final_metrics,
            "results": results
        }
    
    def execute_recipes_parallel(self, recipes: List[Dict[str, Any]], 
                                max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute multiple recipes in parallel."""
        if max_workers is None:
            max_workers = min(len(recipes), self.config.get("max_parallel_workers", 4))
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_recipe = {
                executor.submit(self.execute_recipe, recipe): recipe 
                for recipe in recipes
            }
            
            for future in as_completed(future_to_recipe):
                recipe = future_to_recipe[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Recipe {result['recipe_id']} completed")
                except Exception as e:
                    self.logger.error(f"Recipe execution failed: {e}")
                    results.append({
                        "recipe_id": recipe.get("id", "unknown"),
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the complete optimization pipeline."""
        start_time = time.time()
        self.logger.info("Starting LLM compression optimization")
        
        # Generate recipes
        recipes = self.generate_recipes()
        
        # Execute recipes
        results = self.execute_recipes_parallel(recipes)
        
        # Analyze Pareto frontier
        successful_results = [r for r in results if r.get("success", False)]
        pareto_candidates = self.pareto.analyze(successful_results)
        
        # Generate summary
        summary = {
            "total_recipes": len(recipes),
            "successful_recipes": len(successful_results),
            "pareto_candidates": len(pareto_candidates),
            "execution_time": time.time() - start_time,
            "best_candidates": pareto_candidates[:3],  # Top 3
            "all_results": results
        }
        
        self.logger.info(
            f"Optimization complete: {len(successful_results)}/{len(recipes)} "
            f"recipes successful, {len(pareto_candidates)} Pareto candidates"
        )
        
        return summary
    
    def _collect_final_metrics(self, results: Dict[str, AgentResult], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and aggregate final metrics from all agents."""
        metrics = {}
        
        # Aggregate metrics from all agents
        for agent_name, result in results.items():
            if result.success:
                for key, value in result.metrics.items():
                    metrics[f"{agent_name}_{key}"] = value
        
        # Calculate composite score
        weights = self.config.get("objective_weights", {
            "accuracy": 1.0,
            "latency": -1.0,
            "vram": -1.0,
            "energy": -1.0,
            "co2e": -1.0
        })
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
        
        metrics["composite_score"] = score
        return metrics