"""Evaluation and safety assessment agent."""

import json
import random
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from .base import BaseAgent, AgentResult


class EvalSafetyAgent(BaseAgent):
    """Agent for model evaluation and safety assessment."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.evaluation_suites = ["mmlu", "gsm8k", "mtbench", "safety"]
        
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Evaluation can run on any optimized model."""
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute evaluation and safety assessment."""
        try:
            # Get model path from previous agents
            model_path = self._get_model_path(context)
            
            # Run evaluations
            evaluation_results = {}
            
            for suite in self.evaluation_suites:
                if self._should_run_evaluation(suite):
                    results = self._run_evaluation_suite(suite, model_path, recipe, context)
                    evaluation_results[suite] = results
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(evaluation_results)
            
            # Assess safety
            safety_assessment = self._assess_safety(model_path, recipe, context)
            
            # Combine all results
            all_metrics = {
                **aggregate_metrics,
                **safety_assessment
            }
            
            artifacts = {
                "evaluation_results": evaluation_results,
                "model_path": model_path,
                "evaluation_config": self.config
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
    
    def _get_model_path(self, context: Dict[str, Any]) -> str:
        """Get the most recent model path from context."""
        artifacts = context.get("artifacts", {})
        
        # Priority order: distillation -> pruning -> kv -> quantization -> base
        for agent in ["distillation", "pruning_sparsity", "kv_longcontext", "quantization"]:
            if agent in artifacts:
                model_path = artifacts[agent].get("optimized_model_path") or \
                           artifacts[agent].get("quantized_model_path") or \
                           artifacts[agent].get("pruned_model_path") or \
                           artifacts[agent].get("distilled_model_path")
                if model_path:
                    return model_path
        
        return context.get("model_path", "mock_model")
    
    def _should_run_evaluation(self, suite: str) -> bool:
        """Check if evaluation suite should be run."""
        suite_config = self.config.get("evaluations", {}).get(suite, {})
        return suite_config.get("enabled", True)
    
    def _run_evaluation_suite(self, suite: str, model_path: str, 
                             recipe: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific evaluation suite."""
        
        if suite == "mmlu":
            return self._run_mmlu_evaluation(model_path, recipe, context)
        elif suite == "gsm8k":
            return self._run_gsm8k_evaluation(model_path, recipe, context)
        elif suite == "mtbench":
            return self._run_mtbench_evaluation(model_path, recipe, context)
        elif suite == "safety":
            return self._run_safety_evaluation(model_path, recipe, context)
        else:
            raise ValueError(f"Unknown evaluation suite: {suite}")
    
    def _run_mmlu_evaluation(self, model_path: str, recipe: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Run MMLU evaluation."""
        self.logger.info("Running MMLU evaluation")
        
        # Mock MMLU evaluation - in real implementation, load dataset and run inference
        base_accuracy = 0.85  # Base model accuracy
        
        # Apply optimization penalties
        accuracy_drop = self._estimate_accuracy_drop(recipe)
        final_accuracy = max(0.0, base_accuracy - accuracy_drop)
        
        # Simulate per-subject results
        subjects = ["mathematics", "physics", "chemistry", "biology", "history", 
                   "literature", "philosophy", "economics", "computer_science", "law"]
        
        subject_results = {}
        for subject in subjects:
            # Add some variance per subject
            subject_acc = final_accuracy + np.random.normal(0, 0.02)
            subject_acc = max(0.0, min(1.0, subject_acc))
            subject_results[subject] = {
                "accuracy": subject_acc,
                "num_samples": 100,  # Mock sample count
                "correct": int(subject_acc * 100)
            }
        
        return {
            "overall_accuracy": final_accuracy,
            "subject_results": subject_results,
            "total_samples": len(subjects) * 100,
            "evaluation_time_minutes": 15
        }
    
    def _run_gsm8k_evaluation(self, model_path: str, recipe: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Run GSM8K mathematical reasoning evaluation."""
        self.logger.info("Running GSM8K evaluation")
        
        base_accuracy = 0.78  # Base model accuracy for math
        
        # Math reasoning is more sensitive to quantization
        accuracy_drop = self._estimate_accuracy_drop(recipe) * 1.2
        final_accuracy = max(0.0, base_accuracy - accuracy_drop)
        
        # Mock detailed results
        total_samples = 500
        correct_answers = int(final_accuracy * total_samples)
        
        return {
            "accuracy": final_accuracy,
            "correct_answers": correct_answers,
            "total_samples": total_samples,
            "avg_reasoning_steps": 4.2,
            "evaluation_time_minutes": 25
        }
    
    def _run_mtbench_evaluation(self, model_path: str, recipe: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Run MT-Bench conversational evaluation."""
        self.logger.info("Running MT-Bench evaluation")
        
        base_score = 7.5  # Base score out of 10
        
        # Conversational ability impact
        score_drop = self._estimate_accuracy_drop(recipe) * 5  # Scale to 0-10
        final_score = max(1.0, base_score - score_drop)
        
        # Mock category scores
        categories = ["writing", "roleplay", "reasoning", "math", "coding", 
                     "extraction", "stem", "humanities"]
        
        category_scores = {}
        for category in categories:
            category_score = final_score + np.random.normal(0, 0.3)
            category_score = max(1.0, min(10.0, category_score))
            category_scores[category] = category_score
        
        return {
            "overall_score": final_score,
            "category_scores": category_scores,
            "total_conversations": 80,
            "evaluation_time_minutes": 45
        }
    
    def _run_safety_evaluation(self, model_path: str, recipe: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Run safety evaluation."""
        self.logger.info("Running safety evaluation")
        
        # Mock safety evaluation
        safety_categories = ["toxicity", "bias", "truthfulness", "privacy", "jailbreak_resistance"]
        
        safety_scores = {}
        for category in safety_categories:
            # Safety should be relatively stable across optimizations
            base_score = 0.95
            score_variation = np.random.normal(0, 0.02)
            safety_score = max(0.7, min(1.0, base_score + score_variation))
            safety_scores[category] = safety_score
        
        overall_safety = np.mean(list(safety_scores.values()))
        
        return {
            "overall_safety_score": overall_safety,
            "category_scores": safety_scores,
            "red_team_attempts": 100,
            "successful_jailbreaks": max(0, int((1 - overall_safety) * 100)),
            "evaluation_time_minutes": 30
        }
    
    def _estimate_accuracy_drop(self, recipe: Dict[str, Any]) -> float:
        """Estimate accuracy drop from optimizations."""
        total_drop = 0.0
        
        # Quantization impact
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            method = recipe["quantization"].get("method", "awq")
            
            drop_map = {
                ("awq", 4): 0.005,
                ("awq", 8): 0.002,
                ("gptq", 4): 0.008,
                ("gptq", 8): 0.003,
                ("bnb", 4): 0.015,
                ("bnb", 8): 0.008
            }
            total_drop += drop_map.get((method, bits), 0.01)
        
        # Pruning impact
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            head_ratio = prune_config.get("head_pruning_ratio", 0.0)
            ffn_ratio = prune_config.get("ffn_pruning_ratio", 0.0)
            total_drop += (head_ratio + ffn_ratio) * 0.1
        
        # Distillation impact
        if recipe.get("distillation", {}).get("enabled", False):
            total_drop += 0.01  # Small drop from distillation
        
        # KV optimization generally doesn't impact accuracy much
        if recipe.get("kv_longcontext", {}).get("enabled", False):
            if recipe["kv_longcontext"].get("attention_type") == "sliding_window":
                window_size = recipe["kv_longcontext"].get("sliding_window_size", 4096)
                if window_size < 4096:
                    total_drop += 0.005  # Small drop for reduced context
        
        return total_drop
    
    def _calculate_aggregate_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all evaluations."""
        metrics = {}
        
        # Extract key metrics
        if "mmlu" in evaluation_results:
            metrics["accuracy"] = evaluation_results["mmlu"]["overall_accuracy"]
            metrics["mmlu_accuracy"] = evaluation_results["mmlu"]["overall_accuracy"]
        
        if "gsm8k" in evaluation_results:
            metrics["math_accuracy"] = evaluation_results["gsm8k"]["accuracy"]
        
        if "mtbench" in evaluation_results:
            metrics["conversational_score"] = evaluation_results["mtbench"]["overall_score"]
        
        if "safety" in evaluation_results:
            metrics["safety_score"] = evaluation_results["safety"]["overall_safety_score"]
        
        # Calculate composite accuracy (if multiple accuracy metrics available)
        accuracy_metrics = []
        for key in ["accuracy", "math_accuracy"]:
            if key in metrics:
                accuracy_metrics.append(metrics[key])
        
        if accuracy_metrics:
            metrics["composite_accuracy"] = np.mean(accuracy_metrics)
        
        # Total evaluation time
        total_time = sum(
            result.get("evaluation_time_minutes", 0) 
            for result in evaluation_results.values()
        )
        metrics["total_evaluation_time_minutes"] = total_time
        
        return metrics
    
    def _assess_safety(self, model_path: str, recipe: Dict[str, Any], 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model safety post-optimization."""
        
        # Check for potential safety issues from optimizations
        safety_issues = []
        
        # Aggressive quantization can sometimes cause instability
        if recipe.get("quantization", {}).get("enabled", False):
            bits = recipe["quantization"].get("bits", 8)
            if bits <= 4:
                safety_issues.append("low_bit_quantization")
        
        # Heavy pruning can affect model behavior
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            prune_config = recipe["pruning_sparsity"]
            total_pruning = prune_config.get("head_pruning_ratio", 0) + \
                           prune_config.get("ffn_pruning_ratio", 0)
            if total_pruning > 0.4:
                safety_issues.append("heavy_pruning")
        
        return {
            "safety_assessment_complete": True,
            "potential_safety_issues": safety_issues,
            "safety_risk_level": "low" if len(safety_issues) == 0 else "medium"
        }
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate evaluation cost."""
        # Base evaluation time
        base_time = 60  # 1 hour for full evaluation
        
        # Number of evaluation suites to run
        enabled_suites = sum(
            1 for suite in self.evaluation_suites 
            if self._should_run_evaluation(suite)
        )
        
        total_time = base_time * (enabled_suites / len(self.evaluation_suites))
        
        return {
            "time": total_time,
            "memory": 2.0,  # Moderate memory for inference
            "energy": total_time * 0.5  # GPU time for inference
        }