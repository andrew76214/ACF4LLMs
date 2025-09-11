"""Data curation agent for calibration and evaluation datasets."""

import json
import random
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from .base import BaseAgent, AgentResult


class DataCurationAgent(BaseAgent):
    """Agent for curating and preparing calibration and evaluation datasets."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.dataset_cache = {}
        
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Data curation is always valid."""
        return True
    
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute data curation."""
        try:
            # Prepare calibration data for quantization
            calibration_data = self._prepare_calibration_data(recipe, context)
            
            # Prepare evaluation datasets
            evaluation_data = self._prepare_evaluation_data(recipe, context)
            
            # Validate data quality
            quality_metrics = self._validate_data_quality(calibration_data, evaluation_data)
            
            artifacts = {
                "calibration_data": calibration_data,
                "evaluation_data": evaluation_data,
                "data_statistics": quality_metrics
            }
            
            metrics = {
                "calibration_samples": len(calibration_data.get("samples", [])),
                "evaluation_samples": sum(len(d.get("samples", [])) for d in evaluation_data.values()),
                **quality_metrics
            }
            
            return AgentResult(
                success=True,
                metrics=metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e)
            )
    
    def _prepare_calibration_data(self, recipe: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare calibration data for quantization."""
        
        # Get calibration requirements from recipe
        quant_config = recipe.get("quantization", {})
        num_samples = quant_config.get("calibration_samples", 512)
        sequence_length = context.get("config", {}).get("sequence_length", 512)
        
        self.logger.info(f"Preparing {num_samples} calibration samples")
        
        # Load or generate calibration data
        if "calibration" not in self.dataset_cache:
            self.dataset_cache["calibration"] = self._load_calibration_dataset()
        
        # Sample and prepare data
        all_samples = self.dataset_cache["calibration"]
        sampled_data = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        # Process samples to fit sequence length
        processed_samples = []
        for sample in sampled_data:
            processed_sample = self._process_calibration_sample(sample, sequence_length)
            processed_samples.append(processed_sample)
        
        return {
            "samples": processed_samples,
            "num_samples": len(processed_samples),
            "sequence_length": sequence_length,
            "data_source": "c4_subset"  # Mock data source
        }
    
    def _prepare_evaluation_data(self, recipe: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare evaluation datasets."""
        
        eval_config = context.get("config", {}).get("evaluation", {})
        datasets = {}
        
        # Prepare MMLU subset
        if eval_config.get("mmlu", {}).get("enabled", True):
            datasets["mmlu"] = self._prepare_mmlu_data(eval_config.get("mmlu", {}))
        
        # Prepare GSM8K subset  
        if eval_config.get("gsm8k", {}).get("enabled", True):
            datasets["gsm8k"] = self._prepare_gsm8k_data(eval_config.get("gsm8k", {}))
        
        # Prepare MT-Bench subset
        if eval_config.get("mtbench", {}).get("enabled", True):
            datasets["mtbench"] = self._prepare_mtbench_data(eval_config.get("mtbench", {}))
        
        return datasets
    
    def _load_calibration_dataset(self) -> List[Dict[str, Any]]:
        """Load calibration dataset (C4 subset)."""
        # In real implementation, load from HuggingFace or local files
        # For now, generate mock data
        
        mock_samples = []
        for i in range(2000):  # Generate 2000 mock samples
            sample = {
                "text": f"This is sample calibration text number {i}. " * 20,
                "source": "c4",
                "length": random.randint(100, 1000)
            }
            mock_samples.append(sample)
        
        return mock_samples
    
    def _process_calibration_sample(self, sample: Dict[str, Any], 
                                   max_length: int) -> Dict[str, Any]:
        """Process calibration sample to fit length requirements."""
        text = sample["text"]
        
        # Truncate or pad to max_length (simplified tokenization)
        words = text.split()
        if len(words) > max_length // 4:  # Rough token estimation
            words = words[:max_length // 4]
        
        processed_text = " ".join(words)
        
        return {
            "text": processed_text,
            "token_count": len(words),
            "source": sample.get("source", "unknown")
        }
    
    def _prepare_mmlu_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare MMLU evaluation data."""
        num_samples = config.get("num_samples", 100)
        
        # Load from cache or generate mock data
        if "mmlu" not in self.dataset_cache:
            self.dataset_cache["mmlu"] = self._generate_mock_mmlu_data()
        
        all_samples = self.dataset_cache["mmlu"]
        sampled_data = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        return {
            "samples": sampled_data,
            "num_samples": len(sampled_data),
            "subjects": ["mathematics", "physics", "chemistry", "biology", "history"],
            "format": "multiple_choice"
        }
    
    def _prepare_gsm8k_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare GSM8K evaluation data."""
        num_samples = config.get("num_samples", 100)
        
        if "gsm8k" not in self.dataset_cache:
            self.dataset_cache["gsm8k"] = self._generate_mock_gsm8k_data()
        
        all_samples = self.dataset_cache["gsm8k"]
        sampled_data = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        return {
            "samples": sampled_data,
            "num_samples": len(sampled_data),
            "format": "math_word_problems"
        }
    
    def _prepare_mtbench_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare MT-Bench evaluation data."""
        num_samples = config.get("num_samples", 80)
        
        if "mtbench" not in self.dataset_cache:
            self.dataset_cache["mtbench"] = self._generate_mock_mtbench_data()
        
        all_samples = self.dataset_cache["mtbench"]
        sampled_data = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        return {
            "samples": sampled_data,
            "num_samples": len(sampled_data),
            "categories": ["writing", "roleplay", "reasoning", "math", "coding"],
            "format": "conversational"
        }
    
    def _generate_mock_mmlu_data(self) -> List[Dict[str, Any]]:
        """Generate mock MMLU data."""
        subjects = ["mathematics", "physics", "chemistry", "biology", "history", 
                   "literature", "philosophy", "economics", "computer_science", "law"]
        
        samples = []
        for i in range(1000):
            subject = random.choice(subjects)
            sample = {
                "id": f"mmlu_{i}",
                "subject": subject,
                "question": f"Mock MMLU question {i} for {subject}",
                "choices": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "answer": random.choice(["A", "B", "C", "D"]),
                "difficulty": random.choice(["easy", "medium", "hard"])
            }
            samples.append(sample)
        
        return samples
    
    def _generate_mock_gsm8k_data(self) -> List[Dict[str, Any]]:
        """Generate mock GSM8K data."""
        samples = []
        for i in range(500):
            sample = {
                "id": f"gsm8k_{i}",
                "question": f"Mock math word problem {i}: John has {random.randint(1, 100)} apples...",
                "answer": random.randint(1, 1000),
                "solution_steps": [
                    "Step 1: Identify the given information",
                    "Step 2: Set up the equation", 
                    "Step 3: Solve for the unknown",
                    "Step 4: Verify the answer"
                ]
            }
            samples.append(sample)
        
        return samples
    
    def _generate_mock_mtbench_data(self) -> List[Dict[str, Any]]:
        """Generate mock MT-Bench data."""
        categories = ["writing", "roleplay", "reasoning", "math", "coding", 
                     "extraction", "stem", "humanities"]
        
        samples = []
        for i in range(160):  # 20 per category
            category = categories[i % len(categories)]
            sample = {
                "id": f"mtbench_{i}",
                "category": category,
                "conversation": [
                    {
                        "role": "user",
                        "content": f"Mock {category} prompt {i}"
                    }
                ],
                "reference_answer": f"Mock reference answer for {category} task {i}",
                "scoring_criteria": ["relevance", "coherence", "accuracy", "creativity"]
            }
            samples.append(sample)
        
        return samples
    
    def _validate_data_quality(self, calibration_data: Dict[str, Any],
                              evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality metrics."""
        
        quality_metrics = {}
        
        # Calibration data quality
        calib_samples = calibration_data.get("samples", [])
        if calib_samples:
            avg_length = sum(s.get("token_count", 0) for s in calib_samples) / len(calib_samples)
            quality_metrics["calibration_avg_length"] = avg_length
            quality_metrics["calibration_coverage"] = 1.0  # Mock coverage score
        
        # Evaluation data quality
        for dataset_name, dataset in evaluation_data.items():
            samples = dataset.get("samples", [])
            quality_metrics[f"{dataset_name}_samples"] = len(samples)
            quality_metrics[f"{dataset_name}_quality_score"] = 0.95  # Mock quality score
        
        # Data diversity metrics
        quality_metrics["data_diversity_score"] = 0.85
        quality_metrics["data_balance_score"] = 0.90
        
        return quality_metrics
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate data curation cost."""
        
        # Cost depends on amount of data to process
        quant_config = recipe.get("quantization", {})
        num_calibration_samples = quant_config.get("calibration_samples", 512)
        
        # Time scales with number of samples
        time_cost = 5 + (num_calibration_samples / 100) * 2  # Base 5 min + processing time
        
        return {
            "time": time_cost,
            "memory": 1.0,  # Minimal memory for data processing
            "energy": 2.0   # Minimal energy for data ops
        }