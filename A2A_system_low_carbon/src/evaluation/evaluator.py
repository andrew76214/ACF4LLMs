"""
Main Evaluator for A2A Pipeline
Orchestrates evaluation across different datasets and metrics
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from agents.router import RouterAgent
from agents.search import SearchAgent
from agents.solver import SolverAgent
from agents.judge import JudgeAgent
from models.vllm_client import VLLMClient
from models.model_manager import ModelManager
from utils.data_processing import load_dataset_by_name, save_results, sample_dataset
from utils.logging import get_logger, PerformanceLogger
from utils.metrics import format_metrics_report, compute_variance_metrics
from configs.config import config

logger = get_logger("evaluator")

class Evaluator:
    """
    Main evaluator that orchestrates the entire A2A pipeline evaluation
    """

    def __init__(self, model_client: Optional[Any] = None):
        """
        Initialize Evaluator

        Args:
            model_client: Optional model client (will initialize default if None)
        """
        self.config = config
        self.model_client = model_client

        # Initialize agents
        self.router = RouterAgent()
        self.search_agent = SearchAgent()
        self.solver = SolverAgent(model_client)
        self.judge = JudgeAgent()

        logger.info("A2A Pipeline Evaluator initialized")

    def evaluate_single_question(
        self,
        question: str,
        gold_answer: Any,
        dataset: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through the full pipeline

        Args:
            question: Question to evaluate
            gold_answer: Ground truth answer
            dataset: Dataset name for routing
            context: Optional context
            **kwargs: Additional parameters

        Returns:
            Evaluation results
        """
        start_time = time.time()

        try:
            # Step 1: Route the question
            routing_decision = self.router.route(question, dataset, context)
            logger.debug(f"Routed to {routing_decision.task_type}")

            # Step 2: Search if needed
            retrieval_context = None
            if routing_decision.use_search:
                if routing_decision.task_type == "multihop":
                    retrieval_context = self.search_agent.multi_hop_search(question)
                else:
                    retrieval_context = self.search_agent.retrieve_and_rerank(question)
                logger.debug(f"Retrieved {len(retrieval_context.passages)} passages")

            # Step 3: Solve
            solver_response = self.solver.solve(question, routing_decision, retrieval_context)
            logger.debug(f"Generated answer: {solver_response.answer[:50]}...")

            # Step 4: Judge
            judgment = self.judge.judge_single(
                solver_response.answer,
                gold_answer,
                routing_decision.task_type
            )

            # Compile results
            result = {
                "question": question,
                "prediction": solver_response.answer,
                "gold_answer": gold_answer,
                "correct": judgment.correct,
                "score": judgment.score,
                "task_type": routing_decision.task_type,
                "used_search": routing_decision.use_search,
                "processing_time": time.time() - start_time,
                "routing": {
                    "task_type": routing_decision.task_type,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning
                },
                "solver": {
                    "reasoning": solver_response.reasoning,
                    "confidence": solver_response.confidence,
                    "metadata": solver_response.metadata
                },
                "judgment": {
                    "explanation": judgment.explanation,
                    "details": judgment.details
                }
            }

            if retrieval_context:
                result["retrieval"] = {
                    "num_passages": len(retrieval_context.passages),
                    "search_time": retrieval_context.search_time,
                    "sources": [p.url for p in retrieval_context.passages[:3]]
                }

            return result

        except Exception as e:
            logger.error(f"Single question evaluation failed: {e}")
            return {
                "question": question,
                "prediction": "",
                "gold_answer": gold_answer,
                "correct": False,
                "score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def evaluate_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        num_trials: int = 1,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on a complete dataset

        Args:
            dataset_name: Name of dataset to evaluate
            sample_size: Number of samples (None for full dataset)
            num_trials: Number of trials for variance analysis
            save_predictions: Whether to save predictions

        Returns:
            Evaluation results with metrics
        """
        logger.info(f"Starting evaluation on {dataset_name}")

        try:
            # Load dataset
            data = load_dataset_by_name(dataset_name)
            if sample_size:
                data = sample_dataset(data, sample_size)

            logger.info(f"Loaded {len(data)} questions from {dataset_name}")

            # Run multiple trials if specified
            all_trial_results = []
            trial_metrics = []

            for trial in range(num_trials):
                if num_trials > 1:
                    logger.info(f"Starting trial {trial + 1}/{num_trials}")

                # Evaluate all questions in this trial
                trial_results = []
                with PerformanceLogger(f"evaluating {len(data)} questions"):
                    for i, item in enumerate(data):
                        if (i + 1) % 10 == 0:
                            logger.info(f"Progress: {i + 1}/{len(data)}")

                        # Extract question and answer based on dataset format
                        question, gold_answer = self._extract_question_answer(item, dataset_name)

                        # Evaluate
                        result = self.evaluate_single_question(
                            question, gold_answer, dataset_name
                        )
                        result["trial"] = trial
                        result["question_id"] = i
                        trial_results.append(result)

                all_trial_results.extend(trial_results)

                # Compute trial metrics
                trial_judgments = [
                    self.judge.judge_single(r["prediction"], r["gold_answer"], r.get("task_type", "qa"))
                    for r in trial_results
                ]
                trial_metric = self.judge.compute_dataset_metrics(trial_judgments, dataset_name)
                trial_metrics.append(trial_metric["accuracy"])

                logger.info(f"Trial {trial + 1} accuracy: {trial_metric['accuracy']:.3f}")

            # Aggregate results across trials
            final_metrics = self._aggregate_trial_metrics(all_trial_results, trial_metrics, dataset_name)

            # Save results
            if save_predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"{dataset_name}_results_{timestamp}.json"
                save_results({
                    "dataset": dataset_name,
                    "metrics": final_metrics,
                    "predictions": all_trial_results,
                    "config": self._get_evaluation_config()
                }, results_file)

            logger.info(f"Evaluation completed: {final_metrics.get('accuracy', 0):.3f} accuracy")
            return final_metrics

        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
            return {"error": str(e), "dataset": dataset_name}

    def evaluate_multiple_datasets(
        self,
        dataset_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple datasets

        Args:
            dataset_names: List of dataset names
            **kwargs: Arguments passed to evaluate_dataset

        Returns:
            Combined evaluation results
        """
        logger.info(f"Evaluating on {len(dataset_names)} datasets")

        all_results = {}
        summary_metrics = {}

        for dataset_name in dataset_names:
            logger.info(f"Starting evaluation on {dataset_name}")
            try:
                results = self.evaluate_dataset(dataset_name, **kwargs)
                all_results[dataset_name] = results
                summary_metrics[dataset_name] = {
                    "accuracy": results.get("accuracy", 0.0),
                    "count": results.get("total_count", 0)
                }
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                all_results[dataset_name] = {"error": str(e)}
                summary_metrics[dataset_name] = {"accuracy": 0.0, "count": 0}

        # Generate summary report
        report = self._generate_summary_report(summary_metrics, all_results)

        return {
            "summary": summary_metrics,
            "detailed_results": all_results,
            "report": report
        }

    def _extract_question_answer(self, item: Dict, dataset_name: str) -> tuple:
        """Extract question and answer from dataset item"""
        dataset_lower = dataset_name.lower()

        if "gsm8k" in dataset_lower:
            return item.get("question", ""), item.get("answer", "")
        elif "aime" in dataset_lower:
            return item.get("question", item.get("problem", "")), item.get("answer", "")
        elif "math" in dataset_lower:
            return item.get("problem", ""), item.get("solution", item.get("answer", ""))
        elif "hotpot" in dataset_lower:
            return item.get("question", ""), {
                "answer": item.get("answer", ""),
                "supporting_facts": item.get("supporting_facts", [])
            }
        elif "squad" in dataset_lower:
            answers = item.get("answers", {})
            if isinstance(answers, dict) and "text" in answers:
                gold_answers = answers["text"]
            else:
                gold_answers = [item.get("answer", "")]
            return item.get("question", ""), gold_answers
        elif "nq" in dataset_lower or "natural" in dataset_lower:
            return item.get("question", ""), item.get("answer", item.get("answers", []))
        else:
            # Generic format
            return item.get("question", item.get("input", "")), item.get("answer", item.get("output", ""))

    def _aggregate_trial_metrics(
        self,
        all_results: List[Dict],
        trial_accuracies: List[float],
        dataset_name: str
    ) -> Dict[str, Any]:
        """Aggregate metrics across multiple trials"""
        # Basic aggregation
        correct_count = sum(1 for r in all_results if r.get("correct", False))
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "dataset": dataset_name
        }

        # Variance analysis if multiple trials
        if len(trial_accuracies) > 1:
            import numpy as np
            variance_metrics = {
                "mean_accuracy": np.mean(trial_accuracies),
                "std_accuracy": np.std(trial_accuracies),
                "min_accuracy": np.min(trial_accuracies),
                "max_accuracy": np.max(trial_accuracies),
                "num_trials": len(trial_accuracies)
            }
            metrics.update(variance_metrics)

        # Task-specific metrics
        task_types = [r.get("task_type") for r in all_results]
        if task_types and task_types[0]:
            task_type = task_types[0]

            # Aggregate judgments for detailed metrics
            all_judgments = []
            all_predictions = []
            all_gold_answers = []

            for result in all_results:
                all_predictions.append(result.get("prediction", ""))
                all_gold_answers.append(result.get("gold_answer"))

                # Create judgment from result
                judgment = self.judge.judge_single(
                    result.get("prediction", ""),
                    result.get("gold_answer"),
                    task_type
                )
                all_judgments.append(judgment)

            detailed_metrics = self.judge.compute_dataset_metrics(all_judgments, task_type)
            metrics.update(detailed_metrics)

        return metrics

    def _generate_summary_report(
        self,
        summary_metrics: Dict[str, Any],
        detailed_results: Dict[str, Any]
    ) -> str:
        """Generate a summary report of evaluation results"""
        report_lines = [
            "=== A2A PIPELINE EVALUATION REPORT ===",
            f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Overall summary
        total_questions = sum(m.get("count", 0) for m in summary_metrics.values())
        avg_accuracy = sum(m.get("accuracy", 0) for m in summary_metrics.values()) / len(summary_metrics)

        report_lines.extend([
            f"Total datasets evaluated: {len(summary_metrics)}",
            f"Total questions: {total_questions}",
            f"Average accuracy: {avg_accuracy:.3f}",
            ""
        ])

        # Per-dataset results
        for dataset, metrics in summary_metrics.items():
            if "error" not in detailed_results.get(dataset, {}):
                report_lines.append(f"{dataset.upper()}:")
                report_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
                report_lines.append(f"  Questions: {metrics.get('count', 0)}")

                # Add detailed metrics if available
                detailed = detailed_results.get(dataset, {})
                if "mean_accuracy" in detailed:
                    report_lines.append(f"  Mean±Std: {detailed['mean_accuracy']:.3f}±{detailed.get('std_accuracy', 0):.3f}")

                report_lines.append("")

        report_lines.append("=" * 50)
        return "\n".join(report_lines)

    def _get_evaluation_config(self) -> Dict[str, Any]:
        """Get current evaluation configuration"""
        return {
            "model": self.config.model.model_name,
            "quantization": self.config.model.quantization,
            "search_enabled": bool(self.config.search.tavily_api_key),
            "timestamp": datetime.now().isoformat()
        }

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all configured datasets"""
        datasets = ["gsm8k", "aime", "hotpotqa", "squad_v1"]

        logger.info("Starting comprehensive evaluation")
        results = self.evaluate_multiple_datasets(datasets, num_trials=config.experiment.num_trials_math)

        # Print report
        print(results["report"])

        return results

    def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components"""
        health = {
            "router": True,  # Router is always available
            "solver": self.solver.get_solver_info(),
            "judge": self.judge.get_judge_info(),
            "search": self.search_agent.health_check(),
            "timestamp": datetime.now().isoformat()
        }

        # Test with a simple question
        try:
            test_result = self.evaluate_single_question(
                "What is 2+2?", "4", "test"
            )
            health["pipeline_test"] = {
                "success": test_result.get("correct", False),
                "time": test_result.get("processing_time", 0)
            }
        except Exception as e:
            health["pipeline_test"] = {"success": False, "error": str(e)}

        return health