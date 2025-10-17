"""
Judge Agent for A2A Pipeline
Handles evaluation using Math-Verify and official dataset metrics
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from utils.math_verify import verify, parse_answer, extract_and_verify_answer
    math_verify_available = True
except ImportError:
    verify = parse_answer = extract_and_verify_answer = None
    math_verify_available = False

from utils.logging import get_logger
from utils.metrics import (
    compute_accuracy, compute_f1, compute_exact_match,
    compute_supporting_facts_metrics, compute_joint_metrics
)
from utils.text_processing import extract_json_answer, normalize_text

logger = get_logger("judge_agent")

@dataclass
class JudgmentResult:
    """Result of evaluation for a single question"""
    correct: bool
    score: float
    details: Dict[str, Any]
    explanation: Optional[str] = None

class JudgeAgent:
    """
    Judge Agent that evaluates model predictions against gold answers
    Uses Math-Verify for math problems and official metrics for QA
    """

    def __init__(self, evaluation_config=None):
        """Initialize JudgeAgent"""
        from configs.config import config
        self.config = evaluation_config or config.evaluation

        # Check Math-Verify availability
        self.math_verify_available = math_verify_available
        if not self.math_verify_available:
            logger.warning("Math-Verify not available - using basic string matching for math")
        else:
            logger.info("Math-Verify successfully loaded for mathematical equivalence checking")

    def judge_single(
        self,
        prediction: str,
        gold_answer: Any,
        task_type: str,
        question_id: Optional[str] = None,
        **kwargs
    ) -> JudgmentResult:
        """
        Judge a single prediction

        Args:
            prediction: Model's predicted answer
            gold_answer: Ground truth answer
            task_type: Type of task (math, qa, multihop)
            question_id: Question identifier
            **kwargs: Additional context

        Returns:
            JudgmentResult
        """
        try:
            if task_type == "math":
                return self._judge_math(prediction, gold_answer, question_id)
            elif task_type == "multihop":
                return self._judge_multihop(prediction, gold_answer, question_id)
            elif task_type == "multiple_choice":
                return self._judge_multiple_choice(prediction, gold_answer, question_id)
            elif task_type in ["qa", "open_qa"]:
                return self._judge_qa(prediction, gold_answer, question_id)
            else:
                logger.warning(f"Unknown task type: {task_type}, defaulting to QA")
                return self._judge_qa(prediction, gold_answer, question_id)

        except Exception as e:
            logger.error(f"Judgment failed: {e}")
            return JudgmentResult(
                correct=False,
                score=0.0,
                details={"error": str(e)},
                explanation=f"Evaluation error: {e}"
            )

    def _judge_math(self, prediction: str, gold_answer: str, question_id: Optional[str]) -> JudgmentResult:
        """Judge mathematical answers"""
        try:
            # Extract answer from prediction using Math-Verify if available
            if self.math_verify_available:
                try:
                    pred_parsed = parse_answer(prediction)
                    gold_parsed = parse_answer(str(gold_answer))

                    # Check equivalence
                    is_correct = verify(gold_parsed, pred_parsed)

                    return JudgmentResult(
                        correct=is_correct,
                        score=1.0 if is_correct else 0.0,
                        details={
                            "prediction_parsed": str(pred_parsed),
                            "gold_parsed": str(gold_parsed),
                            "method": "math_verify"
                        },
                        explanation=f"Math-Verify: {'Correct' if is_correct else 'Incorrect'}"
                    )

                except Exception as e:
                    logger.debug(f"Math-Verify failed, falling back to string matching: {e}")

            # Fallback to string matching
            from utils.text_processing import extract_boxed_answer, extract_number_from_text

            # Try to extract boxed answer
            pred_answer = extract_boxed_answer(prediction)
            if not pred_answer:
                pred_answer = extract_number_from_text(prediction)
            if not pred_answer:
                pred_answer = prediction.strip()

            # Normalize for comparison
            pred_normalized = normalize_text(pred_answer)
            gold_normalized = normalize_text(str(gold_answer))

            is_correct = pred_normalized == gold_normalized

            return JudgmentResult(
                correct=is_correct,
                score=1.0 if is_correct else 0.0,
                details={
                    "prediction_normalized": pred_normalized,
                    "gold_normalized": gold_normalized,
                    "method": "string_matching"
                },
                explanation=f"String matching: {'Correct' if is_correct else 'Incorrect'}"
            )

        except Exception as e:
            logger.error(f"Math judgment failed: {e}")
            return JudgmentResult(
                correct=False,
                score=0.0,
                details={"error": str(e)},
                explanation=f"Math evaluation error: {e}"
            )

    def _judge_qa(self, prediction: str, gold_answer: Any, question_id: Optional[str]) -> JudgmentResult:
        """Judge QA answers"""
        try:
            # Handle multiple gold answers (common in QA datasets)
            if isinstance(gold_answer, list):
                gold_answers = [str(ans) for ans in gold_answer]
            else:
                gold_answers = [str(gold_answer)]

            # Compute metrics against all gold answers
            best_em = 0
            best_f1 = 0.0

            for gold in gold_answers:
                # Exact match
                em = 1 if normalize_text(prediction) == normalize_text(gold) else 0
                best_em = max(best_em, em)

                # F1 score
                f1 = compute_f1(prediction, gold)
                best_f1 = max(best_f1, f1)

            # Question is correct if EM=1 or F1 > threshold
            is_correct = best_em == 1 or best_f1 > 0.8

            return JudgmentResult(
                correct=is_correct,
                score=best_f1,  # Use F1 as continuous score
                details={
                    "exact_match": best_em,
                    "f1_score": best_f1,
                    "prediction_normalized": normalize_text(prediction),
                    "gold_answers": gold_answers,
                    "method": "qa_metrics"
                },
                explanation=f"QA: EM={best_em}, F1={best_f1:.3f}"
            )

        except Exception as e:
            logger.error(f"QA judgment failed: {e}")
            return JudgmentResult(
                correct=False,
                score=0.0,
                details={"error": str(e)},
                explanation=f"QA evaluation error: {e}"
            )

    def _judge_multihop(self, prediction: str, gold_data: Dict, question_id: Optional[str]) -> JudgmentResult:
        """Judge multi-hop QA answers (like HotpotQA)"""
        try:
            # Parse prediction JSON
            pred_data = extract_json_answer(prediction)
            if not pred_data:
                # Fallback: treat as plain answer
                pred_data = {"answer": prediction.strip(), "supporting_facts": []}

            # Extract components
            pred_answer = pred_data.get("answer", "")
            pred_facts = pred_data.get("supporting_facts", [])

            gold_answer = gold_data.get("answer", "")
            gold_facts = gold_data.get("supporting_facts", [])

            # Judge answer
            answer_em = 1 if normalize_text(pred_answer) == normalize_text(gold_answer) else 0
            answer_f1 = compute_f1(pred_answer, gold_answer)

            # Judge supporting facts
            support_metrics = compute_supporting_facts_metrics([pred_facts], [gold_facts])
            support_em = support_metrics["supporting_facts_em"]
            support_f1 = support_metrics["supporting_facts_f1"]

            # Joint metrics
            joint_em = answer_em * support_em
            joint_f1 = (answer_f1 + support_f1) / 2

            # Overall correctness (strict: both answer and facts correct)
            is_correct = joint_em == 1

            return JudgmentResult(
                correct=is_correct,
                score=joint_f1,
                details={
                    "answer_em": answer_em,
                    "answer_f1": answer_f1,
                    "support_em": support_em,
                    "support_f1": support_f1,
                    "joint_em": joint_em,
                    "joint_f1": joint_f1,
                    "prediction_parsed": pred_data,
                    "method": "multihop_metrics"
                },
                explanation=f"Multihop: Answer EM={answer_em}, Support F1={support_f1:.3f}, Joint EM={joint_em}"
            )

        except Exception as e:
            logger.error(f"Multihop judgment failed: {e}")
            return JudgmentResult(
                correct=False,
                score=0.0,
                details={"error": str(e)},
                explanation=f"Multihop evaluation error: {e}"
            )

    def _judge_multiple_choice(self, prediction: str, gold_answer: str, question_id: Optional[str]) -> JudgmentResult:
        """Judge multiple choice answers (A, B, C, D, E)"""
        try:
            # Normalize both answers for comparison
            pred_normalized = normalize_text(prediction).upper()
            gold_normalized = normalize_text(str(gold_answer)).upper()

            # Check if prediction is a single letter
            if len(pred_normalized) == 1 and pred_normalized in ['A', 'B', 'C', 'D', 'E']:
                is_correct = pred_normalized == gold_normalized
            else:
                # Try to extract letter from prediction
                import re
                matches = re.findall(r'\b([A-E])\b', prediction.upper())
                if matches:
                    pred_normalized = matches[-1]  # Take the last match
                    is_correct = pred_normalized == gold_normalized
                else:
                    # No valid letter found
                    is_correct = False
                    pred_normalized = prediction.strip()

            return JudgmentResult(
                correct=is_correct,
                score=1.0 if is_correct else 0.0,
                details={
                    "prediction_normalized": pred_normalized,
                    "gold_normalized": gold_normalized,
                    "method": "multiple_choice_exact_match"
                },
                explanation=f"Multiple choice: {'Correct' if is_correct else 'Incorrect'} ({pred_normalized} vs {gold_normalized})"
            )

        except Exception as e:
            logger.error(f"Multiple choice judgment failed: {e}")
            return JudgmentResult(
                correct=False,
                score=0.0,
                details={"error": str(e)},
                explanation=f"Multiple choice evaluation error: {e}"
            )

    def judge_batch(
        self,
        predictions: List[str],
        gold_answers: List[Any],
        task_type: str,
        question_ids: Optional[List[str]] = None
    ) -> List[JudgmentResult]:
        """
        Judge multiple predictions in batch

        Args:
            predictions: List of model predictions
            gold_answers: List of gold answers
            task_type: Task type
            question_ids: Optional question IDs

        Returns:
            List of JudgmentResults
        """
        if question_ids is None:
            question_ids = [None] * len(predictions)

        results = []
        for i, (pred, gold, qid) in enumerate(zip(predictions, gold_answers, question_ids)):
            try:
                result = self.judge_single(pred, gold, task_type, qid)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch judgment failed for item {i}: {e}")
                results.append(JudgmentResult(
                    correct=False,
                    score=0.0,
                    details={"error": str(e)},
                    explanation=f"Batch evaluation error: {e}"
                ))

        return results

    def compute_dataset_metrics(
        self,
        judgment_results: List[JudgmentResult],
        task_type: str
    ) -> Dict[str, float]:
        """
        Compute overall metrics for a dataset

        Args:
            judgment_results: List of individual judgments
            task_type: Task type

        Returns:
            Dictionary of aggregated metrics
        """
        if not judgment_results:
            return {"accuracy": 0.0, "count": 0}

        # Basic metrics
        correct_count = sum(1 for r in judgment_results if r.correct)
        accuracy = correct_count / len(judgment_results)
        avg_score = sum(r.score for r in judgment_results) / len(judgment_results)

        metrics = {
            "accuracy": accuracy,
            "average_score": avg_score,
            "correct_count": correct_count,
            "total_count": len(judgment_results)
        }

        # Task-specific metrics
        if task_type == "multihop":
            # Aggregate multihop-specific metrics
            answer_ems = []
            answer_f1s = []
            support_ems = []
            support_f1s = []
            joint_ems = []
            joint_f1s = []

            for result in judgment_results:
                details = result.details
                if "answer_em" in details:
                    answer_ems.append(details["answer_em"])
                    answer_f1s.append(details["answer_f1"])
                    support_ems.append(details["support_em"])
                    support_f1s.append(details["support_f1"])
                    joint_ems.append(details["joint_em"])
                    joint_f1s.append(details["joint_f1"])

            if answer_ems:
                metrics.update({
                    "answer_em": sum(answer_ems) / len(answer_ems),
                    "answer_f1": sum(answer_f1s) / len(answer_f1s),
                    "support_em": sum(support_ems) / len(support_ems),
                    "support_f1": sum(support_f1s) / len(support_f1s),
                    "joint_em": sum(joint_ems) / len(joint_ems),
                    "joint_f1": sum(joint_f1s) / len(joint_f1s)
                })

        elif task_type in ["qa", "open_qa"]:
            # Aggregate QA metrics
            ems = []
            f1s = []

            for result in judgment_results:
                details = result.details
                if "exact_match" in details:
                    ems.append(details["exact_match"])
                    f1s.append(details["f1_score"])

            if ems:
                metrics.update({
                    "exact_match": sum(ems) / len(ems),
                    "f1": sum(f1s) / len(f1s)
                })

        return metrics

    def analyze_errors(
        self,
        judgment_results: List[JudgmentResult],
        predictions: List[str],
        gold_answers: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze common error patterns

        Args:
            judgment_results: List of judgments
            predictions: Original predictions
            gold_answers: Gold answers

        Returns:
            Error analysis results
        """
        error_analysis = {
            "total_errors": 0,
            "error_types": {},
            "common_mistakes": [],
            "examples": []
        }

        for i, result in enumerate(judgment_results):
            if not result.correct:
                error_analysis["total_errors"] += 1

                # Categorize error type
                if "error" in result.details:
                    error_type = "evaluation_error"
                elif result.details.get("method") == "math_verify":
                    error_type = "math_computation"
                elif result.details.get("method") == "string_matching":
                    error_type = "format_mismatch"
                elif result.details.get("method") == "qa_metrics":
                    error_type = "qa_understanding"
                elif result.details.get("method") == "multihop_metrics":
                    error_type = "multihop_reasoning"
                else:
                    error_type = "unknown"

                error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1

                # Store example (first few)
                if len(error_analysis["examples"]) < 5:
                    error_analysis["examples"].append({
                        "prediction": predictions[i] if i < len(predictions) else "N/A",
                        "gold": str(gold_answers[i]) if i < len(gold_answers) else "N/A",
                        "explanation": result.explanation,
                        "error_type": error_type
                    })

        return error_analysis

    def save_detailed_results(
        self,
        judgment_results: List[JudgmentResult],
        predictions: List[str],
        gold_answers: List[Any],
        output_path: str
    ):
        """Save detailed evaluation results to file"""
        detailed_results = []

        for i, (result, pred, gold) in enumerate(zip(judgment_results, predictions, gold_answers)):
            detailed_results.append({
                "question_id": i,
                "prediction": pred,
                "gold_answer": str(gold),
                "correct": result.correct,
                "score": result.score,
                "explanation": result.explanation,
                "details": result.details
            })

        with open(output_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        logger.info(f"Detailed results saved to {output_path}")

    def get_judge_info(self) -> Dict[str, Any]:
        """Get information about the judge"""
        return {
            "math_verify_available": self.math_verify_available,
            "supported_task_types": ["math", "qa", "open_qa", "multihop"],
            "config": {
                "use_math_verify": self.config.use_math_verify,
                "compute_detailed_metrics": self.config.compute_detailed_metrics
            }
        }