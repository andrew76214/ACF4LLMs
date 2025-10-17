"""
Metrics and evaluation utilities for A2A Pipeline
"""

import numpy as np
from typing import List, Dict, Any, Union, Tuple
from collections import Counter
from .text_processing import normalize_text

def compute_accuracy(predictions: List[str], gold_answers: List[str]) -> float:
    """
    Compute accuracy between predictions and gold answers

    Args:
        predictions: Model predictions
        gold_answers: Ground truth answers

    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")

    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        if normalize_text(str(pred)) == normalize_text(str(gold)):
            correct += 1

    return correct / len(predictions)

def compute_f1(prediction: str, gold_answer: str) -> float:
    """
    Compute F1 score between a prediction and gold answer

    Args:
        prediction: Model prediction
        gold_answer: Ground truth answer

    Returns:
        F1 score (0-1)
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Count common tokens
    common_tokens = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_batch_f1(predictions: List[str], gold_answers: List[str]) -> Tuple[float, List[float]]:
    """
    Compute F1 scores for a batch of predictions

    Args:
        predictions: List of predictions
        gold_answers: List of ground truth answers

    Returns:
        Tuple of (average F1, list of individual F1 scores)
    """
    f1_scores = []
    for pred, gold in zip(predictions, gold_answers):
        f1 = compute_f1(pred, gold)
        f1_scores.append(f1)

    return np.mean(f1_scores), f1_scores

def compute_exact_match(predictions: List[str], gold_answers: List[str]) -> Tuple[float, List[int]]:
    """
    Compute exact match scores

    Args:
        predictions: List of predictions
        gold_answers: List of ground truth answers

    Returns:
        Tuple of (EM score, list of exact matches)
    """
    exact_matches = []
    for pred, gold in zip(predictions, gold_answers):
        em = 1 if normalize_text(pred) == normalize_text(gold) else 0
        exact_matches.append(em)

    em_score = np.mean(exact_matches)
    return em_score, exact_matches

def majority_vote(predictions: List[str]) -> str:
    """
    Get majority vote from multiple predictions (used in self-consistency)

    Args:
        predictions: List of predictions

    Returns:
        Most common prediction
    """
    if not predictions:
        return ""

    # Normalize predictions for voting
    normalized_preds = [normalize_text(pred) for pred in predictions]

    # Count votes
    vote_counts = Counter(normalized_preds)

    # Return most common (in case of tie, returns first encountered)
    return vote_counts.most_common(1)[0][0]

def compute_self_consistency_metrics(
    multiple_predictions: List[List[str]],
    gold_answers: List[str]
) -> Dict[str, float]:
    """
    Compute metrics for self-consistency decoding

    Args:
        multiple_predictions: List of prediction lists (one list per question)
        gold_answers: Ground truth answers

    Returns:
        Dictionary of metrics
    """
    majority_preds = []
    agreement_scores = []

    for preds in multiple_predictions:
        # Get majority vote
        majority_pred = majority_vote(preds)
        majority_preds.append(majority_pred)

        # Compute agreement (what fraction agree with majority)
        majority_count = Counter([normalize_text(p) for p in preds]).most_common(1)[0][1]
        agreement = majority_count / len(preds)
        agreement_scores.append(agreement)

    # Compute final metrics
    accuracy = compute_accuracy(majority_preds, gold_answers)
    avg_agreement = np.mean(agreement_scores)

    return {
        "majority_vote_accuracy": accuracy,
        "average_agreement": avg_agreement,
        "num_questions": len(multiple_predictions)
    }

def compute_supporting_facts_metrics(
    pred_facts: List[List[List]],
    gold_facts: List[List[List]]
) -> Dict[str, float]:
    """
    Compute metrics for supporting facts (HotpotQA style)

    Args:
        pred_facts: Predicted supporting facts
        gold_facts: Ground truth supporting facts

    Returns:
        Dictionary of metrics
    """
    em_scores = []
    f1_scores = []

    for pred, gold in zip(pred_facts, gold_facts):
        # Convert to sets for comparison
        pred_set = set(tuple(fact) for fact in pred)
        gold_set = set(tuple(fact) for fact in gold)

        # Exact match
        em = 1 if pred_set == gold_set else 0
        em_scores.append(em)

        # F1 score
        if not pred_set and not gold_set:
            f1 = 1.0
        elif not pred_set or not gold_set:
            f1 = 0.0
        else:
            common = pred_set & gold_set
            precision = len(common) / len(pred_set)
            recall = len(common) / len(gold_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)

    return {
        "supporting_facts_em": np.mean(em_scores),
        "supporting_facts_f1": np.mean(f1_scores)
    }

def compute_joint_metrics(
    answer_em: List[int],
    answer_f1: List[float],
    support_em: List[int],
    support_f1: List[float]
) -> Dict[str, float]:
    """
    Compute joint metrics (both answer and supporting facts correct)

    Args:
        answer_em: Answer exact match scores
        answer_f1: Answer F1 scores
        support_em: Supporting facts exact match scores
        support_f1: Supporting facts F1 scores

    Returns:
        Dictionary of joint metrics
    """
    joint_em = [a * s for a, s in zip(answer_em, support_em)]
    joint_f1 = [(a + s) / 2 for a, s in zip(answer_f1, support_f1)]  # Average of both

    return {
        "joint_em": np.mean(joint_em),
        "joint_f1": np.mean(joint_f1)
    }

def compute_variance_metrics(scores_across_runs: List[List[float]]) -> Dict[str, float]:
    """
    Compute variance metrics across multiple runs

    Args:
        scores_across_runs: List of score lists from different runs

    Returns:
        Variance statistics
    """
    # Convert to numpy array for easier computation
    scores_array = np.array(scores_across_runs)

    # Compute statistics across runs for each question
    question_means = np.mean(scores_array, axis=0)
    question_stds = np.std(scores_array, axis=0)

    # Compute overall statistics
    overall_mean = np.mean(question_means)
    overall_std = np.mean(question_stds)
    run_means = np.mean(scores_array, axis=1)
    run_std = np.std(run_means)

    return {
        "mean_accuracy": overall_mean,
        "mean_question_std": overall_std,
        "run_std": run_std,
        "min_run_accuracy": np.min(run_means),
        "max_run_accuracy": np.max(run_means)
    }

def format_metrics_report(metrics: Dict[str, Any], dataset_name: str) -> str:
    """
    Format metrics into a readable report

    Args:
        metrics: Dictionary of computed metrics
        dataset_name: Name of the dataset

    Returns:
        Formatted report string
    """
    report_lines = [
        f"=== {dataset_name.upper()} RESULTS ===",
        ""
    ]

    # Core metrics
    if "accuracy" in metrics:
        report_lines.append(f"Accuracy: {metrics['accuracy']:.3f}")

    if "em" in metrics:
        report_lines.append(f"Exact Match: {metrics['em']:.3f}")

    if "f1" in metrics:
        report_lines.append(f"F1 Score: {metrics['f1']:.3f}")

    # Multi-hop specific metrics
    if "answer_em" in metrics:
        report_lines.extend([
            "",
            "Multi-hop QA Metrics:",
            f"  Answer EM: {metrics['answer_em']:.3f}",
            f"  Answer F1: {metrics['answer_f1']:.3f}",
            f"  Support EM: {metrics['supporting_facts_em']:.3f}",
            f"  Support F1: {metrics['supporting_facts_f1']:.3f}",
            f"  Joint EM: {metrics['joint_em']:.3f}",
            f"  Joint F1: {metrics['joint_f1']:.3f}"
        ])

    # Self-consistency metrics
    if "majority_vote_accuracy" in metrics:
        report_lines.extend([
            "",
            "Self-Consistency Metrics:",
            f"  Majority Vote Accuracy: {metrics['majority_vote_accuracy']:.3f}",
            f"  Average Agreement: {metrics['average_agreement']:.3f}"
        ])

    # Variance metrics
    if "run_std" in metrics:
        report_lines.extend([
            "",
            "Variance Analysis:",
            f"  Mean Accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['run_std']:.3f}",
            f"  Min Run: {metrics['min_run_accuracy']:.3f}",
            f"  Max Run: {metrics['max_run_accuracy']:.3f}"
        ])

    report_lines.append("=" * len(f"=== {dataset_name.upper()} RESULTS ==="))

    return "\n".join(report_lines)