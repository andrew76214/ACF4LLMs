"""lm-eval harness integration for standardized benchmarking.

This module provides a wrapper around EleutherAI's lm-evaluation-harness,
enabling standardized evaluation on popular LLM benchmarks.

References:
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LMEvalEvaluator:
    """Wrapper for EleutherAI's lm-evaluation-harness.

    This evaluator uses the standardized lm-eval framework to evaluate models
    on various benchmarks. It wraps pre-loaded HuggingFace models using HFLM
    and runs evaluation through the simple_evaluate API.

    Example:
        >>> evaluator = LMEvalEvaluator(device="cuda")
        >>> scores = evaluator.evaluate(model, tokenizer, ["gsm8k", "mmlu"], limit=100)
        >>> print(scores)
        {'gsm8k': 0.45, 'mmlu': 0.62}
    """

    # Map internal benchmark names to lm-eval task names
    TASK_MAPPING = {
        # Existing benchmarks
        "gsm8k": "gsm8k",
        "commonsenseqa": "commonsense_qa",
        "truthfulqa": "truthfulqa_mc2",
        "bigbench_hard": "bbh",
        # New benchmarks
        "mmlu": "mmlu",
        "hellaswag": "hellaswag",
        "arc_easy": "arc_easy",
        "arc_challenge": "arc_challenge",
        "winogrande": "winogrande",
    }

    # Primary metric for each task (used to extract scores)
    METRIC_MAPPING = {
        "gsm8k": "exact_match,strict-match",
        "commonsense_qa": "acc",
        "truthfulqa_mc2": "acc",
        "bbh": "acc_norm",
        "mmlu": "acc",
        "hellaswag": "acc_norm",
        "arc_easy": "acc_norm",
        "arc_challenge": "acc_norm",
        "winogrande": "acc",
    }

    def __init__(self, device: str = "cuda"):
        """Initialize the lm-eval evaluator.

        Args:
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.device = device
        self._lm_eval_available = None

    def _check_lm_eval_available(self) -> bool:
        """Check if lm-eval is available."""
        if self._lm_eval_available is None:
            try:
                import lm_eval
                from lm_eval.models.huggingface import HFLM

                self._lm_eval_available = True
                logger.info("lm-eval harness is available")
            except ImportError as e:
                self._lm_eval_available = False
                logger.warning(f"lm-eval harness not available: {e}")
        return self._lm_eval_available

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        tasks: List[str],
        limit: Optional[int] = None,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Evaluate model on specified tasks using lm-eval.

        Args:
            model: Pre-loaded HuggingFace model (PreTrainedModel)
            tokenizer: Pre-loaded tokenizer (PreTrainedTokenizer)
            tasks: List of benchmark names (internal names, e.g., ['gsm8k', 'mmlu'])
            limit: Max samples per task (for proxy/fast evaluation). None for full eval.
            batch_size: Batch size for evaluation

        Returns:
            Dict mapping task name to score (0.0 to 1.0)

        Raises:
            RuntimeError: If lm-eval is not available
        """
        if not self._check_lm_eval_available():
            logger.error("lm-eval not available, returning zero scores")
            return {task.lower(): 0.0 for task in tasks}

        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # Filter to only supported tasks
        supported_tasks = [t for t in tasks if t.lower() in self.TASK_MAPPING]
        unsupported = [t for t in tasks if t.lower() not in self.TASK_MAPPING]

        if unsupported:
            logger.warning(f"Unsupported tasks (will be skipped): {unsupported}")

        if not supported_tasks:
            logger.warning("No supported lm-eval tasks to evaluate")
            return {}

        # Map to lm-eval task names
        lm_eval_tasks = [self.TASK_MAPPING[t.lower()] for t in supported_tasks]

        logger.info(f"Running lm-eval on tasks: {lm_eval_tasks}")
        if limit:
            logger.info(f"Using limit={limit} samples per task (proxy mode)")

        try:
            # Wrap model in HFLM for lm-eval compatibility
            # HFLM accepts pretrained as PreTrainedModel and tokenizer as PreTrainedTokenizer
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )

            # Run evaluation using simple_evaluate
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=lm_eval_tasks,
                limit=limit,
                batch_size=batch_size,
            )

            # Parse and return results
            return self._parse_results(results, supported_tasks)

        except Exception as e:
            logger.error(f"lm-eval evaluation failed: {e}")
            raise RuntimeError(f"lm-eval evaluation failed: {e}") from e

    def _parse_results(
        self, results: Dict[str, Any], original_tasks: List[str]
    ) -> Dict[str, float]:
        """Parse lm-eval results into simple task->score mapping.

        Args:
            results: Raw results from lm_eval.simple_evaluate()
            original_tasks: List of original task names (internal names)

        Returns:
            Dict mapping internal task name to score
        """
        scores = {}

        if "results" not in results:
            logger.warning("No 'results' key in lm-eval output")
            return scores

        for task in original_tasks:
            task_lower = task.lower()
            lm_task = self.TASK_MAPPING.get(task_lower)

            if not lm_task:
                continue

            # lm-eval results may have the task name directly or with subtask
            task_results = results["results"].get(lm_task, {})

            if not task_results:
                # Try to find results with subtask prefix (e.g., 'mmlu' might be 'mmlu_*')
                for key in results["results"]:
                    if key.startswith(lm_task):
                        task_results = results["results"][key]
                        break

            if task_results:
                # Get the primary metric for this task
                metric_key = self.METRIC_MAPPING.get(lm_task, "acc")

                # Try to get the metric value
                score = task_results.get(metric_key)
                if score is None:
                    # Fallback to 'acc' if primary metric not found
                    score = task_results.get("acc")
                if score is None:
                    # Try acc_norm
                    score = task_results.get("acc_norm")
                if score is None:
                    # Last resort: get any numeric value
                    for v in task_results.values():
                        if isinstance(v, (int, float)) and 0 <= v <= 1:
                            score = v
                            break

                if score is not None:
                    scores[task_lower] = float(score)
                    logger.info(f"{task}: {score:.4f}")
                else:
                    logger.warning(f"Could not extract score for {task}")
            else:
                logger.warning(f"No results found for task {task} (lm-eval: {lm_task})")

        return scores

    def evaluate_proxy(
        self,
        model: Any,
        tokenizer: Any,
        tasks: List[str],
        num_samples: int = 100,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Run fast proxy evaluation with limited samples.

        This is a convenience wrapper around evaluate() with limit set.

        Args:
            model: Pre-loaded HuggingFace model
            tokenizer: Pre-loaded tokenizer
            tasks: List of benchmark names
            num_samples: Number of samples per task
            batch_size: Batch size

        Returns:
            Dict mapping task name to score
        """
        return self.evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            limit=num_samples,
            batch_size=batch_size,
        )

    @classmethod
    def get_supported_tasks(cls) -> List[str]:
        """Return list of supported benchmark names (internal names).

        Returns:
            List of task names that can be passed to evaluate()
        """
        return list(cls.TASK_MAPPING.keys())

    @classmethod
    def is_task_supported(cls, task: str) -> bool:
        """Check if a task is supported by lm-eval.

        Args:
            task: Task name to check

        Returns:
            True if task is supported
        """
        return task.lower() in cls.TASK_MAPPING
