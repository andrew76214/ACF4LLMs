"""Base evaluator class for all benchmark evaluators."""

import torch
import time
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for benchmark evaluators."""

    def __init__(self, device: str = "cuda"):
        """Initialize evaluator.

        Args:
            device: Device to run evaluation on
        """
        self.device = device
        self.benchmark_name = "base"

    @abstractmethod
    def evaluate(self, model: Any, tokenizer: Any, batch_size: int = 8) -> float:
        """Run full evaluation on the benchmark.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            batch_size: Batch size for evaluation

        Returns:
            Score (typically accuracy between 0 and 1)
        """
        pass

    def evaluate_proxy(
        self, model: Any, tokenizer: Any, num_samples: int = 100, batch_size: int = 8
    ) -> float:
        """Run proxy evaluation with limited samples.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_samples: Number of samples to evaluate
            batch_size: Batch size

        Returns:
            Proxy score
        """
        # Default implementation: run full evaluation with sample limit
        # Subclasses can override for more efficient proxy evaluation
        return self.evaluate(model, tokenizer, batch_size)

    def prepare_prompt(self, example: Dict[str, Any]) -> str:
        """Prepare prompt from example.

        Args:
            example: Dataset example

        Returns:
            Formatted prompt string
        """
        return str(example.get("question", ""))

    def extract_answer(self, output: str) -> str:
        """Extract answer from model output.

        Args:
            output: Model output string

        Returns:
            Extracted answer
        """
        # Default: return full output
        # Subclasses should override with specific extraction logic
        return output.strip()

    def compute_metric(self, predictions: list, references: list) -> float:
        """Compute evaluation metric.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            Metric score
        """
        # Default: exact match accuracy
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        return correct / len(predictions) if predictions else 0.0

    @torch.no_grad()
    def generate_batch(
        self, model: Any, tokenizer: Any, prompts: list, max_length: int = 512
    ) -> list:
        """Generate responses for a batch of prompts.

        Args:
            model: Model to use
            tokenizer: Tokenizer
            prompts: List of prompts
            max_length: Maximum generation length

        Returns:
            List of generated texts
        """
        # Tokenize inputs
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove input prompt from output
        results = []
        for prompt, text in zip(prompts, generated_texts):
            if text.startswith(prompt):
                results.append(text[len(prompt):].strip())
            else:
                results.append(text.strip())

        return results