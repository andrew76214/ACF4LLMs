"""Perplexity evaluator for measuring language model quality.

Perplexity is a standard metric for evaluating language models that measures
how well the model predicts the next token. Lower perplexity indicates
better language modeling capability.

This evaluator supports:
- Multiple datasets (WikiText, C4, etc.)
- Sliding window evaluation for long sequences
- Various perplexity variants (standard, word-level, bits-per-byte)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from src.evaluation.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Maximum perplexity value to return on failure (avoids numerical issues with infinity)
MAX_PERPLEXITY = 1e6


class PerplexityEvaluator(BaseEvaluator):
    """Evaluator for model perplexity on text datasets.

    Perplexity measures how well a probability model predicts a sample.
    It is defined as: PPL = exp(average negative log-likelihood)

    Lower perplexity indicates better model performance.
    """

    # Supported datasets with their HuggingFace identifiers
    SUPPORTED_DATASETS = {
        "wikitext": "wikitext-2-raw-v1",
        "wikitext-103": "wikitext-103-raw-v1",
        "c4": "c4",
        "ptb": "ptb_text_only",
        "lambada": "lambada",
    }

    def __init__(
        self,
        device: str = "cuda",
        stride: int = 512,
        max_length: Optional[int] = None,
    ):
        """Initialize the perplexity evaluator.

        Args:
            device: Device to run evaluation on ('cuda' or 'cpu')
            stride: Stride for sliding window evaluation
            max_length: Maximum sequence length (default: model's max length)
        """
        super().__init__(device)
        self.benchmark_name = "perplexity"
        self.stride = stride
        self.max_length = max_length

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int = 1,
        dataset: str = "wikitext",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> float:
        """Evaluate perplexity on a dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            batch_size: Batch size (currently only 1 supported for perplexity)
            dataset: Dataset name ('wikitext', 'c4', 'ptb', 'lambada')
            split: Dataset split ('test', 'validation')
            max_samples: Maximum number of samples to use

        Returns:
            Perplexity score (lower is better)
        """
        # Load dataset
        text = self._load_dataset(dataset, split, max_samples)
        if not text:
            logger.error(f"Failed to load dataset {dataset}")
            return MAX_PERPLEXITY

        # Calculate perplexity
        result = self.calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=text,
        )

        return result["perplexity"]

    def evaluate_full(
        self,
        model: Any,
        tokenizer: Any,
        dataset: str = "wikitext",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Comprehensive perplexity evaluation with multiple metrics.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            dataset: Dataset name
            split: Dataset split
            max_samples: Maximum number of samples

        Returns:
            Dictionary with multiple perplexity metrics:
            - perplexity: Standard perplexity
            - bits_per_byte: Bits per byte (cross-entropy in base 2)
            - word_perplexity: Per-word perplexity
            - avg_nll: Average negative log-likelihood
            - total_tokens: Total tokens evaluated
        """
        # Load dataset
        text = self._load_dataset(dataset, split, max_samples)
        if not text:
            logger.error(f"Failed to load dataset {dataset}")
            return {
                "perplexity": MAX_PERPLEXITY,
                "bits_per_byte": MAX_PERPLEXITY,
                "word_perplexity": MAX_PERPLEXITY,
                "avg_nll": MAX_PERPLEXITY,
                "total_tokens": 0,
                "dataset": dataset,
                "split": split,
                "error": f"Failed to load dataset {dataset}",
            }

        # Calculate all metrics
        result = self.calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=text,
        )

        # Calculate word-level perplexity
        words = text.split()
        num_words = len(words)
        if num_words > 0:
            word_perplexity = math.exp(result["avg_nll"] * result["total_tokens"] / num_words)
        else:
            word_perplexity = MAX_PERPLEXITY

        result["word_perplexity"] = word_perplexity
        result["num_words"] = num_words
        result["dataset"] = dataset
        result["split"] = split

        return result

    def calculate_perplexity(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
    ) -> Dict[str, float]:
        """Calculate perplexity for a given text using sliding window.

        This method uses a sliding window approach to handle long texts
        that exceed the model's maximum context length.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            text: Text to evaluate perplexity on

        Returns:
            Dictionary with:
            - perplexity: Standard perplexity
            - bits_per_byte: Bits per byte
            - avg_nll: Average negative log-likelihood
            - total_tokens: Total tokens processed
        """
        # Tokenize the full text
        encodings = tokenizer(text, return_tensors="pt")

        # Get max length from model or use default
        max_length = self.max_length or getattr(model.config, "max_position_embeddings", 2048)
        max_length = min(max_length, 4096)  # Cap at 4096 to avoid OOM

        seq_len = encodings.input_ids.size(1)
        logger.info(f"Evaluating perplexity on {seq_len} tokens with max_length={max_length}")

        nlls = []
        prev_end_loc = 0

        # Use tqdm for progress tracking
        pbar = tqdm(range(0, seq_len, self.stride), desc="Calculating perplexity")

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Number of tokens to predict

            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            if self.device == "cuda":
                input_ids = input_ids.cuda()

            # Create target labels
            target_ids = input_ids.clone()
            # Mask tokens we've already seen (before trg_len)
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # Loss is averaged over tokens, so we need to multiply back
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

            # Update progress bar with current PPL
            current_ppl = torch.exp(torch.stack(nlls).sum() / (end_loc - 0))
            pbar.set_postfix({"ppl": f"{current_ppl.item():.2f}"})

        # Calculate final metrics
        total_nll = torch.stack(nlls).sum()
        total_tokens = prev_end_loc

        avg_nll = (total_nll / total_tokens).item()
        perplexity = math.exp(avg_nll)

        # Bits per byte (assuming average of ~4 bytes per token)
        bits_per_byte = avg_nll / math.log(2) / 4

        return {
            "perplexity": perplexity,
            "bits_per_byte": bits_per_byte,
            "avg_nll": avg_nll,
            "total_tokens": total_tokens,
        }

    def _load_dataset(
        self,
        dataset: str,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> str:
        """Load a dataset and concatenate text.

        Args:
            dataset: Dataset name
            split: Dataset split
            max_samples: Maximum samples to load

        Returns:
            Concatenated text from the dataset
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            return ""

        # Map dataset name to HuggingFace identifier
        hf_dataset = self.SUPPORTED_DATASETS.get(dataset, dataset)

        try:
            if dataset in ["wikitext", "wikitext-103"]:
                # WikiText has a specific loading pattern
                ds = load_dataset("wikitext", hf_dataset, split=split)
                text_field = "text"
            elif dataset == "c4":
                # C4 is large, use streaming
                ds = load_dataset("c4", "en", split=split, streaming=True)
                text_field = "text"
                # Take only max_samples
                if max_samples:
                    texts = []
                    for i, example in enumerate(ds):
                        if i >= max_samples:
                            break
                        texts.append(example[text_field])
                    return "\n\n".join(texts)
            elif dataset == "ptb":
                ds = load_dataset("ptb_text_only", split=split)
                text_field = "sentence"
            elif dataset == "lambada":
                ds = load_dataset("lambada", split=split)
                text_field = "text"
            else:
                # Try loading as generic dataset
                ds = load_dataset(dataset, split=split)
                # Try common text field names
                if "text" in ds.column_names:
                    text_field = "text"
                elif "content" in ds.column_names:
                    text_field = "content"
                elif "sentence" in ds.column_names:
                    text_field = "sentence"
                else:
                    text_field = ds.column_names[0]

            # Limit samples if specified
            if max_samples and max_samples < len(ds):
                ds = ds.select(range(max_samples))

            # Concatenate all text
            texts = [example[text_field] for example in ds if example[text_field].strip()]
            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset}: {e}")
            return ""

    def evaluate_proxy(
        self,
        model: Any,
        tokenizer: Any,
        num_samples: int = 100,
        batch_size: int = 1,
    ) -> float:
        """Run proxy perplexity evaluation with limited samples.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_samples: Number of samples to evaluate
            batch_size: Batch size (unused, kept for API compatibility)

        Returns:
            Proxy perplexity score
        """
        return self.evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            split="test",
            max_samples=num_samples,
        )


def calculate_perplexity_on_text(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str = "cuda",
    stride: int = 512,
    max_length: int = 2048,
) -> Dict[str, float]:
    """Convenience function to calculate perplexity on arbitrary text.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        text: Text to evaluate
        device: Device to use
        stride: Sliding window stride
        max_length: Maximum sequence length

    Returns:
        Dictionary with perplexity metrics
    """
    evaluator = PerplexityEvaluator(
        device=device,
        stride=stride,
        max_length=max_length,
    )
    return evaluator.calculate_perplexity(model, tokenizer, text)
