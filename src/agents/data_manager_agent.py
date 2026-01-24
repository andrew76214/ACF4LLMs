"""Data Manager Agent for dataset handling, calibration data, and preprocessing."""

import json
import os
import random
import hashlib
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def load_calibration_dataset(
    dataset_name: str,
    num_samples: int = 512,
    sequence_length: int = 2048,
    cache_dir: Optional[str] = None,
    split: str = "train",
) -> Dict[str, Any]:
    """Load calibration dataset for quantization/compression.

    Args:
        dataset_name: Name of dataset (wikitext, c4, pile, etc.)
        num_samples: Number of calibration samples
        sequence_length: Maximum sequence length
        cache_dir: Directory to cache datasets
        split: Dataset split to use

    Returns:
        Dictionary with dataset information and samples
    """
    print(f"[Data] Loading {num_samples} samples from {dataset_name}")

    if not cache_dir:
        cache_dir = "data/cache/datasets"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "name": dataset_name,
        "num_samples": num_samples,
        "sequence_length": sequence_length,
        "split": split,
        "cache_dir": cache_dir,
    }

    try:
        from datasets import load_dataset

        # Load dataset
        if dataset_name.lower() == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
            text_column = "text"
        elif dataset_name.lower() == "c4":
            dataset = load_dataset("c4", "en", split=split, streaming=True, cache_dir=cache_dir)
            text_column = "text"
        elif dataset_name.lower() == "pile":
            dataset = load_dataset("EleutherAI/pile", split=split, streaming=True, cache_dir=cache_dir)
            text_column = "text"
        elif dataset_name.lower() == "openwebtext":
            dataset = load_dataset("openwebtext", split=split, cache_dir=cache_dir)
            text_column = "text"
        else:
            # Default to wikitext
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
            text_column = "text"

        # Sample texts
        samples = []
        if hasattr(dataset, "__iter__"):
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                text = item[text_column]
                if text and len(text) > 10:  # Filter empty/short texts
                    samples.append(text[:sequence_length])
        else:
            # For non-iterable datasets
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            samples = [dataset[i][text_column][:sequence_length] for i in indices]

        dataset_info["samples"] = samples
        dataset_info["actual_num_samples"] = len(samples)

    except ImportError:
        logger.warning("datasets library not available, using mock data")
        samples = _generate_mock_calibration_data(num_samples, sequence_length)
        dataset_info["samples"] = samples
        dataset_info["actual_num_samples"] = len(samples)
        dataset_info["mock"] = True

    # Cache samples for reuse
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{num_samples}_{sequence_length}.json")
    with open(cache_file, "w") as f:
        json.dump({"samples": samples[:100]}, f)  # Cache subset

    print(f"[Data] Loaded {len(samples)} calibration samples")
    return dataset_info


@tool
def load_benchmark_dataset(
    benchmark: str,
    split: str = "test",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Load benchmark dataset for evaluation.

    Args:
        benchmark: Benchmark name (gsm8k, commonsenseqa, etc.)
        split: Dataset split
        num_samples: Limit number of samples (None = all)

    Returns:
        Dictionary with benchmark data
    """
    print(f"[Benchmark] Loading {benchmark} dataset")

    benchmark_data = {
        "benchmark": benchmark,
        "split": split,
        "num_samples": num_samples,
    }

    try:
        from datasets import load_dataset

        # Load benchmark-specific dataset
        if benchmark.lower() == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split)
            benchmark_data["task_type"] = "math_reasoning"
            benchmark_data["metric"] = "exact_match"
        elif benchmark.lower() in ["commonsenseqa", "commonsense_qa"]:
            dataset = load_dataset("commonsense_qa", split=split)
            benchmark_data["task_type"] = "multiple_choice"
            benchmark_data["metric"] = "accuracy"
        elif benchmark.lower() in ["truthfulqa", "truthful_qa"]:
            dataset = load_dataset("truthful_qa", "generation", split="validation")
            benchmark_data["task_type"] = "generation"
            benchmark_data["metric"] = "truthfulness"
        elif benchmark.lower() == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
            benchmark_data["task_type"] = "code_generation"
            benchmark_data["metric"] = "pass@1"
        elif benchmark.lower() in ["bigbench", "bigbench_hard"]:
            dataset = load_dataset("lukaemon/bbh", split="test")
            benchmark_data["task_type"] = "reasoning"
            benchmark_data["metric"] = "accuracy"
        else:
            # Default mock dataset
            dataset = None

        if dataset:
            if num_samples:
                samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
            else:
                samples = list(dataset)
            benchmark_data["samples"] = samples
            benchmark_data["total_samples"] = len(samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    except Exception as e:
        logger.warning(f"Failed to load benchmark: {e}, using mock data")
        samples = _generate_mock_benchmark_data(benchmark, num_samples or 100)
        benchmark_data["samples"] = samples
        benchmark_data["total_samples"] = len(samples)
        benchmark_data["mock"] = True

    print(f"[Benchmark] Loaded {benchmark_data['total_samples']} samples")
    return benchmark_data


@tool
def preprocess_text_data(
    texts: List[str],
    tokenizer_name: str,
    max_length: int = 2048,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, Any]:
    """Preprocess text data for model input.

    Args:
        texts: List of text strings
        tokenizer_name: Name or path of tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Dictionary with tokenized data
    """
    print(f"[Preprocess] Processing {len(texts)} texts with {tokenizer_name}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize texts
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "num_samples": len(texts),
            "max_length": max_length,
            "tokenizer": tokenizer_name,
            "vocab_size": tokenizer.vocab_size,
        }

    except ImportError:
        logger.warning("transformers not available, using mock tokenization")
        # Mock tokenization
        mock_ids = [[random.randint(1, 50000) for _ in range(max_length)] for _ in texts]
        mock_mask = [[1] * max_length for _ in texts]

        return {
            "input_ids": mock_ids,
            "attention_mask": mock_mask,
            "num_samples": len(texts),
            "max_length": max_length,
            "tokenizer": tokenizer_name,
            "vocab_size": 50000,
            "mock": True,
        }


@tool
def create_data_splits(
    dataset_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Split dataset into train/validation/test sets.

    Args:
        dataset_path: Path to dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed

    Returns:
        Dictionary with split information
    """
    print(f"[Split] Creating data splits: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")

    random.seed(seed)

    # This would load actual data in production
    # For now, work with indices
    total_samples = 10000  # Mock size

    indices = list(range(total_samples))
    random.shuffle(indices)

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    splits = {
        "train": {
            "indices": train_indices,
            "size": len(train_indices),
            "ratio": train_ratio,
        },
        "validation": {
            "indices": val_indices,
            "size": len(val_indices),
            "ratio": val_ratio,
        },
        "test": {
            "indices": test_indices,
            "size": len(test_indices),
            "ratio": test_ratio,
        },
        "total_samples": total_samples,
        "seed": seed,
    }

    print(f"[Split] Train: {splits['train']['size']}, Val: {splits['validation']['size']}, Test: {splits['test']['size']}")

    return splits


@tool
def cache_dataset(
    dataset_name: str,
    data: Union[List, Dict],
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Cache dataset to disk for faster loading.

    Args:
        dataset_name: Name for cached dataset
        data: Data to cache
        cache_dir: Cache directory

    Returns:
        Dictionary with cache information
    """
    print(f"[Cache] Caching dataset: {dataset_name}")

    if not cache_dir:
        cache_dir = "data/cache/datasets"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Generate cache key
    data_str = json.dumps(data if isinstance(data, dict) else {"data": data[:10]}, sort_keys=True)
    cache_key = hashlib.md5(data_str.encode()).hexdigest()[:8]

    cache_file = os.path.join(cache_dir, f"{dataset_name}_{cache_key}.json")

    # Save to cache
    cache_data = {
        "dataset_name": dataset_name,
        "cache_key": cache_key,
        "timestamp": datetime.now().isoformat(),
        "data": data if isinstance(data, dict) else {"samples": data},
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    file_size = os.path.getsize(cache_file) / (1024**2)  # MB

    print(f"[Cache] Saved to {cache_file} ({file_size:.1f} MB)")

    return {
        "cache_file": cache_file,
        "cache_key": cache_key,
        "file_size_mb": file_size,
        "dataset_name": dataset_name,
    }


@tool
def load_cached_dataset(
    dataset_name: str,
    cache_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load dataset from cache.

    Args:
        dataset_name: Dataset name
        cache_key: Specific cache key
        cache_dir: Cache directory

    Returns:
        Dictionary with cached data
    """
    print(f"[Cache] Loading cached dataset: {dataset_name}")

    if not cache_dir:
        cache_dir = "data/cache/datasets"

    if cache_key:
        cache_file = os.path.join(cache_dir, f"{dataset_name}_{cache_key}.json")
    else:
        # Find most recent cache
        pattern = f"{dataset_name}_*.json"
        import glob

        files = glob.glob(os.path.join(cache_dir, pattern))
        if not files:
            return {"error": f"No cached files found for {dataset_name}"}
        cache_file = max(files, key=os.path.getctime)

    if not os.path.exists(cache_file):
        return {"error": f"Cache file not found: {cache_file}"}

    with open(cache_file, "r") as f:
        cache_data = json.load(f)

    print(f"[Cache] Loaded from {cache_file}")

    return {
        "cache_file": cache_file,
        "data": cache_data.get("data", cache_data),
        "cache_key": cache_data.get("cache_key"),
        "timestamp": cache_data.get("timestamp"),
    }


@tool
def estimate_dataset_statistics(
    dataset_name: str,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """Estimate statistics for a dataset.

    Args:
        dataset_name: Dataset name
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with dataset statistics
    """
    print(f"[Stats] Analyzing {dataset_name} dataset")

    # Load sample data
    data = load_calibration_dataset(dataset_name, num_samples=num_samples)

    if "samples" not in data:
        return {"error": "Failed to load dataset samples"}

    samples = data["samples"]

    # Calculate statistics
    lengths = [len(s) for s in samples]

    stats = {
        "dataset_name": dataset_name,
        "num_samples_analyzed": len(samples),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "median_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "total_characters": sum(lengths),
        "unique_samples": len(set(samples)),
        "duplicate_ratio": 1 - (len(set(samples)) / len(samples)) if samples else 0,
    }

    # Estimate tokens (rough approximation)
    stats["estimated_tokens"] = stats["total_characters"] // 4

    print(f"[Stats] Avg length: {stats['avg_length']:.0f} chars")
    print(f"[Stats] Estimated tokens: {stats['estimated_tokens']:,}")

    return stats


def _generate_mock_calibration_data(num_samples: int, sequence_length: int) -> List[str]:
    """Generate mock calibration data."""
    templates = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning models can be compressed using various techniques. " * 8,
        "Natural language processing has made significant advances recently. " * 8,
        "Quantization reduces model size while maintaining performance. " * 8,
        "Knowledge distillation transfers knowledge from teacher to student. " * 8,
    ]

    samples = []
    for i in range(num_samples):
        text = templates[i % len(templates)]
        # Add variation
        text = text + f" Sample number {i}. "
        samples.append(text[:sequence_length])

    return samples


def _generate_mock_benchmark_data(benchmark: str, num_samples: int) -> List[Dict]:
    """Generate mock benchmark data."""
    samples = []

    if "gsm8k" in benchmark.lower():
        for i in range(num_samples):
            samples.append({
                "question": f"John has {i+5} apples and gives away {i+2}. How many does he have left?",
                "answer": str((i+5) - (i+2)),
            })
    elif "commonsense" in benchmark.lower():
        for i in range(num_samples):
            samples.append({
                "question": f"What is commonly used for transportation? (Sample {i})",
                "choices": ["car", "table", "cloud", "idea"],
                "answer": "car",
            })
    elif "humaneval" in benchmark.lower():
        for i in range(num_samples):
            samples.append({
                "prompt": f"def add_{i}(a, b):\n    '''Add two numbers'''\n",
                "canonical_solution": f"    return a + b",
                "test": f"assert add_{i}(1, 2) == 3",
            })
    else:
        for i in range(num_samples):
            samples.append({
                "input": f"Sample input {i}",
                "output": f"Sample output {i}",
            })

    return samples


def get_data_manager_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the data manager subagent configuration.

    Args:
        spec: Model specification

    Returns:
        Subagent configuration dictionary
    """
    model_name = spec.get("model_name", "unknown")
    dataset = spec.get("dataset", "wikitext")

    prompt = f"""You are a Data Manager Agent responsible for dataset handling and preprocessing.

Model: {model_name}
Primary Dataset: {dataset}

Your responsibilities:
1. Load and manage calibration datasets for compression
2. Load benchmark datasets for evaluation
3. Preprocess and tokenize text data
4. Create train/validation/test splits
5. Cache datasets for efficient reuse
6. Analyze dataset statistics

Available tools:
- load_calibration_dataset: Load calibration data for quantization
- load_benchmark_dataset: Load evaluation benchmarks
- preprocess_text_data: Tokenize and prepare text
- create_data_splits: Split data into train/val/test
- cache_dataset: Save dataset to cache
- load_cached_dataset: Load from cache
- estimate_dataset_statistics: Analyze dataset properties

Data Strategy:
1. Use appropriate datasets for calibration:
   - Wikitext: General purpose
   - C4: Web text
   - Pile: Diverse sources
2. Cache frequently used datasets
3. Use correct number of calibration samples:
   - Quantization: 512-1024 samples
   - Pruning: 100-500 samples
   - Distillation: 5000+ samples
4. Match sequence length to model's training

Guidelines:
- Always check cache before downloading
- Use streaming for large datasets
- Preprocess in batches for memory efficiency
- Ensure data quality (remove empty/duplicate samples)

When you receive a data request:
1. Check if cached version exists
2. Load appropriate dataset
3. Preprocess if needed
4. Cache for future use
5. Return data statistics
"""

    return {
        "name": "data_manager",
        "description": "Manages datasets, calibration data, and preprocessing",
        "prompt": prompt,
        "tools": [
            load_calibration_dataset,
            load_benchmark_dataset,
            preprocess_text_data,
            create_data_splits,
            cache_dataset,
            load_cached_dataset,
            estimate_dataset_statistics,
        ],
        "model": "anthropic:claude-3-haiku-20240307",
    }


__all__ = ["get_data_manager_subagent", "load_calibration_dataset", "load_benchmark_dataset"]