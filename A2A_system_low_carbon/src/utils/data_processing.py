"""
Data processing utilities for A2A Pipeline
Handles dataset loading, saving results, and batch processing
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Union
import pandas as pd
from datasets import load_dataset, Dataset
from configs.config import DATA_DIR, RESULTS_DIR

def load_dataset_from_hf(dataset_name: str, split: str = "test", **kwargs) -> Dataset:
    """
    Load dataset from Hugging Face

    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
        **kwargs: Additional arguments for load_dataset

    Returns:
        Loaded dataset
    """
    try:
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")

def load_local_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load dataset from local file

    Args:
        file_path: Path to the dataset file

    Returns:
        List of data samples
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def load_gsm8k_platinum():
    """Load GSM8K Platinum dataset"""
    try:
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        return list(dataset)
    except:
        # Fallback to local file if available
        local_path = DATA_DIR / "gsm8k_platinum.jsonl"
        if local_path.exists():
            return load_local_dataset(local_path)
        raise ValueError("GSM8K Platinum dataset not found")

def load_aime_dataset(year: str = "2024"):
    """Load AIME dataset for specified year"""
    local_path = DATA_DIR / f"aime_{year}.json"
    if local_path.exists():
        return load_local_dataset(local_path)

    # If not found locally, create sample data structure
    sample_data = [
        {
            "question": f"AIME {year} Problem {i}",
            "answer": str(i % 1000).zfill(3),
            "problem_id": f"aime_{year}_{i}"
        }
        for i in range(1, 16)  # AIME typically has 15 problems
    ]
    return sample_data

def load_math_dataset(subset_size: Optional[int] = None):
    """Load MATH dataset"""
    try:
        dataset = load_dataset("hendrycks/competition_math", split="test")
        data = list(dataset)
        if subset_size:
            data = data[:subset_size]
        return data
    except:
        # Fallback
        local_path = DATA_DIR / "math_dataset.jsonl"
        if local_path.exists():
            data = load_local_dataset(local_path)
            if subset_size:
                data = data[:subset_size]
            return data
        raise ValueError("MATH dataset not found")

def load_hotpot_qa():
    """Load HotpotQA dataset"""
    try:
        dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        return list(dataset)
    except:
        local_path = DATA_DIR / "hotpotqa_dev.jsonl"
        if local_path.exists():
            return load_local_dataset(local_path)
        raise ValueError("HotpotQA dataset not found")

def load_squad_dataset(version: str = "v1"):
    """Load SQuAD dataset"""
    try:
        if version == "v1":
            dataset = load_dataset("squad", split="validation")
        else:
            dataset = load_dataset("squad_v2", split="validation")
        return list(dataset)
    except:
        local_path = DATA_DIR / f"squad_{version}_dev.jsonl"
        if local_path.exists():
            return load_local_dataset(local_path)
        raise ValueError(f"SQuAD {version} dataset not found")

def load_natural_questions():
    """Load Natural Questions dataset"""
    try:
        dataset = load_dataset("natural_questions", split="validation")
        return list(dataset)
    except:
        local_path = DATA_DIR / "nq_dev.jsonl"
        if local_path.exists():
            return load_local_dataset(local_path)
        raise ValueError("Natural Questions dataset not found")

def save_results(results: Dict[str, Any], filename: str, format: str = "json"):
    """
    Save results to file

    Args:
        results: Results dictionary
        filename: Output filename
        format: Output format (json, pickle, csv)
    """
    output_path = RESULTS_DIR / filename

    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == "csv":
        if isinstance(results, dict) and "predictions" in results:
            df = pd.DataFrame(results["predictions"])
            df.to_csv(output_path, index=False)
        else:
            df = pd.DataFrame([results])
            df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_predictions(predictions: List[Dict], dataset_name: str, timestamp: str):
    """
    Save predictions in format required by evaluation scripts

    Args:
        predictions: List of prediction dictionaries
        dataset_name: Name of the dataset
        timestamp: Timestamp for file naming
    """
    filename = f"{dataset_name}_predictions_{timestamp}.json"
    save_results({"predictions": predictions}, filename)

    # Also save in evaluation script format if needed
    if dataset_name == "hotpotqa":
        # HotpotQA format: {question_id: {answer: str, supporting_facts: list}}
        hotpot_format = {}
        for pred in predictions:
            if "id" in pred and "prediction" in pred:
                hotpot_format[pred["id"]] = pred["prediction"]

        eval_filename = f"hotpotqa_eval_format_{timestamp}.json"
        save_results(hotpot_format, eval_filename)

def batch_iterator(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Create batches from data

    Args:
        data: Input data list
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def sample_dataset(data: List[Dict], sample_size: Optional[int], seed: int = 42) -> List[Dict]:
    """
    Sample a subset of the dataset

    Args:
        data: Full dataset
        sample_size: Number of samples to take (None for full dataset)
        seed: Random seed

    Returns:
        Sampled dataset
    """
    if sample_size is None or sample_size >= len(data):
        return data

    import random
    random.seed(seed)
    return random.sample(data, sample_size)

def format_dataset_for_evaluation(predictions: List[Dict], dataset_name: str) -> Dict:
    """
    Format predictions for official evaluation scripts

    Args:
        predictions: Model predictions
        dataset_name: Name of the dataset

    Returns:
        Formatted predictions dictionary
    """
    if dataset_name == "squad":
        # SQuAD format: {question_id: answer_string}
        return {pred["id"]: pred["prediction"] for pred in predictions}

    elif dataset_name == "hotpotqa":
        # HotpotQA format: {question_id: {answer: str, supporting_facts: [[title, sent_id]]}}
        formatted = {}
        for pred in predictions:
            if isinstance(pred["prediction"], dict):
                formatted[pred["id"]] = pred["prediction"]
            else:
                # Fallback if prediction is just a string
                formatted[pred["id"]] = {
                    "answer": pred["prediction"],
                    "supporting_facts": []
                }
        return formatted

    elif dataset_name in ["gsm8k", "math", "aime"]:
        # Math format: just need the answers
        return [pred["prediction"] for pred in predictions]

    else:
        # Generic format
        return {pred.get("id", i): pred["prediction"] for i, pred in enumerate(predictions)}

def create_dataset_info(data: List[Dict]) -> Dict[str, Any]:
    """
    Create summary information about a dataset

    Args:
        data: Dataset

    Returns:
        Dataset information dictionary
    """
    info = {
        "num_samples": len(data),
        "sample_keys": list(data[0].keys()) if data else [],
        "sample_example": data[0] if data else None
    }

    # Add type-specific info
    if data and "question" in data[0]:
        avg_question_length = sum(len(item["question"]) for item in data) / len(data)
        info["avg_question_length"] = avg_question_length

    return info

def validate_dataset_format(data: List[Dict], required_keys: List[str]) -> bool:
    """
    Validate that dataset has required keys

    Args:
        data: Dataset to validate
        required_keys: Keys that must be present

    Returns:
        True if valid, False otherwise
    """
    if not data:
        return False

    # Check first few samples
    samples_to_check = min(5, len(data))
    for i in range(samples_to_check):
        if not all(key in data[i] for key in required_keys):
            return False

    return True

# Dataset loading registry
DATASET_LOADERS = {
    "gsm8k": load_gsm8k_platinum,
    "aime": load_aime_dataset,
    "math": load_math_dataset,
    "hotpotqa": load_hotpot_qa,
    "squad_v1": lambda: load_squad_dataset("v1"),
    "squad_v2": lambda: load_squad_dataset("v2"),
    "nq": load_natural_questions
}

def load_dataset_by_name(name: str, **kwargs) -> List[Dict]:
    """
    Load dataset by name using registered loaders

    Args:
        name: Dataset name
        **kwargs: Additional arguments for loader

    Returns:
        Loaded dataset
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")

    return DATASET_LOADERS[name](**kwargs)