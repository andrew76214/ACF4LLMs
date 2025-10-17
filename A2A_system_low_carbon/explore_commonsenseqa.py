#!/usr/bin/env python3
"""
Explore CommonsenseQA dataset structure to understand answer formats
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset
from utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=False)
logger = get_logger("commonsenseqa_exploration")

def explore_commonsenseqa():
    """Explore CommonsenseQA dataset structure"""
    logger.info("🔍 Loading CommonsenseQA dataset from HuggingFace...")

    try:
        # Load the dataset
        dataset = load_dataset("commonsense_qa")

        logger.info("📊 Dataset structure:")
        logger.info(f"Available splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            logger.info(f"\n=== {split_name.upper()} SPLIT ===")
            logger.info(f"Number of examples: {len(split_data)}")
            logger.info(f"Features: {split_data.features}")

            # Look at first few examples
            logger.info(f"\n--- First 3 examples from {split_name} ---")
            for i in range(min(3, len(split_data))):
                example = split_data[i]
                logger.info(f"\nExample {i+1}:")
                logger.info(f"ID: {example.get('id', 'N/A')}")
                logger.info(f"Question: {example.get('question', 'N/A')}")
                logger.info(f"Choices: {example.get('choices', 'N/A')}")
                logger.info(f"Answer Key: {example.get('answerKey', 'N/A')}")

                # Show the structure of choices if it's a dict
                if 'choices' in example and example['choices']:
                    choices = example['choices']
                    if isinstance(choices, dict):
                        logger.info("Choice structure:")
                        for key, value in choices.items():
                            logger.info(f"  {key}: {value}")

                logger.info("---")

        # Check validation set specifically
        if 'validation' in dataset:
            val_data = dataset['validation']
            logger.info(f"\n🎯 VALIDATION SET ANALYSIS")
            logger.info(f"Total validation examples: {len(val_data)}")

            # Analyze answer formats
            answer_keys = []
            for example in val_data[:100]:  # Check first 100
                answer_keys.append(example.get('answerKey', ''))

            unique_answers = set(answer_keys)
            logger.info(f"Unique answer keys found: {sorted(unique_answers)}")

            # Count frequency
            from collections import Counter
            answer_counts = Counter(answer_keys)
            logger.info(f"Answer key frequencies: {answer_counts}")

    except Exception as e:
        logger.error(f"Failed to load CommonsenseQA dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    explore_commonsenseqa()