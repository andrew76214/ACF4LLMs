"""
Text processing utilities for A2A Pipeline
Includes answer extraction, normalization, and parsing functions
"""

import re
import json
import string
from typing import Optional, Dict, Any, List
from configs.config import config

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} format used in math problems

    Args:
        text: Input text containing boxed answer

    Returns:
        Extracted answer string or None if not found
    """
    # Pattern for \\boxed{answer}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer found
        return matches[-1].strip()

    # Fallback: look for boxed without backslashes
    pattern = r'boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if matches:
        return matches[-1].strip()

    return None

def extract_json_answer(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON answer from text, commonly used in QA tasks

    Args:
        text: Input text containing JSON

    Returns:
        Parsed JSON dictionary or None if not found/invalid
    """
    # Find JSON-like patterns
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            # Try to parse as JSON
            parsed = json.loads(match)
            return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try to extract answer and supporting facts manually
    answer_pattern = r'"answer":\s*"([^"]*)"'
    support_pattern = r'"supporting_facts":\s*(\[[^\]]*\])'

    answer_match = re.search(answer_pattern, text)
    support_match = re.search(support_pattern, text)

    if answer_match:
        result = {"answer": answer_match.group(1)}
        if support_match:
            try:
                support_facts = json.loads(support_match.group(1))
                result["supporting_facts"] = support_facts
            except json.JSONDecodeError:
                pass
        return result

    return None

def extract_final_answer(text: str, task_type: str) -> Optional[str]:
    """
    Extract final answer based on task type

    Args:
        text: Input text
        task_type: Type of task (math, qa, multihop)

    Returns:
        Extracted answer or None
    """
    if task_type == "math":
        return extract_boxed_answer(text)
    elif task_type in ["qa", "multihop"]:
        json_result = extract_json_answer(text)
        if json_result and "answer" in json_result:
            return json_result["answer"]
        # Fallback: treat entire text as answer for simple QA
        return text.strip()
    else:
        return text.strip()

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (used in evaluation)

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove common articles
    articles = ['the', 'a', 'an']
    words = text.split()
    words = [w for w in words if w not in articles]

    return ' '.join(words)

def clean_math_expression(text: str) -> str:
    """
    Clean mathematical expression for parsing

    Args:
        text: Math expression

    Returns:
        Cleaned expression
    """
    if not text:
        return ""

    # Remove common LaTeX commands that don't affect value
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)

    # Clean up spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_number_from_text(text: str) -> Optional[str]:
    """
    Extract numerical answer from text as fallback

    Args:
        text: Input text

    Returns:
        Extracted number as string or None
    """
    # Pattern for various number formats
    patterns = [
        r'-?\d+\.?\d*',  # Decimal numbers
        r'-?\d+/\d+',    # Fractions
        r'-?\d+',        # Integers
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last number found
            return matches[-1]

    return None

def split_reasoning_and_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Split text into reasoning and final answer parts

    Args:
        text: Full model output

    Returns:
        Tuple of (reasoning, final_answer)
    """
    # Look for thinking tags
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
        # Everything after </think> is the final answer
        after_think = text[think_match.end():].strip()
        return reasoning, after_think
    else:
        # No explicit thinking tags, treat entire text as reasoning + answer
        # Try to find a clear answer delimiter
        if '\\boxed{' in text:
            boxed_match = re.search(r'\\boxed\{[^}]+\}', text)
            if boxed_match:
                reasoning = text[:boxed_match.start()].strip()
                answer = text[boxed_match.start():].strip()
                return reasoning, answer

        # Default: no clear separation
        return None, text.strip()

def format_supporting_facts(facts: List[List]) -> str:
    """
    Format supporting facts for display

    Args:
        facts: List of [title, sentence_id] pairs

    Returns:
        Formatted string
    """
    if not facts:
        return "None"

    formatted = []
    for fact in facts:
        if len(fact) >= 2:
            title, sent_id = fact[0], fact[1]
            formatted.append(f"({title}, {sent_id})")

    return "; ".join(formatted)

def validate_json_format(text: str, required_keys: List[str]) -> bool:
    """
    Validate that text contains valid JSON with required keys

    Args:
        text: Text to validate
        required_keys: Keys that must be present

    Returns:
        True if valid, False otherwise
    """
    try:
        data = extract_json_answer(text)
        if not data:
            return False

        return all(key in data for key in required_keys)
    except:
        return False

def repair_json(text: str) -> Optional[str]:
    """
    Attempt to repair malformed JSON

    Args:
        text: Potentially malformed JSON text

    Returns:
        Repaired JSON string or None if can't be fixed
    """
    if not text:
        return None

    # Common fixes
    fixes = [
        (r"'", '"'),  # Single quotes to double quotes
        (r'(\w+):', r'"\1":'),  # Unquoted keys
        (r':\s*([^",\[\]{}\s]+)(?=\s*[,\}])', r': "\1"'),  # Unquoted string values
    ]

    fixed_text = text
    for pattern, replacement in fixes:
        fixed_text = re.sub(pattern, replacement, fixed_text)

    try:
        json.loads(fixed_text)
        return fixed_text
    except json.JSONDecodeError:
        return None