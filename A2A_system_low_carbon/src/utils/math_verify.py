"""
Math Verification Utility
Based on Math-Verify principles for mathematical equivalence checking
"""

import re
import sympy as sp
from sympy import sympify, simplify, N, latex
from typing import Union, Any, Optional
# Simple logger to avoid circular imports
class SimpleLogger:
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = SimpleLogger()

def normalize_latex(latex_str: str) -> str:
    """Normalize LaTeX expressions for parsing"""
    latex_str = latex_str.strip()

    # Remove common LaTeX artifacts
    latex_str = latex_str.replace("\\,", "")
    latex_str = latex_str.replace("\\:", "")
    latex_str = latex_str.replace("\\;", "")
    latex_str = latex_str.replace("\\!", "")
    latex_str = latex_str.replace("\\ ", "")

    # Normalize fractions
    latex_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', latex_str)

    # Normalize square roots
    latex_str = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', latex_str)
    latex_str = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'(\2)**(1/(\1))', latex_str)

    # Normalize powers
    latex_str = re.sub(r'\^{([^}]+)}', r'**(\1)', latex_str)
    latex_str = re.sub(r'\^([^{}\s]+)', r'**\1', latex_str)

    # Normalize common functions
    latex_str = latex_str.replace("\\sin", "sin")
    latex_str = latex_str.replace("\\cos", "cos")
    latex_str = latex_str.replace("\\tan", "tan")
    latex_str = latex_str.replace("\\log", "log")
    latex_str = latex_str.replace("\\ln", "ln")

    # Normalize pi and e
    latex_str = latex_str.replace("\\pi", "pi")
    latex_str = latex_str.replace("\\e", "E")

    return latex_str

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format"""
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1]  # Return the last boxed answer

    # Look for $...$ pattern at end
    dollar_pattern = r'\$([^$]+)\$\s*$'
    matches = re.findall(dollar_pattern, text.strip())
    if matches:
        return matches[-1]

    # Look for final number pattern
    number_pattern = r'(?:answer|result|solution).*?(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)'
    matches = re.findall(number_pattern, text.lower())
    if matches:
        return matches[-1]

    # Last resort: find any number at the end
    final_number = re.findall(r'([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$', text.strip())
    if final_number:
        return final_number[-1]

    return None

def parse_answer(answer_str: str) -> Any:
    """Parse mathematical expression into SymPy object"""
    if not answer_str:
        return None

    answer_str = str(answer_str).strip()

    # Handle empty or None
    if not answer_str or answer_str.lower() in ['none', 'null', '']:
        return None

    try:
        # First try to normalize LaTeX
        normalized = normalize_latex(answer_str)

        # Try direct sympify
        try:
            result = sympify(normalized, rational=True)
            return result
        except:
            pass

        # Try as plain number
        try:
            if '/' in normalized:
                parts = normalized.split('/')
                if len(parts) == 2:
                    num, den = parts
                    return sp.Rational(int(num), int(den))
            else:
                # Try as float first, then convert to rational if no decimal
                float_val = float(normalized)
                if float_val.is_integer():
                    return sp.Integer(int(float_val))
                else:
                    return sp.Float(float_val)
        except:
            pass

        # Try as expression with common math functions
        try:
            # Replace some common patterns
            expr = normalized.replace('^', '**')
            expr = re.sub(r'(\d+)\s*\*\*\s*\(([^)]+)\)', r'\1**(\2)', expr)
            result = sympify(expr, rational=True)
            return result
        except:
            pass

        # Last resort: return as string for exact matching
        return answer_str

    except Exception as e:
        logger.warning(f"Failed to parse answer '{answer_str}': {e}")
        return answer_str

def verify(gold_answer: Any, predicted_answer: Any, tolerance: float = 1e-10) -> bool:
    """
    Verify if two mathematical expressions are equivalent

    Args:
        gold_answer: Ground truth answer (parsed)
        predicted_answer: Model's answer (parsed)
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if answers are mathematically equivalent
    """
    try:
        # Handle None cases
        if gold_answer is None and predicted_answer is None:
            return True
        if gold_answer is None or predicted_answer is None:
            return False

        # If both are strings, do exact comparison
        if isinstance(gold_answer, str) and isinstance(predicted_answer, str):
            return gold_answer.strip().lower() == predicted_answer.strip().lower()

        # If one is string and other is not, convert and try
        if isinstance(gold_answer, str):
            gold_answer = parse_answer(gold_answer)
        if isinstance(predicted_answer, str):
            predicted_answer = parse_answer(predicted_answer)

        # Try direct equality first
        if gold_answer == predicted_answer:
            return True

        # If both are SymPy objects
        if hasattr(gold_answer, 'equals') and hasattr(predicted_answer, 'equals'):
            try:
                return gold_answer.equals(predicted_answer)
            except:
                pass

        # Try simplification and comparison
        try:
            diff = simplify(gold_answer - predicted_answer)
            if diff == 0:
                return True
        except:
            pass

        # Try numerical evaluation
        try:
            gold_num = complex(N(gold_answer))
            pred_num = complex(N(predicted_answer))

            return abs(gold_num - pred_num) < tolerance
        except:
            pass

        # For sets, lists, tuples
        if isinstance(gold_answer, (list, tuple, set)) and isinstance(predicted_answer, (list, tuple, set)):
            if len(gold_answer) != len(predicted_answer):
                return False
            gold_set = set(str(x) for x in gold_answer)
            pred_set = set(str(x) for x in predicted_answer)
            return gold_set == pred_set

        # Final fallback: string comparison
        return str(gold_answer).strip() == str(predicted_answer).strip()

    except Exception as e:
        logger.error(f"Error in verification: {e}")
        return False

def extract_and_verify_answer(model_output: str, gold_answer: str) -> dict:
    """
    Complete pipeline: extract answer from model output and verify against gold

    Args:
        model_output: Raw model output text
        gold_answer: Ground truth answer string

    Returns:
        Dict with extraction and verification results
    """
    # Extract answer from model output
    extracted = extract_boxed_answer(model_output)

    if extracted is None:
        logger.warning(f"Could not extract answer from: {model_output[:100]}...")
        return {
            "extracted": None,
            "parsed_prediction": None,
            "parsed_gold": None,
            "correct": False,
            "error": "Could not extract answer"
        }

    # Parse both answers
    try:
        parsed_pred = parse_answer(extracted)
        parsed_gold = parse_answer(gold_answer)

        # Verify equivalence
        is_correct = verify(parsed_gold, parsed_pred)

        return {
            "extracted": extracted,
            "parsed_prediction": parsed_pred,
            "parsed_gold": parsed_gold,
            "correct": is_correct,
            "error": None
        }

    except Exception as e:
        logger.error(f"Error in extract_and_verify_answer: {e}")
        return {
            "extracted": extracted,
            "parsed_prediction": None,
            "parsed_gold": None,
            "correct": False,
            "error": str(e)
        }

# For backward compatibility with the pipeline
def parse(answer_str: str) -> Any:
    """Alias for parse_answer"""
    return parse_answer(answer_str)

# Test function
def test_math_verify():
    """Test the math verification system"""
    test_cases = [
        ("4", "4", True),
        ("2+2", "4", True),
        ("1/2", "0.5", True),
        ("\\frac{1}{2}", "0.5", True),
        ("\\boxed{42}", "42", True),
        ("The answer is \\boxed{123}", "123", True),
        ("2*3", "6", True),
        ("sqrt(4)", "2", True),
        ("pi", "3.14159", False),  # Should be false due to precision
        ("5", "6", False),
    ]

    print("Testing Math Verification System...")
    passed = 0
    for pred, gold, expected in test_cases:
        if "boxed" in pred:
            extracted = extract_boxed_answer(pred)
            parsed_pred = parse_answer(extracted) if extracted else None
        else:
            parsed_pred = parse_answer(pred)

        parsed_gold = parse_answer(gold)
        result = verify(parsed_gold, parsed_pred)

        status = "✅" if result == expected else "❌"
        print(f"{status} verify('{pred}', '{gold}') = {result} (expected: {expected})")

        if result == expected:
            passed += 1

    print(f"Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)

if __name__ == "__main__":
    test_math_verify()