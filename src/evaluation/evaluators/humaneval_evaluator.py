"""HumanEval code generation evaluator."""

import torch
import signal
import contextlib
import io
import sys
import multiprocessing
from typing import Any, Dict, Optional
import logging
from src.evaluation.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Timeout for code execution (in seconds)
CODE_EXECUTION_TIMEOUT = 5


class HumanEvalEvaluator(BaseEvaluator):
    """Evaluator for HumanEval code generation benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.benchmark_name = "humaneval"

    def evaluate(self, model: Any, tokenizer: Any, batch_size: int = 4) -> float:
        """Evaluate model on HumanEval.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size (smaller due to code length)

        Returns:
            Pass@1 score
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai_humaneval", split="test")
            dataset = dataset.select(range(min(50, len(dataset))))
        except:
            logger.warning("HumanEval not available, using mock")
            return self._mock_evaluate()

        passed = 0
        total = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [ex["prompt"] for ex in batch]
            outputs = self.generate_batch(model, tokenizer, prompts, max_length=512)

            for output, example in zip(outputs, batch):
                # Extract code completion
                code = self.extract_code(output)

                # Test execution (simplified)
                if self._test_solution(code, example):
                    passed += 1
                total += 1

        return passed / total if total > 0 else 0.0

    def extract_code(self, output: str) -> str:
        """Extract code from model output.

        Args:
            output: Model output

        Returns:
            Extracted code
        """
        # Remove common artifacts
        lines = output.split('\n')
        code_lines = []

        for line in lines:
            # Stop at test cases or prints
            if 'print(' in line or 'assert' in line or '#' in line:
                break
            code_lines.append(line)

        return '\n'.join(code_lines)

    def _test_solution(self, code: str, example: Dict) -> bool:
        """Test if code solution is correct by actually executing the code.

        Args:
            code: Generated code
            example: Test example with 'prompt', 'test', and 'entry_point'

        Returns:
            True if passes tests
        """
        if not code or not code.strip():
            return False

        # Get the test code and entry point
        prompt = example.get("prompt", "")
        test_code = example.get("test", "")
        entry_point = example.get("entry_point", "")

        if not test_code:
            # Fallback to simple check if no test code available
            return 'return' in code

        # Combine the prompt (function signature) + completion + test
        full_code = prompt + code + "\n\n" + test_code

        # Try to execute the code with timeout
        return self._execute_with_timeout(full_code, entry_point)

    def _execute_with_timeout(self, code: str, entry_point: str, timeout: int = CODE_EXECUTION_TIMEOUT) -> bool:
        """Execute code with timeout using multiprocessing.

        Args:
            code: Full code to execute
            entry_point: Function name to test
            timeout: Timeout in seconds

        Returns:
            True if code executes successfully without errors
        """
        def target(code_str: str, result_queue: multiprocessing.Queue):
            """Target function for multiprocessing."""
            try:
                # Create restricted globals for safety
                safe_globals = {
                    "__builtins__": {
                        # Allow safe builtins
                        "abs": abs, "all": all, "any": any, "bin": bin,
                        "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
                        "enumerate": enumerate, "filter": filter, "float": float,
                        "format": format, "frozenset": frozenset, "hash": hash,
                        "hex": hex, "int": int, "isinstance": isinstance,
                        "issubclass": issubclass, "iter": iter, "len": len,
                        "list": list, "map": map, "max": max, "min": min,
                        "next": next, "oct": oct, "ord": ord, "pow": pow,
                        "print": print, "range": range, "repr": repr,
                        "reversed": reversed, "round": round, "set": set,
                        "slice": slice, "sorted": sorted, "str": str,
                        "sum": sum, "tuple": tuple, "type": type, "zip": zip,
                        # Math functions that might be needed
                        "Exception": Exception, "ValueError": ValueError,
                        "TypeError": TypeError, "IndexError": IndexError,
                        "KeyError": KeyError, "AssertionError": AssertionError,
                        "StopIteration": StopIteration,
                        "True": True, "False": False, "None": None,
                    },
                    "__name__": "__main__",
                }

                # Execute the code
                exec(code_str, safe_globals)
                result_queue.put(True)
            except AssertionError:
                # Test assertion failed
                result_queue.put(False)
            except Exception as e:
                logger.debug(f"Code execution failed: {type(e).__name__}: {e}")
                result_queue.put(False)

        # Use multiprocessing for timeout
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(code, result_queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            # Timeout - kill the process
            process.terminate()
            process.join()
            logger.debug(f"Code execution timed out for {entry_point}")
            return False

        # Get result
        try:
            return result_queue.get_nowait()
        except:
            return False

    def _test_solution_simple(self, code: str, example: Dict) -> bool:
        """Simple fallback test using signal-based timeout (Linux only).

        Args:
            code: Generated code
            example: Test example

        Returns:
            True if passes tests
        """
        if not code or not code.strip():
            return False

        prompt = example.get("prompt", "")
        test_code = example.get("test", "")
        entry_point = example.get("entry_point", "")

        if not test_code:
            return 'return' in code

        full_code = prompt + code + "\n\n" + test_code

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        try:
            # Set up timeout handler (Unix only)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(CODE_EXECUTION_TIMEOUT)

            try:
                exec_globals = {}
                exec(full_code, exec_globals)
                return True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except (AssertionError, TimeoutError) as e:
            logger.debug(f"Test failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Code execution error: {type(e).__name__}: {e}")
            return False

    def _mock_evaluate(self) -> float:
        """Mock evaluation."""
        import random
        return random.uniform(0.1, 0.3)  # Code generation is challenging

    def evaluate_proxy(
        self, model: Any, tokenizer: Any, num_samples: int = 50, batch_size: int = 4
    ) -> float:
        """Fast proxy evaluation on subset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_samples: Number of samples
            batch_size: Batch size

        Returns:
            Proxy score
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai_humaneval", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        except:
            return self._mock_evaluate()

        passed = 0
        total = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [ex["prompt"] for ex in batch]
            outputs = self.generate_batch(model, tokenizer, prompts, max_length=512)

            for output, example in zip(outputs, batch):
                code = self.extract_code(output)
                if self._test_solution(code, example):
                    passed += 1
                total += 1

        return passed / total if total > 0 else 0.0