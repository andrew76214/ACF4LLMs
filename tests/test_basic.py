#!/usr/bin/env python3
"""Basic tests for the Agentic Compression Framework."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.common.schemas import ModelSpec, CompressionStrategy, EvaluationResult
        print("✅ Common schemas imported")
    except ImportError as e:
        print(f"❌ Failed to import common schemas: {e}")
        return False

    try:
        from src.coordinator.spec_inference import infer_spec
        print("✅ Spec inference imported")
    except ImportError as e:
        print(f"❌ Failed to import spec inference: {e}")
        return False

    try:
        from src.coordinator.pareto import ParetoFrontier
        print("✅ Pareto frontier imported")
    except ImportError as e:
        print(f"❌ Failed to import Pareto frontier: {e}")
        return False

    try:
        from src.agents.quantization_agent import get_quantization_subagent
        print("✅ Quantization agent imported")
    except ImportError as e:
        print(f"❌ Failed to import quantization agent: {e}")
        return False

    try:
        from src.agents.evaluation_agent import get_evaluation_subagent
        print("✅ Evaluation agent imported")
    except ImportError as e:
        print(f"❌ Failed to import evaluation agent: {e}")
        return False

    try:
        from src.coordinator.langgraph_coordinator import LangGraphCoordinator
        print("✅ Coordinator imported")
    except ImportError as e:
        print(f"❌ Failed to import coordinator: {e}")
        return False

    return True


def test_spec_inference():
    """Test specification inference."""
    print("\nTesting spec inference...")

    from src.coordinator.spec_inference import infer_spec, format_spec_summary

    # Test with known models
    test_cases = [
        ("gpt2", "gsm8k"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "commonsenseqa"),
        ("mistralai/Mistral-7B-v0.1", "humaneval"),
    ]

    for model, dataset in test_cases:
        try:
            spec = infer_spec(model, dataset)
            assert spec.model_name == model
            assert spec.model_size_gb > 0
            assert len(spec.preferred_methods) > 0
            print(f"✅ Spec inference works for {model}")
        except Exception as e:
            print(f"❌ Spec inference failed for {model}: {e}")
            return False

    return True


def test_pareto_frontier():
    """Test Pareto frontier tracking."""
    print("\nTesting Pareto frontier...")

    from src.coordinator.pareto import ParetoFrontier
    from src.common.schemas import CompressionStrategy, EvaluationResult, CompressionMethod
    from uuid import uuid4

    try:
        # Create frontier
        frontier = ParetoFrontier()

        # Add some test solutions
        for i in range(3):
            strategy = CompressionStrategy(
                episode_id=i,
                strategy_id=str(uuid4())[:8],
                methods=[CompressionMethod.AUTOROUND],
                quantization_bits=4 if i == 0 else 8,
            )

            result = EvaluationResult(
                strategy_id=strategy.strategy_id,
                checkpoint_path=f"/test/checkpoint_{i}",
                model_size_gb=10.0 - i,
                compression_ratio=1.5 + i * 0.5,
                accuracy=0.95 - i * 0.05,
                latency_ms=100 + i * 10,
                throughput_tokens_per_sec=100,
                memory_gb=8.0,
                evaluation_time_sec=60.0,
            )

            is_pareto, dominated = frontier.add_solution(strategy, result)
            print(f"  Solution {i}: Pareto={is_pareto}, Dominates {len(dominated)} solutions")

        # Check frontier
        assert len(frontier.solutions) > 0
        print(f"✅ Pareto frontier works, {len(frontier.solutions)} solutions on frontier")

    except Exception as e:
        print(f"❌ Pareto frontier test failed: {e}")
        return False

    return True


def test_mock_quantization():
    """Test mock quantization tool."""
    print("\nTesting mock quantization...")

    try:
        from src.agents.quantization_agent import quantize_model
    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True  # Skip but don't fail

    try:
        result = quantize_model.invoke({
            "model_path": "gpt2",
            "method": "autoround",
            "bit_width": 4,
        })

        assert "checkpoint_path" in result
        assert "model_size_gb" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] > 1

        print(f"✅ Mock quantization works, compression ratio: {result['compression_ratio']:.1f}x")

    except Exception as e:
        print(f"❌ Mock quantization failed: {e}")
        return False

    return True


def test_mock_evaluation():
    """Test mock evaluation tool."""
    print("\nTesting mock evaluation...")

    try:
        from src.agents.evaluation_agent import evaluate_model
    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True  # Skip but don't fail

    try:
        result = evaluate_model.invoke({
            "checkpoint_path": "/test/checkpoint",
            "benchmarks": ["gsm8k", "commonsenseqa"],
            "use_proxy": True,
        })

        assert "benchmark_scores" in result
        assert "average_accuracy" in result
        assert "latency_ms" in result
        assert len(result["benchmark_scores"]) == 2

        print(f"✅ Mock evaluation works, avg accuracy: {result['average_accuracy']:.3f}")

    except Exception as e:
        print(f"❌ Mock evaluation failed: {e}")
        return False

    return True


def test_lora_trainer():
    """Test LoRA trainer initialization and mock fine-tuning."""
    print("\nTesting LoRA trainer...")

    try:
        from src.tools.quantization_wrapper import get_quantizer, LoRATrainer
    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True

    try:
        # Test LoRATrainer initialization
        trainer = get_quantizer("lora")
        assert trainer.method == "lora", f"Expected method 'lora', got '{trainer.method}'"
        assert isinstance(trainer, LoRATrainer)
        print(f"  LoRATrainer initialized ✓")

        # Test mock fine-tuning
        result = trainer.finetune(
            model_name="gpt2",
            lora_rank=8,
            training_steps=1,
            output_dir="/tmp/test_lora_adapter",
        )

        assert "checkpoint_path" in result
        assert "adapter_size_mb" in result
        assert "training_loss" in result
        assert "training_time_sec" in result
        print(f"  Mock fine-tuning completed: {result['checkpoint_path']} ✓")

        print("✅ LoRA trainer works")
        return True

    except Exception as e:
        print(f"❌ LoRA trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qlora_trainer():
    """Test QLoRA trainer initialization and mock fine-tuning."""
    print("\nTesting QLoRA trainer...")

    try:
        from src.tools.quantization_wrapper import get_quantizer, QLoRATrainer
    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True

    try:
        # Test QLoRATrainer initialization
        trainer = get_quantizer("qlora")
        assert trainer.method == "qlora", f"Expected method 'qlora', got '{trainer.method}'"
        assert isinstance(trainer, QLoRATrainer)
        print(f"  QLoRATrainer initialized ✓")

        # Test mock fine-tuning
        result = trainer.finetune(
            model_name="gpt2",
            lora_rank=8,
            bits=4,
            training_steps=1,
            output_dir="/tmp/test_qlora_adapter",
        )

        assert "checkpoint_path" in result
        assert "adapter_size_mb" in result
        assert "training_loss" in result
        assert "base_model_bits" in result
        assert result["base_model_bits"] == 4
        print(f"  Mock fine-tuning completed: {result['checkpoint_path']} ✓")

        print("✅ QLoRA trainer works")
        return True

    except Exception as e:
        print(f"❌ QLoRA trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_tools():
    """Test LoRA/QLoRA tool functions."""
    print("\nTesting LoRA/QLoRA tools...")

    try:
        from src.agents.quantization_agent import (
            apply_lora_finetuning,
            apply_qlora_finetuning,
            list_available_quantization_methods,
        )
    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True

    try:
        # Test that tools are callable
        assert callable(apply_lora_finetuning.invoke)
        assert callable(apply_qlora_finetuning.invoke)
        print(f"  LoRA/QLoRA tools are callable ✓")

        # Test list_available_quantization_methods includes LoRA/QLoRA
        methods = list_available_quantization_methods.invoke({})
        method_names = [m["name"] for m in methods]
        assert "lora" in method_names, "LoRA not in available methods"
        assert "qlora" in method_names, "QLoRA not in available methods"
        print(f"  list_available_quantization_methods includes LoRA/QLoRA ✓")

        # Test apply_lora_finetuning tool (mock)
        result = apply_lora_finetuning.invoke({
            "model_path": "gpt2",
            "lora_rank": 16,
            "training_steps": 1,
            "output_dir": "/tmp/test_lora_tool",
        })
        assert "checkpoint_path" in result
        print(f"  apply_lora_finetuning tool works ✓")

        # Test apply_qlora_finetuning tool (mock)
        result = apply_qlora_finetuning.invoke({
            "model_path": "gpt2",
            "lora_rank": 16,
            "bits": 4,
            "training_steps": 1,
            "output_dir": "/tmp/test_qlora_tool",
        })
        assert "checkpoint_path" in result
        assert "base_model_bits" in result
        print(f"  apply_qlora_finetuning tool works ✓")

        print("✅ LoRA/QLoRA tools work")
        return True

    except Exception as e:
        print(f"❌ LoRA/QLoRA tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evolutionary_search_fitness():
    """Test evolutionary search with real fitness computation."""
    print("\nTesting evolutionary search with real fitness...")

    try:
        from src.agents.search_agent import evolutionary_search, _compute_fitness

        # Test _compute_fitness function
        result1 = {'accuracy': 0.85, 'latency_ms': 100, 'memory_gb': 8}
        fitness1 = _compute_fitness(result1)
        # Expected: 0.85 * 1.0 + 0.5 * 0.3 + 0.5 * 0.2 = 1.1
        assert 1.0 < fitness1 < 1.2, f"Expected ~1.1, got {fitness1}"
        print(f"  _compute_fitness({result1}) = {fitness1:.4f} ✓")

        # Test that better performance = higher fitness
        result2 = {'accuracy': 0.85, 'latency_ms': 50, 'memory_gb': 4}
        fitness2 = _compute_fitness(result2)
        assert fitness2 > fitness1, "Better latency/memory should have higher fitness"
        print(f"  Better performance has higher fitness: {fitness2:.4f} > {fitness1:.4f} ✓")

        # Test evolutionary_search with no history (initial population)
        result = evolutionary_search.invoke({
            'population_size': 5,
            'num_generations': 2,
        })
        assert 'candidates_to_evaluate' in result
        assert len(result['candidates_to_evaluate']) == 5
        assert result['is_complete'] == False
        print(f"  Initial population: {len(result['candidates_to_evaluate'])} candidates ✓")

        # Test evolutionary_search with fitness history from real results
        fitness_history = [
            {'parameters': {'quantization_bits': 4, 'quantization_method': 'gptq'},
             'result': {'accuracy': 0.90, 'latency_ms': 80, 'memory_gb': 6}},
            {'parameters': {'quantization_bits': 8, 'quantization_method': 'int8'},
             'result': {'accuracy': 0.88, 'latency_ms': 100, 'memory_gb': 8}},
            {'parameters': {'quantization_bits': 4, 'quantization_method': 'awq'},
             'result': {'accuracy': 0.92, 'latency_ms': 90, 'memory_gb': 5}},
            {'parameters': {'quantization_bits': 4, 'quantization_method': 'autoround'},
             'result': {'accuracy': 0.85, 'latency_ms': 70, 'memory_gb': 6}},
            {'parameters': {'quantization_bits': 8, 'quantization_method': 'gptq'},
             'result': {'accuracy': 0.87, 'latency_ms': 120, 'memory_gb': 10}},
        ]
        result = evolutionary_search.invoke({
            'population_size': 5,
            'num_generations': 2,
            'fitness_history': fitness_history,
        })
        assert result['best_fitness'] > 0, "Should have computed real fitness"
        assert 'candidates_to_evaluate' in result
        print(f"  Evolution with real fitness: best={result['best_fitness']:.4f} ✓")

        print("✅ Evolutionary search with real fitness works")
        return True

    except ImportError as e:
        print(f"⚠️ Skipped (missing dependency): {e}")
        return True  # Skip but don't fail
    except Exception as e:
        print(f"❌ Evolutionary search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinator_creation():
    """Test coordinator creation."""
    print("\nTesting coordinator creation...")

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Skipped (OPENAI_API_KEY not set)")
        return True  # Skip but don't fail

    from src.coordinator.langgraph_coordinator import LangGraphCoordinator

    try:
        coordinator = LangGraphCoordinator(
            model_name="gpt2",
            dataset="gsm8k",
            max_episodes=1,
        )

        assert coordinator.model_name == "gpt2"
        assert coordinator.max_episodes == 1
        print("✅ Coordinator created successfully")

    except Exception as e:
        print(f"❌ Coordinator creation failed: {e}")
        return False

    return True


def run_all_tests():
    """Run all basic tests."""
    print("=" * 60)
    print("🧪 Running Basic Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Spec Inference", test_spec_inference),
        ("Pareto Frontier", test_pareto_frontier),
        ("Mock Quantization", test_mock_quantization),
        ("Mock Evaluation", test_mock_evaluation),
        ("LoRA Trainer", test_lora_trainer),
        ("QLoRA Trainer", test_qlora_trainer),
        ("LoRA/QLoRA Tools", test_lora_tools),
        ("Evolutionary Search Fitness", test_evolutionary_search_fitness),
        ("Coordinator Creation", test_coordinator_creation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)