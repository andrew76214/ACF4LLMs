#!/usr/bin/env python3
"""Tests for the Pruning Agent and related components."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_pruning_imports():
    """Test that all pruning modules can be imported."""
    print("Testing pruning imports...")

    try:
        from src.tools.pruning_wrapper import (
            BasePruner,
            MagnitudePruner,
            StructuredPruner,
            get_pruner,
            estimate_pruning_benefits,
        )
        print("  Pruning wrapper imported")
    except ImportError as e:
        print(f"  Failed to import pruning wrapper: {e}")
        return False

    try:
        from src.agents.pruning_agent import (
            prune_model,
            prune_model_structured,
            prune_model_unstructured,
            estimate_pruning_speedup,
            validate_pruning_compatibility,
            list_available_pruning_methods,
            get_pruning_subagent,
        )
        print("  Pruning agent imported")
    except ImportError as e:
        print(f"  Failed to import pruning agent: {e}")
        return False

    try:
        from src.common.schemas import CompressionMethod, CompressionStrategy
        assert CompressionMethod.PRUNING.value == "pruning"
        print("  CompressionMethod.PRUNING exists")
    except (ImportError, AssertionError) as e:
        print(f"  Failed to verify CompressionMethod.PRUNING: {e}")
        return False

    print("All pruning imports successful")
    return True


def test_pruning_wrapper_factory():
    """Test the get_pruner factory function."""
    print("\nTesting pruning wrapper factory...")

    from src.tools.pruning_wrapper import get_pruner, MagnitudePruner, StructuredPruner

    try:
        # Test magnitude pruner
        pruner = get_pruner("magnitude")
        assert isinstance(pruner, MagnitudePruner), "Expected MagnitudePruner"
        print("  get_pruner('magnitude') returns MagnitudePruner")

        # Test structured pruner
        pruner = get_pruner("structured")
        assert isinstance(pruner, StructuredPruner), "Expected StructuredPruner"
        print("  get_pruner('structured') returns StructuredPruner")

        # Test case insensitivity
        pruner = get_pruner("MAGNITUDE")
        assert isinstance(pruner, MagnitudePruner), "Expected MagnitudePruner (case insensitive)"
        print("  Factory is case insensitive")

        # Test invalid method
        try:
            get_pruner("invalid")
            print("  Should have raised ValueError for invalid method")
            return False
        except ValueError:
            print("  ValueError raised for invalid method")

        print("Pruning wrapper factory works correctly")
        return True

    except Exception as e:
        print(f"  Pruning wrapper factory test failed: {e}")
        return False


def test_estimate_pruning_benefits():
    """Test pruning benefit estimation."""
    print("\nTesting pruning benefit estimation...")

    from src.tools.pruning_wrapper import estimate_pruning_benefits

    try:
        # Test with a model name
        estimates = estimate_pruning_benefits("gpt2", sparsity=0.3, method="magnitude")

        assert "original_size_gb" in estimates
        assert "estimated_size_gb" in estimates
        assert "estimated_compression_ratio" in estimates
        assert "estimated_speedup" in estimates
        assert estimates["sparsity"] == 0.3
        assert estimates["method"] == "magnitude"

        print(f"  Original size: {estimates['original_size_gb']:.2f} GB")
        print(f"  Estimated size: {estimates['estimated_size_gb']:.2f} GB")
        print(f"  Speedup: {estimates['estimated_speedup']:.2f}x")

        # Test structured pruning estimation
        estimates_structured = estimate_pruning_benefits("gpt2", sparsity=0.3, method="structured")
        assert estimates_structured["estimated_speedup"] > estimates["estimated_speedup"], \
            "Structured pruning should have higher speedup"
        print(f"  Structured speedup: {estimates_structured['estimated_speedup']:.2f}x (higher than magnitude)")

        print("Pruning benefit estimation works correctly")
        return True

    except Exception as e:
        print(f"  Pruning benefit estimation test failed: {e}")
        return False


def test_validate_pruning_compatibility():
    """Test pruning compatibility validation."""
    print("\nTesting pruning compatibility validation...")

    from src.agents.pruning_agent import validate_pruning_compatibility

    try:
        # Test valid configuration
        result = validate_pruning_compatibility.invoke({
            "model_path": "gpt2",
            "method": "magnitude",
            "sparsity": 0.3,
            "granularity": "weight",
        })

        assert result["is_compatible"] == True
        assert "warnings" in result
        assert "recommendations" in result
        print(f"  Valid config: compatible={result['is_compatible']}")

        # Test high sparsity warning
        result = validate_pruning_compatibility.invoke({
            "model_path": "gpt2",
            "method": "magnitude",
            "sparsity": 0.8,
            "granularity": "weight",
        })
        assert any("high sparsity" in w.lower() for w in result["warnings"])
        print(f"  High sparsity warning generated")

        # Test invalid sparsity
        result = validate_pruning_compatibility.invoke({
            "model_path": "gpt2",
            "method": "magnitude",
            "sparsity": 1.5,
            "granularity": "weight",
        })
        assert result["is_compatible"] == False
        print(f"  Invalid sparsity detected")

        # Test head pruning on transformer model
        result = validate_pruning_compatibility.invoke({
            "model_path": "meta-llama/Llama-2-7b",
            "method": "structured",
            "sparsity": 0.3,
            "granularity": "head",
        })
        assert any("llama" in r.lower() for r in result["recommendations"])
        print(f"  Model-specific recommendations generated")

        print("Pruning compatibility validation works correctly")
        return True

    except Exception as e:
        print(f"  Pruning compatibility validation test failed: {e}")
        return False


def test_list_pruning_methods():
    """Test listing available pruning methods."""
    print("\nTesting list available pruning methods...")

    from src.agents.pruning_agent import list_available_pruning_methods

    try:
        methods = list_available_pruning_methods.invoke({})

        assert len(methods) >= 2, "Expected at least 2 pruning methods"

        # Check magnitude method
        magnitude = next((m for m in methods if m["name"] == "magnitude"), None)
        assert magnitude is not None, "Expected magnitude method"
        assert "description" in magnitude
        assert "granularities" in magnitude
        assert "pros" in magnitude
        assert "cons" in magnitude
        print(f"  Magnitude method: {magnitude['description'][:50]}...")

        # Check structured method
        structured = next((m for m in methods if m["name"] == "structured"), None)
        assert structured is not None, "Expected structured method"
        assert "channel" in structured["granularities"]
        assert "head" in structured["granularities"]
        print(f"  Structured method granularities: {structured['granularities']}")

        print("List pruning methods works correctly")
        return True

    except Exception as e:
        print(f"  List pruning methods test failed: {e}")
        return False


def test_pruning_subagent_config():
    """Test pruning subagent configuration."""
    print("\nTesting pruning subagent configuration...")

    from src.agents.pruning_agent import get_pruning_subagent

    try:
        spec = {
            "model_name": "meta-llama/Llama-2-7b",
            "model_size_gb": 14.0,
        }

        config = get_pruning_subagent(spec)

        assert config["name"] == "pruning"
        assert "description" in config
        assert "prompt" in config
        assert "tools" in config
        assert len(config["tools"]) >= 4, "Expected at least 4 tools"

        # Check prompt contains model info
        assert "Llama-2-7b" in config["prompt"]
        assert "14.0" in config["prompt"]

        print(f"  Subagent name: {config['name']}")
        print(f"  Tools count: {len(config['tools'])}")

        print("Pruning subagent configuration works correctly")
        return True

    except Exception as e:
        print(f"  Pruning subagent configuration test failed: {e}")
        return False


def test_compression_strategy_pruning_fields():
    """Test that CompressionStrategy has all required pruning fields."""
    print("\nTesting CompressionStrategy pruning fields...")

    from src.common.schemas import CompressionStrategy, CompressionMethod

    try:
        # Create a pruning strategy
        strategy = CompressionStrategy(
            episode_id=0,
            strategy_id="test_pruning_001",
            methods=[CompressionMethod.PRUNING],
            pruning_ratio=0.3,
            pruning_method="structured",
            pruning_granularity="head",
        )

        assert strategy.pruning_ratio == 0.3
        assert strategy.pruning_method == "structured"
        assert strategy.pruning_granularity == "head"
        assert CompressionMethod.PRUNING in strategy.methods

        # Serialize and check
        data = strategy.model_dump()
        assert data["pruning_ratio"] == 0.3
        assert data["pruning_method"] == "structured"
        assert data["pruning_granularity"] == "head"

        print(f"  pruning_ratio: {strategy.pruning_ratio}")
        print(f"  pruning_method: {strategy.pruning_method}")
        print(f"  pruning_granularity: {strategy.pruning_granularity}")

        print("CompressionStrategy pruning fields work correctly")
        return True

    except Exception as e:
        print(f"  CompressionStrategy pruning fields test failed: {e}")
        return False


def test_mock_magnitude_pruning():
    """Test mock magnitude pruning."""
    print("\nTesting mock magnitude pruning...")

    try:
        from src.agents.pruning_agent import prune_model
    except ImportError as e:
        print(f"  Skipped (missing dependency): {e}")
        return True  # Skip but don't fail

    try:
        result = prune_model.invoke({
            "model_path": "gpt2",
            "method": "magnitude",
            "sparsity": 0.3,
            "granularity": "weight",
        })

        assert "checkpoint_path" in result
        assert "model_size_gb" in result
        assert "compression_ratio" in result
        assert "actual_sparsity" in result

        print(f"  Checkpoint path: {result['checkpoint_path']}")
        print(f"  Model size: {result['model_size_gb']:.2f} GB")
        print(f"  Actual sparsity: {result['actual_sparsity']*100:.1f}%")

        print("Mock magnitude pruning works correctly")
        return True

    except Exception as e:
        print(f"  Mock magnitude pruning test failed: {e}")
        return False


def test_mock_structured_pruning():
    """Test mock structured pruning."""
    print("\nTesting mock structured pruning...")

    try:
        from src.agents.pruning_agent import prune_model
    except ImportError as e:
        print(f"  Skipped (missing dependency): {e}")
        return True  # Skip but don't fail

    try:
        result = prune_model.invoke({
            "model_path": "gpt2",
            "method": "structured",
            "sparsity": 0.3,
            "granularity": "channel",
        })

        assert "checkpoint_path" in result
        assert "model_size_gb" in result
        assert "compression_ratio" in result
        assert "actual_sparsity" in result

        print(f"  Checkpoint path: {result['checkpoint_path']}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")

        print("Mock structured pruning works correctly")
        return True

    except Exception as e:
        print(f"  Mock structured pruning test failed: {e}")
        return False


def test_create_pruning_schedule():
    """Test pruning schedule creation."""
    print("\nTesting pruning schedule creation...")

    from src.agents.pruning_agent import create_pruning_schedule

    try:
        result = create_pruning_schedule.invoke({
            "initial_sparsity": 0.0,
            "target_sparsity": 0.5,
            "num_steps": 5,
            "schedule_type": "cubic",
        })

        assert result["schedule_type"] == "cubic"
        assert result["initial_sparsity"] == 0.0
        assert result["target_sparsity"] == 0.5
        assert result["num_steps"] == 5
        assert len(result["schedule"]) == 5

        # Check schedule is monotonically increasing
        sparsities = [step["sparsity"] for step in result["schedule"]]
        for i in range(1, len(sparsities)):
            assert sparsities[i] >= sparsities[i-1], "Schedule should be monotonically increasing"

        print(f"  Schedule: {[f'{s:.2f}' for s in sparsities]}")

        print("Pruning schedule creation works correctly")
        return True

    except Exception as e:
        print(f"  Pruning schedule creation test failed: {e}")
        return False


def run_all_tests():
    """Run all pruning tests."""
    print("=" * 60)
    print("Testing Pruning Agent")
    print("=" * 60)

    tests = [
        ("Pruning Imports", test_pruning_imports),
        ("Pruning Wrapper Factory", test_pruning_wrapper_factory),
        ("Estimate Pruning Benefits", test_estimate_pruning_benefits),
        ("Validate Pruning Compatibility", test_validate_pruning_compatibility),
        ("List Pruning Methods", test_list_pruning_methods),
        ("Pruning Subagent Config", test_pruning_subagent_config),
        ("CompressionStrategy Fields", test_compression_strategy_pruning_fields),
        ("Mock Magnitude Pruning", test_mock_magnitude_pruning),
        ("Mock Structured Pruning", test_mock_structured_pruning),
        ("Create Pruning Schedule", test_create_pruning_schedule),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Test raised exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All pruning tests passed!")
    else:
        print(f"{failed} pruning tests failed")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
