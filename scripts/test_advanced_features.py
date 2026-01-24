#!/usr/bin/env python3
"""Test script demonstrating Phase 2 and 3 advanced features."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.coordinator.spec_inference import infer_spec
from src.coordinator.strategy_generator import create_strategy_generator
from src.reward.reward_function import MultiObjectiveReward, create_balanced_reward
from src.agents.search_agent import bayesian_optimization_search, evolutionary_search
from src.monitoring.mlflow_tracker import create_experiment_tracker
from src.agents.resource_monitor_agent import check_gpu_availability


def test_advanced_features():
    """Test Phase 2 and 3 features."""

    print("=" * 60)
    print("Testing Advanced Features (Phase 2 & 3)")
    print("=" * 60)
    print()

    # 1. Test Spec Inference
    print("1. Model Specification Inference")
    print("-" * 40)
    spec = infer_spec("meta-llama/Meta-Llama-3-8B-Instruct", "gsm8k")
    print(f"Model: {spec.model_name}")
    print(f"Size: {spec.model_size_gb} GB")
    print(f"Recommended methods: {[m.value for m in spec.preferred_methods]}")
    print(f"Primary objective: {spec.primary_objective}")
    print()

    # 2. Test Strategy Generation
    print("2. Adaptive Strategy Generation")
    print("-" * 40)
    generator = create_strategy_generator(spec.model_dump(), "balanced")

    # Generate strategies for different episodes
    strategies = []
    for episode in [1, 4, 8]:
        strategy = generator.generate_next_strategy(episode, strategies)
        print(f"Episode {episode}: {strategy.methods[0].value if strategy.methods else 'None'}")
        if hasattr(strategy, 'quantization_bits'):
            print(f"  Bits: {strategy.quantization_bits}")
        strategies.append(strategy)
    print()

    # 3. Test Multi-Objective Reward
    print("3. Multi-Objective Reward Function")
    print("-" * 40)
    reward_func = create_balanced_reward()

    # Test different scenarios
    scenarios = [
        {"accuracy": 0.95, "latency_ms": 50, "memory_gb": 4, "size_gb": 2},
        {"accuracy": 0.85, "latency_ms": 100, "memory_gb": 8, "size_gb": 4},
        {"accuracy": 0.75, "latency_ms": 25, "memory_gb": 2, "size_gb": 1},
    ]

    for i, metrics in enumerate(scenarios, 1):
        reward, scores = reward_func.calculate_reward(**metrics)
        print(f"Scenario {i}: Reward={reward:.2f}, Acc={metrics['accuracy']}, Lat={metrics['latency_ms']}ms")
    print()

    # 4. Test Search Algorithms
    print("4. Advanced Search Algorithms")
    print("-" * 40)

    # Bayesian optimization
    search_space = {
        "quantization_bits": {"type": "choice", "values": [4, 8]},
        "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-3},
    }

    bayes_result = bayesian_optimization_search(
        objective_function="accuracy",
        search_space=search_space,
        num_iterations=3,
        history=[]
    )
    print(f"Bayesian next params: {bayes_result['next_parameters']}")

    # Evolutionary search
    evo_result = evolutionary_search(
        population_size=5,
        num_generations=2,
        mutation_rate=0.1,
        search_space=search_space
    )
    print(f"Evolutionary best: {evo_result['best_individual']}")
    print()

    # 5. Test Resource Monitoring
    print("5. Resource Monitoring")
    print("-" * 40)
    gpu_info = check_gpu_availability.invoke({})
    print(f"CUDA available: {gpu_info['cuda_available']}")
    print(f"CPU cores: {gpu_info['cpu_info']['cpu_count']}")
    print(f"RAM: {gpu_info['cpu_info']['ram_total_gb']:.1f} GB")
    print()

    # 6. Test MLflow Tracking (Mock)
    print("6. MLflow Experiment Tracking")
    print("-" * 40)
    tracker = create_experiment_tracker("test_experiment", use_mlflow=False)  # Use mock

    # Simulate episode tracking
    run_id = tracker.start_episode(1, {"strategy_id": "test123"}, spec.model_dump())
    print(f"Started run: {run_id}")

    tracker.log_compression_results(
        checkpoint_path="/test/checkpoint",
        compression_ratio=4.0,
        quantization_time=120.0,
        method="autoround"
    )

    tracker.log_evaluation_results(
        benchmark_scores={"gsm8k": 0.85, "commonsenseqa": 0.90},
        average_accuracy=0.875,
        latency_ms=50,
        memory_gb=4,
        throughput=100
    )

    tracker.end_episode("SUCCESS")
    print("Episode tracking complete")
    print()

    # 7. Test Data Management
    print("7. Data Management")
    print("-" * 40)
    from src.agents.data_manager_agent import estimate_dataset_statistics

    stats = estimate_dataset_statistics.invoke({
        "dataset_name": "wikitext",
        "num_samples": 100
    })
    print(f"Dataset samples: {stats.get('num_samples_analyzed', 0)}")
    print(f"Avg length: {stats.get('avg_length', 0):.0f} chars")
    print()

    # 8. Test Distillation Planning
    print("8. Distillation Planning")
    print("-" * 40)
    from src.agents.distillation_agent import create_student_architecture

    student_config = create_student_architecture.invoke({
        "teacher_checkpoint": "meta-llama/Meta-Llama-3-8B-Instruct",
        "compression_factor": 0.5,
        "architecture_type": "depth"
    })
    print(f"Student compression: {student_config['compression_factor']}")
    print(f"Parameter reduction: {student_config.get('parameter_reduction', 0):.0%}")
    print()

    print("=" * 60)
    print("✅ All Phase 2 & 3 Features Working!")
    print("=" * 60)


if __name__ == "__main__":
    test_advanced_features()