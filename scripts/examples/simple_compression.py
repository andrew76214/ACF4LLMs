#!/usr/bin/env python3
"""Simple example of using the Agentic Compression Framework."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.coordinator.coordinator import CompressionCoordinator
from src.coordinator.spec_inference import infer_spec, format_spec_summary


def main():
    """Run a simple compression optimization example."""

    # Define your model and dataset
    model_name = "gpt2"  # Start with a small model
    dataset = "commonsenseqa"  # Common sense reasoning benchmark

    print("=" * 60)
    print("Agentic Compression Framework - Simple Example")
    print("=" * 60)
    print()

    # Step 1: Infer model specification
    print("Step 1: Inferring model specification...")
    spec = infer_spec(model_name, dataset)
    print(format_spec_summary(spec))
    print()

    # Step 2: Create coordinator
    print("Step 2: Creating compression coordinator...")
    coordinator = CompressionCoordinator(
        model_name=model_name,
        dataset=dataset,
        max_episodes=3,  # Run 3 compression attempts
        use_mock=True,    # Use mock for this example
        experiment_name="simple_example"
    )
    print(f"Experiment directory: {coordinator.experiment_dir}")
    print()

    # Step 3: Run optimization
    print("Step 3: Running compression optimization...")
    print("-" * 40)
    results = coordinator.run(interactive=False)
    print("-" * 40)
    print()

    # Step 4: Display results
    print("Step 4: Results Summary")
    print("=" * 40)

    if results["frontier_summary"]["num_solutions"] > 0:
        print(f"Found {results['frontier_summary']['num_solutions']} Pareto optimal solutions")
        print()

        # Show best solutions
        if results["best_solutions"]["accuracy"]:
            acc = results["best_solutions"]["accuracy"]
            print(f"Best Accuracy: {acc['accuracy']:.3f} (compression: {acc['compression_ratio']:.1f}x)")

        if results["best_solutions"]["latency"]:
            lat = results["best_solutions"]["latency"]
            print(f"Best Latency: {lat['latency_ms']:.1f} ms (compression: {lat['compression_ratio']:.1f}x)")

        if results["best_solutions"]["size"]:
            size = results["best_solutions"]["size"]
            print(f"Smallest Size: {size['model_size_gb']:.1f} GB (compression: {size['compression_ratio']:.1f}x)")

    print()
    print(f"Full results saved to: {results['experiment_dir']}")

    if results.get("visualization"):
        print(f"Visualization available at: {results['visualization']}")

    print()
    print("✅ Example completed successfully!")


if __name__ == "__main__":
    main()