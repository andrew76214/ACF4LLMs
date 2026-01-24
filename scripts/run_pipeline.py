#!/usr/bin/env python3
"""CLI interface for running the Agentic Compression Framework with LangGraph."""

# Suppress third-party deprecation warnings before imports
import warnings
import os

# Suppress all deprecation warnings from problematic third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*device.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Flash Attention.*non-deterministic.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*shouldn't move a model.*")
warnings.filterwarnings("ignore", message=".*unrecognized keys.*")
warnings.filterwarnings("ignore", message=".*this API is deprecated.*")

# Also suppress via environment variable for subprocesses
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

import click
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.coordinator.langgraph_coordinator import LangGraphCoordinator
from src.coordinator.spec_inference import infer_spec, format_spec_summary
from src.common.schemas import Benchmark


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="HuggingFace model name or path (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(["gsm8k", "commonsenseqa", "truthfulqa", "humaneval", "bigbench_hard"], case_sensitive=False),
    help="Target dataset/benchmark for optimization",
)
@click.option(
    "--episodes",
    "-e",
    default=3,
    type=int,
    help="Maximum number of compression episodes (default: 3)",
)
@click.option(
    "--budget",
    "-b",
    default=2.0,
    type=float,
    help="Time budget in hours (default: 2.0)",
)
@click.option(
    "--experiment-name",
    "-n",
    default=None,
    help="Name for this experiment (auto-generated if not provided)",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Run in interactive mode with progress updates",
)
@click.option(
    "--show-spec",
    is_flag=True,
    help="Show inferred specification and exit",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/experiments",
    help="Output directory for experiment results (default: data/experiments)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML)",
)
def main(
    model: str,
    dataset: str,
    episodes: int,
    budget: float,
    experiment_name: Optional[str],
    interactive: bool,
    show_spec: bool,
    output_dir: str,
    config: Optional[str],
):
    """Agentic Compression Framework - LLM-Driven Model Compression Optimization

    This tool uses GPT-4o to autonomously find the best compression strategies
    for your model using multi-objective optimization across accuracy, latency,
    memory, and model size.

    Requires OPENAI_API_KEY environment variable to be set.

    Examples:

    \b
    # Basic usage with minimal input
    python run_pipeline.py --model gpt2 --dataset gsm8k

    \b
    # Run more episodes for better optimization
    python run_pipeline.py -m meta-llama/Meta-Llama-3-8B-Instruct -d commonsenseqa -e 10

    \b
    # Interactive mode with custom experiment name
    python run_pipeline.py -m mistralai/Mistral-7B-v0.1 -d humaneval -i -n "mistral_code_opt"

    \b
    # Just show the inferred specification
    python run_pipeline.py -m Qwen/Qwen2-7B -d truthfulqa --show-spec
    """
    # Print banner
    click.echo("=" * 60)
    click.echo("🚀 Agentic Compression Framework")
    click.echo("=" * 60)
    click.echo()

    # Load config if provided
    config_dict = {}
    if config:
        with open(config, 'r') as f:
            if config.endswith('.json'):
                config_dict = json.load(f)
            elif config.endswith('.yaml') or config.endswith('.yml'):
                import yaml
                config_dict = yaml.safe_load(f)
        click.echo(f"📋 Loaded configuration from {config}")

    # Show specification if requested
    if show_spec:
        click.echo("📊 Inferring model specification...")
        spec = infer_spec(model, dataset)
        click.echo()
        click.echo(format_spec_summary(spec))
        click.echo()

        # Save spec to file
        spec_file = Path(output_dir) / "inferred_spec.json"
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_file, 'w') as f:
            json.dump(spec.model_dump(), f, indent=2, default=str)
        click.echo(f"💾 Specification saved to {spec_file}")
        return

    # Validate inputs
    if episodes < 1:
        click.echo("❌ Error: Episodes must be at least 1", err=True)
        sys.exit(1)

    if budget <= 0:
        click.echo("❌ Error: Budget must be positive", err=True)
        sys.exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("❌ Error: OPENAI_API_KEY environment variable is required", err=True)
        click.echo("  Set it with: export OPENAI_API_KEY=sk-...", err=True)
        sys.exit(1)

    # Show configuration
    click.echo("📝 Configuration:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  Episodes: {episodes}")
    click.echo(f"  Budget: {budget:.1f} hours")
    click.echo(f"  Mode: {'Interactive' if interactive else 'Batch'}")
    click.echo(f"  Coordinator: LangGraph (GPT-4o)")
    click.echo()

    # Create coordinator
    try:
        click.echo("🔧 Initializing LangGraph coordinator...")
        coordinator = LangGraphCoordinator(
            model_name=model,
            dataset=dataset,
            budget_hours=budget,
            max_episodes=episodes,
            experiment_name=experiment_name,
        )
        click.echo(f"✅ Coordinator initialized")
        click.echo(f"📁 Experiment directory: {coordinator.experiment_dir}")
        click.echo()

    except Exception as e:
        click.echo(f"❌ Error initializing coordinator: {e}", err=True)
        sys.exit(1)

    # Run optimization
    try:
        click.echo("🏃 Starting compression optimization...")
        click.echo("=" * 60)

        results = coordinator.run(interactive=interactive)

        click.echo("=" * 60)
        click.echo()

    except KeyboardInterrupt:
        click.echo("\n⚠️  Optimization interrupted by user")
        results = coordinator._compile_results()

    except Exception as e:
        click.echo(f"\n❌ Error during optimization: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display results
    click.echo("📊 Optimization Results:")
    click.echo("-" * 40)

    if results["frontier_summary"]["num_solutions"] > 0:
        click.echo(f"✨ Pareto Solutions: {results['frontier_summary']['num_solutions']}")
        click.echo(f"📈 Episodes Completed: {results['episodes_completed']}")
        click.echo()

        # Show best solutions
        if results["best_solutions"]["accuracy"]:
            acc_sol = results["best_solutions"]["accuracy"]
            click.echo(f"🎯 Best Accuracy:")
            click.echo(f"   Accuracy: {acc_sol['accuracy']:.3f}")
            click.echo(f"   Compression: {acc_sol['compression_ratio']:.1f}x")

        if results["best_solutions"]["latency"]:
            lat_sol = results["best_solutions"]["latency"]
            click.echo(f"⚡ Best Latency:")
            click.echo(f"   Latency: {lat_sol['latency_ms']:.1f} ms")
            click.echo(f"   Compression: {lat_sol['compression_ratio']:.1f}x")

        if results["best_solutions"]["size"]:
            size_sol = results["best_solutions"]["size"]
            click.echo(f"💾 Best Size:")
            click.echo(f"   Model Size: {size_sol['model_size_gb']:.1f} GB")
            click.echo(f"   Compression: {size_sol['compression_ratio']:.1f}x")

        if results["best_solutions"]["balanced"]:
            bal_sol = results["best_solutions"]["balanced"]
            click.echo(f"⚖️  Balanced Solution:")
            click.echo(f"   Accuracy: {bal_sol['accuracy']:.3f}")
            click.echo(f"   Latency: {bal_sol['latency_ms']:.1f} ms")
            click.echo(f"   Size: {bal_sol['model_size_gb']:.1f} GB")

    else:
        click.echo("⚠️  No Pareto solutions found")

    click.echo()
    click.echo("-" * 40)
    click.echo(f"📁 Results saved to: {results['experiment_dir']}")

    if results.get("visualization"):
        click.echo(f"📊 Visualization: {results['visualization']}")

    click.echo()
    click.echo("✅ Optimization complete!")


@click.command(name="analyze")
@click.argument("experiment_dir", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["json", "summary", "detailed"]), default="summary")
def analyze(experiment_dir: str, format: str):
    """Analyze results from a previous experiment.

    EXPERIMENT_DIR: Path to the experiment directory
    """
    experiment_path = Path(experiment_dir)

    # Load results
    results_file = experiment_path / "final_results.json"
    if not results_file.exists():
        click.echo(f"❌ No results found in {experiment_dir}", err=True)
        sys.exit(1)

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load Pareto frontier
    frontier_file = experiment_path / "pareto_frontier.json"
    if frontier_file.exists():
        with open(frontier_file, 'r') as f:
            frontier_data = json.load(f)
    else:
        frontier_data = None

    if format == "json":
        # Output raw JSON
        click.echo(json.dumps(results, indent=2))

    elif format == "summary":
        # Show summary
        click.echo("=" * 60)
        click.echo(f"📊 Experiment: {results['experiment_name']}")
        click.echo("=" * 60)
        click.echo(f"Model: {results['model']}")
        click.echo(f"Dataset: {results['dataset']}")
        click.echo(f"Episodes: {results['episodes_completed']}")

        if frontier_data:
            click.echo(f"Pareto Solutions: {len(frontier_data['frontier'])}")
            click.echo(f"Total Evaluated: {len(frontier_data['history'])}")

        click.echo()
        click.echo("Best Solutions:")
        for objective, solution in results["best_solutions"].items():
            if solution:
                click.echo(f"  {objective.capitalize()}: Strategy {solution.get('strategy_id', 'N/A')}")

    elif format == "detailed":
        # Show detailed analysis
        click.echo("=" * 60)
        click.echo(f"📊 Detailed Analysis: {results['experiment_name']}")
        click.echo("=" * 60)

        # Show all Pareto solutions
        if frontier_data:
            click.echo("\nPareto Frontier Solutions:")
            click.echo("-" * 40)

            for i, sol in enumerate(frontier_data['frontier'], 1):
                result = sol['result']
                strategy = sol['strategy']

                click.echo(f"\nSolution {i} (Strategy {strategy['strategy_id']}):")
                click.echo(f"  Method: {strategy.get('quantization_method', 'N/A')}")
                click.echo(f"  Bits: {strategy.get('quantization_bits', 'N/A')}")
                click.echo(f"  Accuracy: {result['accuracy']:.3f}")
                click.echo(f"  Latency: {result['latency_ms']:.1f} ms")
                click.echo(f"  Memory: {result['memory_gb']:.1f} GB")
                click.echo(f"  Size: {result['model_size_gb']:.1f} GB")
                click.echo(f"  Compression: {result['compression_ratio']:.1f}x")

                if result.get('benchmark_scores'):
                    click.echo("  Benchmark Scores:")
                    for benchmark, score in result['benchmark_scores'].items():
                        click.echo(f"    {benchmark}: {score:.3f}")


@click.group()
def cli():
    """Agentic Compression Framework CLI"""
    pass


# Add commands to group
cli.add_command(main, name="run")
cli.add_command(analyze)


if __name__ == "__main__":
    # If called directly, run the main command
    main()