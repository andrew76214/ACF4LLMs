"""Standalone runner to evaluate a baseline LLM on key benchmarks with energy/CO2 estimates.

Usage examples:
  python run_manual_eval.py --model gpt2 --device cpu --proxy
  python run_manual_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --device cuda
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.benchmark_runner import BenchmarkRunner
from src.agents.evaluation_agent import estimate_energy_consumption


DEFAULT_BENCHMARKS = [
    "gsm8k",
    "commonsenseqa",
    "truthfulqa",
    "humaneval",
    "bigbench_hard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual baseline evaluation on core benchmarks with energy/CO2 estimation."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local checkpoint path (e.g., 'gpt2' or './checkpoints/model').",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks to run. Defaults to the core five.",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use proxy evaluation for speed (subset of data).",
    )
    parser.add_argument(
        "--proxy-samples",
        type=int,
        default=200,
        help="Number of samples for proxy evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--energy-inferences",
        type=int,
        default=1000,
        help="Number of inferences used for energy/CO2 estimation.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments/manual_experiment",
        help="Where to save results JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(device=args.device)
    results = runner.run_full_evaluation(
        checkpoint_path=args.model,
        benchmarks=args.benchmarks,
        use_proxy=args.proxy,
        proxy_samples=args.proxy_samples,
        batch_size=args.batch_size,
        save_results=False,  # we handle persistence below
    )

    energy = estimate_energy_consumption(
        checkpoint_path=args.model,
        num_inferences=args.energy_inferences,
        device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"baseline_eval_results_{timestamp}.json"
    energy_path = output_dir / f"baseline_energy_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(energy_path, "w") as f:
        json.dump(energy, f, indent=2)

    print(f"Saved evaluation to {results_path}")
    print(f"Saved energy/CO2 estimate to {energy_path}")


if __name__ == "__main__":
    main()
