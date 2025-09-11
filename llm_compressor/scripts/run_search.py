#!/usr/bin/env python3
"""
Main script for running LLM compression optimization search.

Usage:
    python scripts/run_search.py --config configs/default.yaml
    python scripts/run_search.py --config configs/default.yaml --recipes baseline
    python scripts/run_search.py --help
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import time
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_compressor.core.orchestrator import Orchestrator
from llm_compressor.core.pareto import ParetoFrontier


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file."""
    
    required_sections = ["model", "hardware", "constraints", "agents"]
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required section '{section}' in config")
            return False
    
    # Validate model config
    model_config = config["model"]
    required_model_fields = ["base_model", "sequence_length"]
    
    for field in required_model_fields:
        if field not in model_config:
            logging.error(f"Missing required field '{field}' in model config")
            return False
    
    # Validate constraints
    constraints = config["constraints"]
    if constraints.get("max_accuracy_drop", 0) > 0.1:  # 10%
        logging.warning("Large accuracy drop tolerance (>10%) specified")
    
    if constraints.get("p95_latency_ms", 0) <= 0:
        logging.error("Invalid P95 latency constraint")
        return False
    
    return True


def load_baseline_recipes(recipes_path: str) -> Dict[str, Any]:
    """Load baseline recipes from YAML file."""
    
    recipes_file = Path(recipes_path)
    if not recipes_file.exists():
        logging.error(f"Recipes file not found: {recipes_path}")
        return {}
    
    try:
        with open(recipes_file, 'r') as f:
            # Load all YAML documents (recipes separated by ---)
            recipes = {}
            for doc in yaml.safe_load_all(f):
                if doc and isinstance(doc, dict):
                    # Skip metadata
                    if "recipe_metadata" in doc:
                        continue
                    
                    # Each recipe should have a top-level key
                    for recipe_name, recipe_config in doc.items():
                        if isinstance(recipe_config, dict) and recipe_config.get("enabled", True):
                            recipe_config["id"] = f"baseline_{recipe_name}"
                            recipe_config["type"] = "baseline"
                            recipes[recipe_name] = recipe_config
            
            logging.info(f"Loaded {len(recipes)} baseline recipes")
            return recipes
            
    except yaml.YAMLError as e:
        logging.error(f"Error parsing recipes file: {e}")
        return {}


def run_baseline_recipes(orchestrator: Orchestrator, recipes: Dict[str, Any]) -> Dict[str, Any]:
    """Run baseline recipes only."""
    
    logging.info("Running baseline recipes")
    
    # Convert recipes to list format expected by orchestrator
    recipe_list = []
    for recipe_name, recipe_config in recipes.items():
        # Ensure pipeline is set
        if "pipeline" not in recipe_config:
            recipe_config["pipeline"] = ["quantization", "kv_longcontext", "perf_carbon", "eval_safety"]
        
        recipe_list.append(recipe_config)
    
    # Execute recipes
    results = orchestrator.execute_recipes_parallel(recipe_list)
    
    # Analyze results
    pareto = orchestrator.pareto
    pareto_candidates = pareto.analyze(results)
    
    # Generate summary
    summary = {
        "execution_type": "baseline_recipes",
        "total_recipes": len(recipe_list),
        "successful_recipes": len([r for r in results if r.get("success", False)]),
        "pareto_candidates": len(pareto_candidates),
        "best_candidates": pareto_candidates[:3],
        "all_results": results
    }
    
    return summary


def run_full_search(orchestrator: Orchestrator) -> Dict[str, Any]:
    """Run full optimization search."""
    
    logging.info("Running full optimization search")
    return orchestrator.run_optimization()


def generate_reports(summary: Dict[str, Any], output_dir: str, config: Dict[str, Any]):
    """Generate comprehensive reports."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create Pareto analysis
    pareto_config = config.get("pareto", {})
    pareto = ParetoFrontier(pareto_config)
    
    results = summary.get("all_results", [])
    if results:
        # Generate visualizations
        plots = pareto.visualize_frontier(results, str(output_path))
        logging.info(f"Generated {len(plots)} visualization plots")
        
        # Export Pareto summary
        pareto_summary_path = pareto.export_summary(results, str(output_path / "pareto_summary.json"))
        logging.info(f"Exported Pareto analysis: {pareto_summary_path}")
    
    # Generate execution summary
    summary_path = output_path / "execution_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Generated execution summary: {summary_path}")
    
    # Generate markdown report
    markdown_report = generate_markdown_report(summary, config)
    markdown_path = output_path / "summary.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    logging.info(f"Generated markdown report: {markdown_path}")
    
    return {
        "output_dir": str(output_path),
        "plots": plots if 'plots' in locals() else {},
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path)
    }


def generate_markdown_report(summary: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate markdown summary report."""
    
    model_name = config.get("model", {}).get("base_model", "Unknown")
    execution_type = summary.get("execution_type", "optimization")
    
    report = f"""# LLM Compression Results

## Configuration
- **Model**: {model_name}
- **Execution Type**: {execution_type.replace('_', ' ').title()}
- **Sequence Length**: {config.get('model', {}).get('sequence_length', 'Unknown')}
- **Hardware**: {config.get('hardware', {}).get('gpu', 'Unknown')}

## Results Summary
- **Total Recipes**: {summary.get('total_recipes', 0)}
- **Successful Recipes**: {summary.get('successful_recipes', 0)}
- **Success Rate**: {(summary.get('successful_recipes', 0) / max(summary.get('total_recipes', 1), 1)) * 100:.1f}%
- **Pareto Candidates**: {summary.get('pareto_candidates', 0)}

"""
    
    # Add top candidates
    best_candidates = summary.get("best_candidates", [])
    if best_candidates:
        report += "## Top Pareto Candidates\n\n"
        
        for i, candidate in enumerate(best_candidates[:3], 1):
            metrics = candidate.get("metrics", {})
            recipe_id = candidate.get("recipe_id", f"candidate_{i}")
            
            report += f"### {i}. {recipe_id}\n"
            report += f"- **Composite Score**: {candidate.get('composite_score', 0):.3f}\n"
            
            if "accuracy" in metrics:
                report += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
            if "latency_mean_ms" in metrics:
                report += f"- **Latency (P50)**: {metrics['latency_mean_ms']:.1f}ms\n"
            if "vram_peak_gb" in metrics:
                report += f"- **VRAM Peak**: {metrics['vram_peak_gb']:.1f}GB\n"
            if "energy_1hour_kwh" in metrics:
                report += f"- **Energy (1h)**: {metrics['energy_1hour_kwh']:.3f}kWh\n"
            if "co2e_1hour_kg" in metrics:
                report += f"- **CO2e (1h)**: {metrics['co2e_1hour_kg']:.3f}kg\n"
            
            report += "\n"
    
    # Add constraints check
    constraints = config.get("constraints", {})
    if best_candidates and constraints:
        report += "## Constraint Compliance\n\n"
        
        best = best_candidates[0]
        metrics = best.get("metrics", {})
        
        # Check accuracy constraint
        max_acc_drop = constraints.get("max_accuracy_drop", 0.01)
        acc_drop = metrics.get("estimated_accuracy_drop", 0)
        acc_status = "✅ PASS" if acc_drop <= max_acc_drop else "❌ FAIL"
        report += f"- **Accuracy Drop**: {acc_drop*100:.1f}% (limit: {max_acc_drop*100:.1f}%) {acc_status}\n"
        
        # Check latency constraint
        max_latency = constraints.get("p95_latency_ms", 150)
        latency = metrics.get("latency_p95_ms", 0)
        latency_status = "✅ PASS" if latency <= max_latency else "❌ FAIL"
        report += f"- **P95 Latency**: {latency:.1f}ms (limit: {max_latency}ms) {latency_status}\n"
        
        # Check VRAM constraint
        max_vram = constraints.get("max_vram_gb", 80)
        vram = metrics.get("vram_peak_gb", 0)
        vram_status = "✅ PASS" if vram <= max_vram else "❌ FAIL"
        report += f"- **VRAM Usage**: {vram:.1f}GB (limit: {max_vram}GB) {vram_status}\n"
        
        report += "\n"
    
    # Add execution info
    if "execution_time" in summary:
        exec_time = summary["execution_time"]
        report += f"## Execution Details\n"
        report += f"- **Total Time**: {exec_time/60:.1f} minutes\n"
        report += f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Visualization Files\n"
    report += "- `pareto_accuracy_latency.html` - Accuracy vs Latency trade-off\n"
    report += "- `pareto_accuracy_vram.html` - Accuracy vs VRAM trade-off\n"
    report += "- `pareto_3d.html` - 3D Pareto frontier\n"
    report += "- `pareto_parallel.html` - Parallel coordinates view\n"
    report += "- `pareto_radar.html` - Top candidates comparison\n\n"
    
    report += "---\n"
    report += "*Generated by LLM Compressor - Multi-agent optimization system*\n"
    
    return report


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Run LLM compression optimization search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full optimization search
    python scripts/run_search.py --config configs/default.yaml
    
    # Run only baseline recipes
    python scripts/run_search.py --config configs/default.yaml --recipes baseline
    
    # Custom output directory
    python scripts/run_search.py --config configs/default.yaml --output reports/my_experiment
    
    # Enable debug logging
    python scripts/run_search.py --config configs/default.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--recipes", "-r",
        type=str,
        choices=["baseline", "search", "full"],
        default="full",
        help="Recipe execution mode (default: full)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reports",
        help="Output directory for results (default: reports)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: stdout only)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running optimization"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logging.info("Starting LLM Compression Optimization")
    logging.info(f"Configuration file: {args.config}")
    logging.info(f"Recipe mode: {args.recipes}")
    logging.info(f"Output directory: {args.output}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    
    # Validate configuration
    if not validate_config(config):
        logging.error("Configuration validation failed")
        sys.exit(1)
    
    logging.info("Configuration validation passed")
    
    # Dry run check
    if args.dry_run:
        logging.info("Dry run completed successfully")
        return
    
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator(args.config)
        
        # Execute based on recipe mode
        if args.recipes == "baseline":
            # Load and run baseline recipes
            recipes_path = config_path.parent / "recipes_baseline.yaml"
            baseline_recipes = load_baseline_recipes(str(recipes_path))
            
            if not baseline_recipes:
                logging.error("No baseline recipes loaded")
                sys.exit(1)
            
            summary = run_baseline_recipes(orchestrator, baseline_recipes)
            
        elif args.recipes == "search":
            # Run search-generated recipes only
            logging.info("Running search-generated recipes only")
            summary = run_full_search(orchestrator)
            
        else:  # full
            # Run complete optimization
            summary = run_full_search(orchestrator)
        
        # Generate reports
        report_info = generate_reports(summary, args.output, config)
        
        # Print summary
        logging.info("=" * 60)
        logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        logging.info(f"Total recipes executed: {summary.get('total_recipes', 0)}")
        logging.info(f"Successful recipes: {summary.get('successful_recipes', 0)}")
        logging.info(f"Pareto candidates found: {summary.get('pareto_candidates', 0)}")
        logging.info(f"Results saved to: {report_info['output_dir']}")
        
        if summary.get("best_candidates"):
            best = summary["best_candidates"][0]
            logging.info(f"Best candidate: {best.get('recipe_id', 'unknown')} "
                        f"(score: {best.get('composite_score', 0):.3f})")
        
        logging.info("=" * 60)
        
    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()