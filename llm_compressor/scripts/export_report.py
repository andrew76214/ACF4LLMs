#!/usr/bin/env python3
"""
Export and analyze results from LLM compression experiments.

Usage:
    python scripts/export_report.py --db experiments.db --output final_report
    python scripts/export_report.py --input reports/execution_summary.json --format html
    python scripts/export_report.py --help
"""

import argparse
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_compressor.core.registry import Registry
from llm_compressor.core.pareto import ParetoFrontier


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_results_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from database."""
    
    if not Path(db_path).exists():
        logging.error(f"Database file not found: {db_path}")
        return []
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, recipe_data, metrics_data, status, created_at
                FROM experiments
                WHERE status = 'completed'
                ORDER BY created_at DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                try:
                    result = {
                        "recipe_id": row[0],
                        "recipe": json.loads(row[1]),
                        "metrics": json.loads(row[2]),
                        "status": row[3],
                        "created_at": row[4],
                        "success": True
                    }
                    results.append(result)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse experiment {row[0]}: {e}")
                    continue
            
            logging.info(f"Loaded {len(results)} experiments from database")
            return results
            
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []


def load_results_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON summary file."""
    
    json_file = Path(json_path)
    if not json_file.exists():
        logging.error(f"JSON file not found: {json_path}")
        return []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract results from summary
        results = data.get("all_results", [])
        
        logging.info(f"Loaded {len(results)} results from JSON")
        return results
        
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error loading JSON file: {e}")
        return []


def generate_comprehensive_report(results: List[Dict[str, Any]], 
                                 output_dir: str,
                                 format_type: str = "html") -> Dict[str, str]:
    """Generate comprehensive analysis report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Perform Pareto analysis
    pareto_config = {
        "objectives": ["accuracy", "latency", "vram", "energy", "co2e"],
        "maximize": ["accuracy"],
        "objective_weights": {
            "accuracy": 1.0,
            "latency": -0.8,
            "vram": -0.6,
            "energy": -0.5,
            "co2e": -0.3
        }
    }
    
    pareto = ParetoFrontier(pareto_config)
    pareto_candidates = pareto.analyze(results)
    
    # Generate visualizations
    plots = pareto.visualize_frontier(results, str(output_path))
    
    # Generate different format reports
    generated_files = {}
    
    if format_type in ["html", "all"]:
        html_file = generate_html_report(results, pareto_candidates, output_path)
        generated_files["html"] = html_file
    
    if format_type in ["csv", "all"]:
        csv_file = generate_csv_report(results, pareto_candidates, output_path)
        generated_files["csv"] = csv_file
    
    if format_type in ["json", "all"]:
        json_file = generate_json_report(results, pareto_candidates, output_path)
        generated_files["json"] = json_file
    
    if format_type in ["markdown", "all"]:
        md_file = generate_markdown_report(results, pareto_candidates, output_path)
        generated_files["markdown"] = md_file
    
    # Generate summary statistics
    stats_file = generate_statistics_report(results, pareto_candidates, output_path)
    generated_files["statistics"] = stats_file
    
    generated_files.update(plots)
    
    return generated_files


def generate_html_report(results: List[Dict[str, Any]], 
                        pareto_candidates: List[Dict[str, Any]], 
                        output_path: Path) -> str:
    """Generate interactive HTML report."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Compression Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc; }}
        .candidate-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .candidate-table th, .candidate-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .candidate-table th {{ background-color: #f2f2f2; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        .plots-section {{ margin: 30px 0; }}
        .plot-link {{ display: inline-block; margin: 10px; padding: 10px 20px; background-color: #007acc; color: white; text-decoration: none; border-radius: 5px; }}
        .plot-link:hover {{ background-color: #005a9c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Compression Analysis Report</h1>
        <p><strong>Generated:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Experiments:</strong> {len(results)}</p>
        <p><strong>Pareto Candidates:</strong> {len(pareto_candidates)}</p>
    </div>

    <h2>Summary Statistics</h2>
    <div class="metric-grid">
"""
    
    # Calculate summary statistics
    successful_results = [r for r in results if r.get("success", False)]
    
    if successful_results:
        # Accuracy statistics
        accuracies = [r["metrics"].get("accuracy", 0) for r in successful_results if "accuracy" in r.get("metrics", {})]
        if accuracies:
            html_content += f"""
        <div class="metric-card">
            <h4>Accuracy</h4>
            <p><strong>Best:</strong> {max(accuracies):.3f}</p>
            <p><strong>Mean:</strong> {sum(accuracies)/len(accuracies):.3f}</p>
            <p><strong>Range:</strong> {min(accuracies):.3f} - {max(accuracies):.3f}</p>
        </div>
"""
        
        # Latency statistics
        latencies = [r["metrics"].get("latency_mean_ms", 0) for r in successful_results if "latency_mean_ms" in r.get("metrics", {})]
        if latencies:
            html_content += f"""
        <div class="metric-card">
            <h4>Latency (ms)</h4>
            <p><strong>Best:</strong> {min(latencies):.1f}</p>
            <p><strong>Mean:</strong> {sum(latencies)/len(latencies):.1f}</p>
            <p><strong>Range:</strong> {min(latencies):.1f} - {max(latencies):.1f}</p>
        </div>
"""
        
        # VRAM statistics
        vram_usage = [r["metrics"].get("vram_peak_gb", 0) for r in successful_results if "vram_peak_gb" in r.get("metrics", {})]
        if vram_usage:
            html_content += f"""
        <div class="metric-card">
            <h4>VRAM Usage (GB)</h4>
            <p><strong>Best:</strong> {min(vram_usage):.1f}</p>
            <p><strong>Mean:</strong> {sum(vram_usage)/len(vram_usage):.1f}</p>
            <p><strong>Range:</strong> {min(vram_usage):.1f} - {max(vram_usage):.1f}</p>
        </div>
"""
        
        # Energy statistics
        energy_usage = [r["metrics"].get("energy_1hour_kwh", 0) for r in successful_results if "energy_1hour_kwh" in r.get("metrics", {})]
        if energy_usage:
            html_content += f"""
        <div class="metric-card">
            <h4>Energy (kWh/hour)</h4>
            <p><strong>Best:</strong> {min(energy_usage):.3f}</p>
            <p><strong>Mean:</strong> {sum(energy_usage)/len(energy_usage):.3f}</p>
            <p><strong>Range:</strong> {min(energy_usage):.3f} - {max(energy_usage):.3f}</p>
        </div>
"""
    
    html_content += """
    </div>

    <h2>Top Pareto Candidates</h2>
    <table class="candidate-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Recipe ID</th>
                <th>Composite Score</th>
                <th>Accuracy</th>
                <th>Latency (ms)</th>
                <th>VRAM (GB)</th>
                <th>Energy (kWh/h)</th>
                <th>CO2e (kg/h)</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add top candidates
    for candidate in pareto_candidates[:10]:  # Top 10
        metrics = candidate.get("metrics", {})
        html_content += f"""
            <tr>
                <td>{candidate.get('pareto_rank', '-')}</td>
                <td>{candidate.get('recipe_id', 'Unknown')}</td>
                <td>{candidate.get('composite_score', 0):.3f}</td>
                <td>{metrics.get('accuracy', 0):.3f}</td>
                <td>{metrics.get('latency_mean_ms', 0):.1f}</td>
                <td>{metrics.get('vram_peak_gb', 0):.1f}</td>
                <td>{metrics.get('energy_1hour_kwh', 0):.3f}</td>
                <td>{metrics.get('co2e_1hour_kg', 0):.3f}</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>

    <div class="plots-section">
        <h2>Interactive Visualizations</h2>
        <p>Click on the links below to view interactive plots:</p>
        <a href="pareto_accuracy_latency.html" class="plot-link">Accuracy vs Latency</a>
        <a href="pareto_accuracy_vram.html" class="plot-link">Accuracy vs VRAM</a>
        <a href="pareto_3d.html" class="plot-link">3D Pareto Frontier</a>
        <a href="pareto_parallel.html" class="plot-link">Parallel Coordinates</a>
        <a href="pareto_radar.html" class="plot-link">Top Candidates Radar</a>
    </div>

    <h2>All Experiments</h2>
    <table class="candidate-table">
        <thead>
            <tr>
                <th>Recipe ID</th>
                <th>Status</th>
                <th>Composite Score</th>
                <th>Optimization Type</th>
                <th>Created</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add all experiments
    for result in results:
        status_class = "success" if result.get("success", False) else "error"
        recipe_config = result.get("recipe", {})
        
        # Determine optimization type
        opt_type = []
        if recipe_config.get("quantization", {}).get("enabled", False):
            opt_type.append("Quantization")
        if recipe_config.get("kv_longcontext", {}).get("enabled", False):
            opt_type.append("KV-Opt")
        if recipe_config.get("pruning_sparsity", {}).get("enabled", False):
            opt_type.append("Pruning")
        if recipe_config.get("distillation", {}).get("enabled", False):
            opt_type.append("Distillation")
        
        opt_type_str = "+".join(opt_type) if opt_type else "None"
        
        html_content += f"""
            <tr>
                <td>{result.get('recipe_id', 'Unknown')}</td>
                <td><span class="{status_class}">{'SUCCESS' if result.get('success', False) else 'FAILED'}</span></td>
                <td>{result.get('composite_score', 0):.3f}</td>
                <td>{opt_type_str}</td>
                <td>{result.get('created_at', 'Unknown')}</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>

    <div style="margin-top: 50px; border-top: 1px solid #ccc; padding-top: 20px; text-align: center; color: #666;">
        <p><em>Generated by LLM Compressor - Multi-agent optimization system</em></p>
    </div>
</body>
</html>
"""
    
    html_file = output_path / "comprehensive_report.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Generated HTML report: {html_file}")
    return str(html_file)


def generate_csv_report(results: List[Dict[str, Any]], 
                       pareto_candidates: List[Dict[str, Any]], 
                       output_path: Path) -> str:
    """Generate CSV export of results."""
    
    import csv
    
    csv_file = output_path / "results_export.csv"
    
    # Collect all possible metric keys
    all_metrics = set()
    for result in results:
        all_metrics.update(result.get("metrics", {}).keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # Define CSV columns
    columns = ["recipe_id", "success", "pareto_rank", "composite_score"] + all_metrics
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            row = {
                "recipe_id": result.get("recipe_id", ""),
                "success": result.get("success", False),
                "pareto_rank": result.get("pareto_rank", ""),
                "composite_score": result.get("composite_score", 0)
            }
            
            # Add metrics
            metrics = result.get("metrics", {})
            for metric in all_metrics:
                row[metric] = metrics.get(metric, "")
            
            writer.writerow(row)
    
    logging.info(f"Generated CSV report: {csv_file}")
    return str(csv_file)


def generate_json_report(results: List[Dict[str, Any]], 
                        pareto_candidates: List[Dict[str, Any]], 
                        output_path: Path) -> str:
    """Generate JSON export of results."""
    
    export_data = {
        "metadata": {
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "total_experiments": len(results),
            "pareto_candidates": len(pareto_candidates),
            "version": "0.1.0"
        },
        "pareto_candidates": pareto_candidates,
        "all_results": results
    }
    
    json_file = output_path / "results_export.json"
    with open(json_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logging.info(f"Generated JSON report: {json_file}")
    return str(json_file)


def generate_markdown_report(results: List[Dict[str, Any]], 
                           pareto_candidates: List[Dict[str, Any]], 
                           output_path: Path) -> str:
    """Generate markdown report."""
    
    md_content = f"""# LLM Compression Analysis Report

**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Experiments:** {len(results)}
- **Successful Experiments:** {len([r for r in results if r.get('success', False)])}
- **Pareto Candidates:** {len(pareto_candidates)}
- **Success Rate:** {(len([r for r in results if r.get('success', False)]) / max(len(results), 1)) * 100:.1f}%

## Top Pareto Candidates

| Rank | Recipe ID | Score | Accuracy | Latency (ms) | VRAM (GB) | Energy (kWh/h) |
|------|-----------|-------|----------|--------------|-----------|----------------|
"""
    
    for candidate in pareto_candidates[:5]:
        metrics = candidate.get("metrics", {})
        md_content += f"""| {candidate.get('pareto_rank', '-')} | {candidate.get('recipe_id', 'Unknown')} | {candidate.get('composite_score', 0):.3f} | {metrics.get('accuracy', 0):.3f} | {metrics.get('latency_mean_ms', 0):.1f} | {metrics.get('vram_peak_gb', 0):.1f} | {metrics.get('energy_1hour_kwh', 0):.3f} |
"""
    
    md_content += """

## Visualization Files

- [Accuracy vs Latency](pareto_accuracy_latency.html)
- [Accuracy vs VRAM](pareto_accuracy_vram.html) 
- [3D Pareto Frontier](pareto_3d.html)
- [Parallel Coordinates](pareto_parallel.html)
- [Top Candidates Radar](pareto_radar.html)

## Methodology

This analysis was performed using a multi-agent optimization system that explores different compression techniques including:

- **Quantization**: AWQ, GPTQ, BitsAndBytes (4-bit, 8-bit)
- **KV Optimization**: FlashAttention, Paged Attention
- **Pruning**: Structured and unstructured pruning
- **Distillation**: LoRA, QLoRA knowledge distillation

Results are evaluated using Pareto frontier analysis across multiple objectives: accuracy, latency, VRAM usage, energy consumption, and CO2 emissions.

---
*Generated by LLM Compressor v0.1.0*
"""
    
    md_file = output_path / "analysis_report.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    logging.info(f"Generated Markdown report: {md_file}")
    return str(md_file)


def generate_statistics_report(results: List[Dict[str, Any]], 
                              pareto_candidates: List[Dict[str, Any]], 
                              output_path: Path) -> str:
    """Generate detailed statistics report."""
    
    successful_results = [r for r in results if r.get("success", False)]
    
    stats = {
        "experiment_summary": {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(results) - len(successful_results),
            "success_rate": len(successful_results) / max(len(results), 1),
            "pareto_candidates": len(pareto_candidates)
        }
    }
    
    if successful_results:
        # Calculate metric statistics
        metrics_stats = {}
        
        # Find all available metrics
        all_metrics = set()
        for result in successful_results:
            all_metrics.update(result.get("metrics", {}).keys())
        
        for metric in all_metrics:
            values = []
            for result in successful_results:
                value = result.get("metrics", {}).get(metric)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            
            if values:
                import numpy as np
                metrics_stats[metric] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }
        
        stats["metrics_statistics"] = metrics_stats
    
    # Optimization technique analysis
    optimization_counts = {
        "quantization_only": 0,
        "kv_only": 0,
        "pruning_only": 0,
        "distillation_only": 0,
        "combined": 0,
        "none": 0
    }
    
    for result in successful_results:
        recipe = result.get("recipe", {})
        techniques = []
        
        if recipe.get("quantization", {}).get("enabled", False):
            techniques.append("quantization")
        if recipe.get("kv_longcontext", {}).get("enabled", False):
            techniques.append("kv")
        if recipe.get("pruning_sparsity", {}).get("enabled", False):
            techniques.append("pruning")
        if recipe.get("distillation", {}).get("enabled", False):
            techniques.append("distillation")
        
        if len(techniques) == 0:
            optimization_counts["none"] += 1
        elif len(techniques) == 1:
            optimization_counts[f"{techniques[0]}_only"] += 1
        else:
            optimization_counts["combined"] += 1
    
    stats["optimization_analysis"] = optimization_counts
    
    # Save statistics
    stats_file = output_path / "detailed_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info(f"Generated statistics report: {stats_file}")
    return str(stats_file)


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Export and analyze LLM compression results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export from database
    python scripts/export_report.py --db experiments.db --output final_report
    
    # Export from JSON summary
    python scripts/export_report.py --input reports/execution_summary.json --output analysis
    
    # Generate specific format
    python scripts/export_report.py --db experiments.db --format html
    
    # Generate all formats
    python scripts/export_report.py --db experiments.db --format all
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--db", "-d",
        type=str,
        help="Path to SQLite database file"
    )
    
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to JSON summary file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_report",
        help="Output directory for reports (default: analysis_report)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["html", "csv", "json", "markdown", "all"],
        default="all",
        help="Output format (default: all)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info("Starting LLM Compression Results Export")
    
    # Load results
    if args.db:
        logging.info(f"Loading results from database: {args.db}")
        results = load_results_from_db(args.db)
    else:
        logging.info(f"Loading results from JSON: {args.input}")
        results = load_results_from_json(args.input)
    
    if not results:
        logging.error("No results loaded, exiting")
        sys.exit(1)
    
    # Generate reports
    try:
        generated_files = generate_comprehensive_report(results, args.output, args.format)
        
        # Print summary
        logging.info("=" * 60)
        logging.info("EXPORT COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        logging.info(f"Processed {len(results)} experiments")
        logging.info(f"Output directory: {args.output}")
        logging.info("Generated files:")
        
        for file_type, file_path in generated_files.items():
            logging.info(f"  {file_type}: {file_path}")
        
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()