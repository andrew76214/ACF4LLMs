"""Pareto frontier analysis for multi-objective optimization."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path


class ParetoFrontier:
    """Pareto frontier analysis and visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pareto")
        self.objectives = config.get("objectives", ["accuracy", "latency", "vram", "energy", "co2e"])
        self.maximize = config.get("maximize", ["accuracy"])  # Objectives to maximize
        self.minimize = [obj for obj in self.objectives if obj not in self.maximize]
        
    def analyze(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze results and find Pareto frontier."""
        if not results:
            return []
        
        # Extract valid results with all required metrics
        valid_results = self._filter_valid_results(results)
        
        if not valid_results:
            self.logger.warning("No valid results found for Pareto analysis")
            return []
        
        # Find Pareto frontier
        pareto_indices = self._find_pareto_frontier(valid_results)
        pareto_candidates = [valid_results[i] for i in pareto_indices]
        
        # Rank candidates by composite score
        ranked_candidates = self._rank_candidates(pareto_candidates)
        
        self.logger.info(f"Found {len(ranked_candidates)} Pareto optimal candidates")
        
        return ranked_candidates
    
    def _filter_valid_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results that have all required metrics."""
        valid_results = []
        
        for result in results:
            if not result.get("success", False):
                continue
                
            metrics = result.get("metrics", {})
            
            # Check if all required objectives are present
            if all(obj in metrics for obj in self.objectives):
                valid_results.append(result)
            else:
                missing = [obj for obj in self.objectives if obj not in metrics]
                self.logger.debug(f"Result {result.get('recipe_id', 'unknown')} missing metrics: {missing}")
        
        return valid_results
    
    def _find_pareto_frontier(self, results: List[Dict[str, Any]]) -> List[int]:
        """Find Pareto frontier indices."""
        if not results:
            return []
        
        # Extract objective values
        objective_matrix = self._extract_objective_matrix(results)
        
        # Find Pareto optimal solutions
        pareto_mask = np.ones(len(results), dtype=bool)
        
        for i, candidate in enumerate(objective_matrix):
            if pareto_mask[i]:
                # Check if any other point dominates this candidate
                dominated = self._is_dominated(candidate, objective_matrix)
                pareto_mask[i] = not dominated
        
        return np.where(pareto_mask)[0].tolist()
    
    def _extract_objective_matrix(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Extract objective values into a matrix."""
        matrix = []
        
        for result in results:
            metrics = result["metrics"]
            values = []
            
            for obj in self.objectives:
                value = metrics[obj]
                
                # Convert to minimization problem (negate maximization objectives)
                if obj in self.maximize:
                    value = -value
                
                values.append(value)
            
            matrix.append(values)
        
        return np.array(matrix)
    
    def _is_dominated(self, candidate: np.ndarray, population: np.ndarray) -> bool:
        """Check if a candidate is dominated by any solution in the population."""
        for other in population:
            if np.array_equal(candidate, other):
                continue
                
            # Check if 'other' dominates 'candidate'
            # (better or equal in all objectives, strictly better in at least one)
            if np.all(other <= candidate) and np.any(other < candidate):
                return True
        
        return False
    
    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank Pareto candidates by composite score."""
        
        # Calculate composite scores
        for candidate in candidates:
            score = self._calculate_composite_score(candidate["metrics"])
            candidate["composite_score"] = score
        
        # Sort by composite score (descending)
        ranked = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)
        
        # Add rank information
        for i, candidate in enumerate(ranked):
            candidate["pareto_rank"] = i + 1
        
        return ranked
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted composite score."""
        weights = self.config.get("objective_weights", {
            "accuracy": 1.0,
            "latency": -1.0,
            "vram": -1.0,
            "energy": -1.0,
            "co2e": -1.0
        })
        
        score = 0.0
        for obj, weight in weights.items():
            if obj in metrics:
                score += weight * metrics[obj]
        
        return score
    
    def visualize_frontier(self, results: List[Dict[str, Any]], 
                          output_dir: str = "reports") -> Dict[str, str]:
        """Create visualizations of the Pareto frontier."""
        
        if len(results) < 2:
            self.logger.warning("Not enough results for visualization")
            return {}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 2D scatter plots for key objective pairs
        plots.update(self._create_2d_plots(results, output_path))
        
        # 3D plot if we have at least 3 objectives
        if len(self.objectives) >= 3:
            plots.update(self._create_3d_plot(results, output_path))
        
        # Parallel coordinates plot
        plots.update(self._create_parallel_plot(results, output_path))
        
        # Radar chart for top candidates
        plots.update(self._create_radar_chart(results, output_path))
        
        return plots
    
    def _create_2d_plots(self, results: List[Dict[str, Any]], 
                        output_path: Path) -> Dict[str, str]:
        """Create 2D scatter plots for objective pairs."""
        plots = {}
        
        # Key objective pairs to plot
        objective_pairs = [
            ("accuracy", "latency"),
            ("accuracy", "vram"),
            ("latency", "energy"),
            ("vram", "co2e")
        ]
        
        for obj1, obj2 in objective_pairs:
            if obj1 not in self.objectives or obj2 not in self.objectives:
                continue
            
            # Extract values
            x_vals = [r["metrics"].get(obj1, 0) for r in results if r.get("success", False)]
            y_vals = [r["metrics"].get(obj2, 0) for r in results if r.get("success", False)]
            
            if not x_vals or not y_vals:
                continue
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add all points
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name='All Results',
                marker=dict(color='lightblue', size=8),
                text=[f"Recipe {r.get('recipe_id', 'unknown')}" for r in results if r.get("success", False)],
                hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<extra></extra>'
            ))
            
            # Highlight Pareto frontier
            pareto_results = [r for r in results if r.get("pareto_rank")]
            if pareto_results:
                pareto_x = [r["metrics"].get(obj1, 0) for r in pareto_results]
                pareto_y = [r["metrics"].get(obj2, 0) for r in pareto_results]
                
                fig.add_trace(go.Scatter(
                    x=pareto_x,
                    y=pareto_y,
                    mode='markers',
                    name='Pareto Frontier',
                    marker=dict(color='red', size=12, symbol='star'),
                    text=[f"Rank {r.get('pareto_rank', 'unknown')}" for r in pareto_results],
                    hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<extra></extra>'
                ))
            
            # Formatting
            fig.update_layout(
                title=f'Pareto Analysis: {obj1.title()} vs {obj2.title()}',
                xaxis_title=obj1.title(),
                yaxis_title=obj2.title(),
                hovermode='closest'
            )
            
            # Save plot
            plot_path = output_path / f"pareto_{obj1}_{obj2}.html"
            fig.write_html(str(plot_path))
            plots[f"{obj1}_{obj2}"] = str(plot_path)
        
        return plots
    
    def _create_3d_plot(self, results: List[Dict[str, Any]], 
                       output_path: Path) -> Dict[str, str]:
        """Create 3D scatter plot for three key objectives."""
        
        # Use first three objectives
        obj1, obj2, obj3 = self.objectives[:3]
        
        # Extract values
        valid_results = [r for r in results if r.get("success", False)]
        x_vals = [r["metrics"].get(obj1, 0) for r in valid_results]
        y_vals = [r["metrics"].get(obj2, 0) for r in valid_results]
        z_vals = [r["metrics"].get(obj3, 0) for r in valid_results]
        
        if not all([x_vals, y_vals, z_vals]):
            return {}
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add all points
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            name='All Results',
            marker=dict(color='lightblue', size=6),
            text=[f"Recipe {r.get('recipe_id', 'unknown')}" for r in valid_results],
            hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<br>{obj3}: %{{z}}<extra></extra>'
        ))
        
        # Highlight Pareto frontier
        pareto_results = [r for r in valid_results if r.get("pareto_rank")]
        if pareto_results:
            pareto_x = [r["metrics"].get(obj1, 0) for r in pareto_results]
            pareto_y = [r["metrics"].get(obj2, 0) for r in pareto_results]
            pareto_z = [r["metrics"].get(obj3, 0) for r in pareto_results]
            
            fig.add_trace(go.Scatter3d(
                x=pareto_x,
                y=pareto_y,
                z=pareto_z,
                mode='markers',
                name='Pareto Frontier',
                marker=dict(color='red', size=10, symbol='diamond'),
                text=[f"Rank {r.get('pareto_rank', 'unknown')}" for r in pareto_results]
            ))
        
        # Formatting
        fig.update_layout(
            title=f'3D Pareto Analysis: {obj1.title()}, {obj2.title()}, {obj3.title()}',
            scene=dict(
                xaxis_title=obj1.title(),
                yaxis_title=obj2.title(),
                zaxis_title=obj3.title()
            )
        )
        
        # Save plot
        plot_path = output_path / "pareto_3d.html"
        fig.write_html(str(plot_path))
        
        return {"3d_pareto": str(plot_path)}
    
    def _create_parallel_plot(self, results: List[Dict[str, Any]], 
                             output_path: Path) -> Dict[str, str]:
        """Create parallel coordinates plot."""
        
        valid_results = [r for r in results if r.get("success", False)]
        if not valid_results:
            return {}
        
        # Prepare data for parallel plot
        data = []
        for result in valid_results:
            metrics = result["metrics"]
            row = {}
            
            for obj in self.objectives:
                if obj in metrics:
                    row[obj] = metrics[obj]
            
            row["recipe_id"] = result.get("recipe_id", "unknown")
            row["is_pareto"] = bool(result.get("pareto_rank"))
            data.append(row)
        
        if not data:
            return {}
        
        # Create parallel coordinates plot
        dimensions = []
        for obj in self.objectives:
            if any(obj in row for row in data):
                values = [row.get(obj, 0) for row in data]
                dimensions.append(dict(
                    label=obj.title(),
                    values=values
                ))
        
        # Color by Pareto membership
        colors = [1 if row["is_pareto"] else 0 for row in data]
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=colors, colorscale='Viridis'),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title='Parallel Coordinates: All Objectives',
            font=dict(size=12)
        )
        
        # Save plot
        plot_path = output_path / "pareto_parallel.html"
        fig.write_html(str(plot_path))
        
        return {"parallel": str(plot_path)}
    
    def _create_radar_chart(self, results: List[Dict[str, Any]], 
                           output_path: Path) -> Dict[str, str]:
        """Create radar chart for top Pareto candidates."""
        
        # Get top 3 Pareto candidates
        pareto_results = [r for r in results if r.get("pareto_rank") and r["pareto_rank"] <= 3]
        
        if not pareto_results:
            return {}
        
        fig = go.Figure()
        
        for result in pareto_results:
            metrics = result["metrics"]
            
            # Normalize metrics to 0-1 scale for radar chart
            normalized_values = []
            labels = []
            
            for obj in self.objectives:
                if obj in metrics:
                    # Normalize to 0-1 (this is simplified - should use proper normalization)
                    value = metrics[obj]
                    
                    # For maximization objectives, use value as-is
                    # For minimization, use 1 - normalized_value
                    if obj in self.maximize:
                        normalized_value = min(1.0, max(0.0, value))
                    else:
                        # Inverse for minimization (higher is better on radar)
                        normalized_value = 1.0 / (1.0 + value) if value > 0 else 1.0
                    
                    normalized_values.append(normalized_value)
                    labels.append(obj.title())
            
            # Add trace for this candidate
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=labels,
                fill='toself',
                name=f"Rank {result['pareto_rank']} ({result.get('recipe_id', 'unknown')})",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Top Pareto Candidates Comparison"
        )
        
        # Save plot
        plot_path = output_path / "pareto_radar.html"
        fig.write_html(str(plot_path))
        
        return {"radar": str(plot_path)}
    
    def export_summary(self, results: List[Dict[str, Any]], 
                      output_path: str = "reports/pareto_summary.json") -> str:
        """Export Pareto analysis summary."""
        
        pareto_results = [r for r in results if r.get("pareto_rank")]
        
        summary = {
            "total_candidates": len(results),
            "pareto_candidates": len(pareto_results),
            "objectives": self.objectives,
            "maximize": self.maximize,
            "minimize": self.minimize,
            "top_candidates": []
        }
        
        # Add top 5 candidates
        for result in pareto_results[:5]:
            candidate_summary = {
                "rank": result["pareto_rank"],
                "recipe_id": result.get("recipe_id", "unknown"),
                "composite_score": result.get("composite_score", 0),
                "metrics": {obj: result["metrics"].get(obj, 0) for obj in self.objectives},
                "recipe_config": result.get("recipe", {})
            }
            summary["top_candidates"].append(candidate_summary)
        
        # Save summary
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Exported Pareto summary to: {output_path}")
        return str(output_path)