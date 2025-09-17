"""Registry for tracking experiments, artifacts, and metrics."""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime


class Registry:
    """Registry for experiment tracking and artifact management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(config.get("db_path", "experiments.db"))
        self.artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))
        self.logger = logging.getLogger("registry")
        
        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for experiment tracking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    recipe_data TEXT,
                    metrics_data TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Artifacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    artifact_type TEXT,
                    artifact_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Metrics table for time series data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.commit()
    
    def register_experiment(self, experiment_id: str, data: Dict[str, Any]):
        """Register a new experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO experiments 
                (id, recipe_data, metrics_data, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                json.dumps(data.get("recipe", {})),
                json.dumps(data.get("metrics", {})),
                "completed" if data.get("success", False) else "failed",
                now,
                now
            ))
            
            # Store individual metrics for querying
            metrics = data.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute("""
                        INSERT INTO metrics (experiment_id, metric_name, metric_value, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (experiment_id, metric_name, metric_value, now))
            
            conn.commit()
            
        self.logger.info(f"Registered experiment: {experiment_id}")
    
    def store_artifact(self, experiment_id: str, artifact_type: str, 
                      artifact_data: Any, metadata: Optional[Dict] = None) -> str:
        """Store an artifact and return its path."""
        artifact_filename = f"{experiment_id}_{artifact_type}_{int(datetime.now().timestamp())}.pkl"
        artifact_path = self.artifacts_dir / artifact_filename
        
        # Store the artifact
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact_data, f)
        
        # Register in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO artifacts (experiment_id, artifact_type, artifact_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                experiment_id,
                artifact_type,
                str(artifact_path),
                json.dumps(metadata or {}),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        self.logger.info(f"Stored artifact: {artifact_path}")
        return str(artifact_path)
    
    def load_artifact(self, artifact_path: str) -> Any:
        """Load an artifact from disk."""
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT recipe_data, metrics_data, status, created_at, updated_at
                FROM experiments WHERE id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": experiment_id,
                    "recipe": json.loads(row[0]),
                    "metrics": json.loads(row[1]),
                    "status": row[2],
                    "created_at": row[3],
                    "updated_at": row[4]
                }
        return None
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT id, recipe_data, metrics_data, status, created_at, updated_at
                    FROM experiments WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT id, recipe_data, metrics_data, status, created_at, updated_at
                    FROM experiments
                    ORDER BY created_at DESC
                """)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    "id": row[0],
                    "recipe": json.loads(row[1]),
                    "metrics": json.loads(row[2]),
                    "status": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                })
            
            return experiments
    
    def get_metrics_history(self, metric_name: str, 
                           experiment_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metric history across experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if experiment_ids:
                placeholders = ",".join("?" for _ in experiment_ids)
                cursor.execute(f"""
                    SELECT experiment_id, metric_value, timestamp
                    FROM metrics 
                    WHERE metric_name = ? AND experiment_id IN ({placeholders})
                    ORDER BY timestamp
                """, [metric_name] + experiment_ids)
            else:
                cursor.execute("""
                    SELECT experiment_id, metric_value, timestamp
                    FROM metrics 
                    WHERE metric_name = ?
                    ORDER BY timestamp
                """, (metric_name,))
            
            return [
                {
                    "experiment_id": row[0],
                    "value": row[1],
                    "timestamp": row[2]
                }
                for row in cursor.fetchall()
            ]
    
    def get_pareto_candidates(self, metrics: List[str], 
                            maximize: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get experiments that form the Pareto frontier for given metrics."""
        maximize = maximize or []
        experiments = self.list_experiments(status="completed")
        
        pareto_candidates = []
        for exp in experiments:
            exp_metrics = exp["metrics"]
            
            # Check if experiment has all required metrics
            if not all(metric in exp_metrics for metric in metrics):
                continue
            
            is_pareto = True
            for other_exp in experiments:
                if other_exp["id"] == exp["id"]:
                    continue
                
                other_metrics = other_exp["metrics"]
                if not all(metric in other_metrics for metric in metrics):
                    continue
                
                # Check if other experiment dominates this one
                dominates = True
                for metric in metrics:
                    if metric in maximize:
                        if other_metrics[metric] <= exp_metrics[metric]:
                            dominates = False
                            break
                    else:
                        if other_metrics[metric] >= exp_metrics[metric]:
                            dominates = False
                            break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_candidates.append(exp)
        
        return pareto_candidates
    
    def export_summary(self, output_path: str):
        """Export experiment summary to JSON."""
        experiments = self.list_experiments()
        summary = {
            "total_experiments": len(experiments),
            "successful_experiments": len([e for e in experiments if e["status"] == "completed"]),
            "experiments": experiments,
            "exported_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Exported summary to: {output_path}")
    
    def cleanup_old_artifacts(self, days: int = 30):
        """Clean up artifacts older than specified days."""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT artifact_path FROM artifacts 
                WHERE created_at < datetime(?, 'unixepoch')
            """, (cutoff_date,))
            
            old_artifacts = cursor.fetchall()
            
            for (artifact_path,) in old_artifacts:
                try:
                    Path(artifact_path).unlink(missing_ok=True)
                    self.logger.info(f"Deleted old artifact: {artifact_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {artifact_path}: {e}")
            
            # Remove from database
            cursor.execute("""
                DELETE FROM artifacts 
                WHERE created_at < datetime(?, 'unixepoch')
            """, (cutoff_date,))
            
            conn.commit()
        
        self.logger.info(f"Cleaned up artifacts older than {days} days")