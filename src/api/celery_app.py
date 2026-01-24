"""Celery configuration for async task processing."""

from celery import Celery
from celery.result import AsyncResult
import os
from typing import Dict, Any, List
import json
from pathlib import Path

from src.coordinator.coordinator import CompressionCoordinator
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.monitoring.mlflow_tracker import create_experiment_tracker

# Configure Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Configurable time limits (in seconds)
# Default: 24 hours for task_time_limit, 23 hours for soft limit
TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "86400"))  # 24 hours
TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "82800"))  # 23 hours

# Worker configuration
WORKER_PREFETCH_MULTIPLIER = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))
WORKER_MAX_TASKS_PER_CHILD = int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "1"))

app = Celery(
    "compression_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.api.celery_app"],
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=WORKER_MAX_TASKS_PER_CHILD,
)


@app.task(bind=True, name="compress_model")
def compress_model_task(
    self,
    model_name: str,
    dataset: str,
    max_episodes: int = 10,
    target_metric: str = "balanced",
    constraints: Dict[str, float] = None,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """Async task for model compression.

    Args:
        model_name: Model to compress
        dataset: Evaluation dataset
        max_episodes: Number of compression episodes
        target_metric: Optimization target
        constraints: Performance constraints
        use_mock: Use mock compression

    Returns:
        Compression results
    """
    job_id = self.request.id

    try:
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={
                "current_episode": 0,
                "max_episodes": max_episodes,
                "status": "Initializing",
            }
        )

        # Initialize coordinator
        coordinator = CompressionCoordinator(
            model_name=model_name,
            dataset=dataset,
            max_episodes=max_episodes,
            budget_hours=24.0,
            use_mock=use_mock,
        )

        # Set constraints
        if constraints:
            coordinator.constraints = constraints

        # Custom progress callback
        def progress_callback(episode: int, status: str):
            self.update_state(
                state="PROGRESS",
                meta={
                    "current_episode": episode,
                    "max_episodes": max_episodes,
                    "status": status,
                }
            )

        # Run compression with progress updates
        result = coordinator.run(
            interactive=False,
            progress_callback=progress_callback,
        )

        # Save results
        output_dir = Path(f"./results/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save Pareto frontier
        if "pareto_frontier" in result:
            with open(output_dir / "pareto_frontier.json", "w") as f:
                json.dump(result["pareto_frontier"], f, indent=2, default=str)

        return {
            "status": "completed",
            "job_id": job_id,
            "best_solution": result.get("best_solution"),
            "pareto_frontier_size": len(result.get("pareto_frontier", [])),
            "output_dir": str(output_dir),
        }

    except Exception as e:
        # Log error
        self.update_state(
            state="FAILURE",
            meta={
                "exc_type": type(e).__name__,
                "exc_message": str(e),
            }
        )
        raise


@app.task(bind=True, name="evaluate_checkpoint")
def evaluate_checkpoint_task(
    self,
    checkpoint_path: str,
    benchmarks: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Async task for model evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        benchmarks: Benchmarks to run
        batch_size: Batch size for evaluation

    Returns:
        Evaluation results
    """
    try:
        # Update state
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Loading model",
                "current_benchmark": None,
                "completed_benchmarks": [],
            }
        )

        # Initialize runner
        runner = BenchmarkRunner(use_mock=False)

        # Progress callback
        def benchmark_callback(benchmark: str, status: str):
            self.update_state(
                state="PROGRESS",
                meta={
                    "status": status,
                    "current_benchmark": benchmark,
                }
            )

        # Run evaluation
        results = runner.run_full_evaluation(
            checkpoint_path=checkpoint_path,
            benchmarks=benchmarks,
            batch_size=batch_size,
            progress_callback=benchmark_callback,
        )

        return results

    except Exception as e:
        self.update_state(
            state="FAILURE",
            meta={
                "exc_type": type(e).__name__,
                "exc_message": str(e),
            }
        )
        raise


@app.task(name="cleanup_old_jobs")
def cleanup_old_jobs_task(days: int = 7) -> Dict[str, Any]:
    """Clean up old job data.

    Args:
        days: Delete jobs older than this many days

    Returns:
        Cleanup statistics
    """
    from datetime import datetime, timedelta
    import shutil

    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0
    freed_space_gb = 0

    # Clean up result directories
    results_dir = Path("./results")
    if results_dir.exists():
        for job_dir in results_dir.iterdir():
            if job_dir.is_dir():
                # Check modification time
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mtime < cutoff_date:
                    # Calculate size
                    size = sum(
                        f.stat().st_size for f in job_dir.rglob('*') if f.is_file()
                    )
                    freed_space_gb += size / (1024**3)

                    # Delete directory
                    shutil.rmtree(job_dir)
                    deleted_count += 1

    # Clean up checkpoints
    checkpoint_dir = Path("./checkpoints")
    if checkpoint_dir.exists():
        for ckpt_dir in checkpoint_dir.iterdir():
            if ckpt_dir.is_dir():
                mtime = datetime.fromtimestamp(ckpt_dir.stat().st_mtime)
                if mtime < cutoff_date:
                    size = sum(
                        f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file()
                    )
                    freed_space_gb += size / (1024**3)
                    shutil.rmtree(ckpt_dir)

    return {
        "deleted_jobs": deleted_count,
        "freed_space_gb": freed_space_gb,
        "cutoff_date": cutoff_date.isoformat(),
    }


@app.task(name="generate_compression_report")
def generate_report_task(job_id: str) -> Dict[str, Any]:
    """Generate detailed compression report.

    Args:
        job_id: Job ID

    Returns:
        Report data
    """
    results_path = Path(f"./results/{job_id}/results.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found for job {job_id}")

    with open(results_path, "r") as f:
        results = json.load(f)

    # Generate report
    report = {
        "job_id": job_id,
        "summary": {
            "best_accuracy": results.get("best_solution", {}).get("accuracy", 0),
            "best_compression": results.get("best_compression_ratio", 1),
            "pareto_solutions": len(results.get("pareto_frontier", [])),
            "total_episodes": results.get("total_strategies", 0),
        },
        "pareto_frontier": results.get("pareto_frontier", []),
        "strategies_tried": results.get("strategies", []),
        "timestamps": {
            "start": results.get("start_time"),
            "end": results.get("end_time"),
            "duration_hours": results.get("duration_hours", 0),
        }
    }

    # Save report
    report_path = Path(f"./results/{job_id}/report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# Periodic tasks
from celery.schedules import crontab

app.conf.beat_schedule = {
    "cleanup-old-jobs": {
        "task": "cleanup_old_jobs",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        "args": (7,),  # Clean up jobs older than 7 days
    },
}


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task.

    Args:
        task_id: Task ID

    Returns:
        Task status and metadata
    """
    result = AsyncResult(task_id, app=app)

    return {
        "task_id": task_id,
        "state": result.state,
        "current": result.info.get("current", 0) if result.info else 0,
        "total": result.info.get("total", 1) if result.info else 1,
        "status": result.info.get("status", "") if result.info else "",
        "result": result.result if result.state == "SUCCESS" else None,
    }