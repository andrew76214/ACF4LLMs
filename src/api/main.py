"""FastAPI application for compression framework."""

import os
import sys
import threading
from collections import deque
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
import json

from src.coordinator.langgraph_coordinator import LangGraphCoordinator
from src.coordinator.spec_inference import infer_spec
from src.common.schemas import CompressionMethod, Benchmark


class LogCapture:
    """Capture stdout and store logs in job storage."""

    def __init__(self, job_id: str, storage: "ThreadSafeJobStorage", max_lines: int = 1000):
        self.job_id = job_id
        self.storage = storage
        self.logs: deque = deque(maxlen=max_lines)
        self._original_stdout = None

    def write(self, text: str):
        if self._original_stdout:
            self._original_stdout.write(text)
        for line in text.splitlines():
            if line.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.logs.append(f"[{timestamp}] {line}")
                self.storage.update(self.job_id, logs=list(self.logs))

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Compression Framework API",
    description="REST API for automated model compression",
    version="1.0.0",
)

# CORS configuration from environment
# In production, set ALLOWED_ORIGINS to comma-separated list of allowed domains
# Example: ALLOWED_ORIGINS=https://myapp.com,https://admin.myapp.com
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# Optional: regex pattern for dynamic origins (e.g., Cloudflare Pages preview URLs)
# Example: ALLOWED_ORIGIN_REGEX=https://.*\.acf4llms\.pages\.dev
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "")

# Development mode allows all origins (not recommended for production)
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "development").lower() == "development"

if IS_DEVELOPMENT and not ALLOWED_ORIGINS:
    # In development, allow localhost by default
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

# Add CORS middleware with secure defaults
app.add_middleware(
    CORSMiddleware,
    # Never use "*" with credentials=True in production
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"] if IS_DEVELOPMENT else [],
    # Regex pattern for dynamic origins (Cloudflare Pages previews, etc.)
    allow_origin_regex=ALLOWED_ORIGIN_REGEX if ALLOWED_ORIGIN_REGEX else None,
    # Only allow credentials if explicitly enabled and origins are specified
    allow_credentials=ALLOW_CREDENTIALS and bool(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


class ThreadSafeJobStorage:
    """Thread-safe storage for job status with lock protection.

    In production, consider using Redis or a database for persistence
    and better scalability across multiple workers.
    """

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status by ID (thread-safe)."""
        with self._lock:
            return self._storage.get(job_id)

    def set(self, job_id: str, data: Dict[str, Any]) -> None:
        """Set job status (thread-safe)."""
        with self._lock:
            self._storage[job_id] = data

    def update(self, job_id: str, **kwargs) -> None:
        """Update specific fields of a job (thread-safe)."""
        with self._lock:
            if job_id in self._storage:
                self._storage[job_id].update(kwargs)

    def delete(self, job_id: str) -> None:
        """Delete a job (thread-safe)."""
        with self._lock:
            self._storage.pop(job_id, None)

    def list_all(self) -> List[Dict[str, Any]]:
        """Get all jobs (thread-safe copy)."""
        with self._lock:
            return list(self._storage.values())

    def __contains__(self, job_id: str) -> bool:
        """Check if job exists (thread-safe)."""
        with self._lock:
            return job_id in self._storage


# Thread-safe storage for job status
jobs_storage = ThreadSafeJobStorage()


class CompressionRequest(BaseModel):
    """Request model for compression job."""
    model_name: str = Field(..., description="Model name or HuggingFace path")
    dataset: str = Field(..., description="Dataset for evaluation")
    max_episodes: int = Field(default=10, description="Maximum episodes to run")
    target_metric: Optional[str] = Field(default="balanced", description="Target optimization metric")
    compression_methods: Optional[List[str]] = Field(default=None, description="Specific methods to use")
    constraints: Optional[Dict[str, float]] = Field(default=None, description="Performance constraints")
    use_mock: bool = Field(default=False, description="Use mock compression for testing")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelSpecRequest(BaseModel):
    """Request for model specification inference."""
    model_name: str
    dataset: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Agentic Compression Framework",
        "version": "1.0.0",
        "endpoints": {
            "POST /compress": "Start compression job",
            "GET /jobs/{job_id}": "Get job status",
            "GET /jobs/{job_id}/logs": "Get job logs",
            "GET /jobs": "List all jobs",
            "POST /spec/infer": "Infer model specification",
            "GET /pareto/{job_id}": "Get Pareto frontier",
            "GET /download/{job_id}/{checkpoint_id}": "Download compressed model",
        }
    }


@app.post("/compress", response_model=JobStatus)
async def start_compression(
    request: CompressionRequest,
    background_tasks: BackgroundTasks,
) -> JobStatus:
    """Start a compression job."""
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create initial job status
    job_status = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "progress": {
            "current_episode": 0,
            "max_episodes": request.max_episodes,
            "pareto_solutions": 0,
        },
        "result": None,
        "error": None,
    }

    jobs_storage.set(job_id, job_status)

    # Start compression in background
    background_tasks.add_task(
        run_compression_job,
        job_id,
        request.dict(),
    )

    return JobStatus(**job_status)


async def run_compression_job(job_id: str, config: Dict[str, Any]):
    """Run compression job in background."""
    # Initialize logs in job storage
    jobs_storage.update(job_id, logs=[])

    try:
        # Update status
        jobs_storage.update(job_id, status="running", updated_at=datetime.now())

        # Initialize coordinator
        coordinator = LangGraphCoordinator(
            model_name=config["model_name"],
            dataset=config["dataset"],
            max_episodes=config["max_episodes"],
            budget_hours=24.0,
        )

        # Store experiment directory in job
        jobs_storage.update(job_id, experiment_dir=str(coordinator.experiment_dir))

        # Run compression with log capture
        with LogCapture(job_id, jobs_storage) as log_capture:
            result = await asyncio.to_thread(
                coordinator.run,
                interactive=False,
            )

        # Update with results
        jobs_storage.update(
            job_id,
            status="completed",
            result={
                "best_solutions": result.get("best_solutions") if result else None,
                "pareto_frontier_size": len(result.get("pareto_frontier", [])) if result else 0,
                "total_strategies_tried": result.get("total_strategies", 0) if result else 0,
                "compression_achieved": result.get("best_compression_ratio", 1.0) if result else 1.0,
            }
        )

    except Exception as e:
        jobs_storage.update(job_id, status="failed", error=str(e))

    finally:
        jobs_storage.update(job_id, updated_at=datetime.now())


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a compression job."""
    job_data = jobs_storage.get(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**job_data)


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    offset: int = 0,
    limit: int = 200,
) -> Dict[str, Any]:
    """Get logs for a compression job."""
    job_data = jobs_storage.get(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = job_data.get("logs", [])
    total = len(logs)

    # Apply pagination
    paginated_logs = logs[offset:offset + limit]

    return {
        "logs": paginated_logs,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/episodes/{job_id}")
async def get_episodes(job_id: str) -> Dict[str, Any]:
    """Get episode history for a job including coordinator reasoning."""
    job_data = jobs_storage.get(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")

    experiment_dir = job_data.get("experiment_dir")
    if not experiment_dir:
        return {"episodes": [], "total": 0, "message": "No experiment directory found"}

    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        return {"episodes": [], "total": 0, "message": "Experiment directory does not exist"}

    # Load Pareto frontier to check which episodes are Pareto optimal
    pareto_strategy_ids = set()
    pareto_path = experiment_path / "pareto_frontier.json"
    if pareto_path.exists():
        try:
            with open(pareto_path, "r") as f:
                pareto_data = json.load(f)
                for sol in pareto_data.get("solutions", []):
                    strategy = sol.get("strategy", {})
                    if strategy.get("strategy_id"):
                        pareto_strategy_ids.add(strategy["strategy_id"])
        except Exception:
            pass

    episodes = []
    # Find all episode directories
    episode_dirs = sorted(experiment_path.glob("episode_*"))

    for episode_dir in episode_dirs:
        # Extract episode number from directory name
        dir_name = episode_dir.name
        try:
            episode_id = int(dir_name.replace("episode_", "").replace("_pipeline", ""))
        except ValueError:
            continue

        # Load strategy.json
        strategy_path = episode_dir / "strategy.json"
        strategy_data = None
        if strategy_path.exists():
            try:
                with open(strategy_path, "r") as f:
                    strategy_data = json.load(f)
            except Exception:
                pass

        # Load results.json
        results_path = episode_dir / "results.json"
        result_data = None
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    result_data = json.load(f)
            except Exception:
                pass

        if strategy_data is None:
            continue

        # Build episode decision info
        decision = {
            "episode_id": episode_id,
            "action": strategy_data.get("quantization_method", "unknown"),
            "method": strategy_data.get("quantization_method"),
            "reasoning": strategy_data.get("coordinator_reasoning", ""),
            "params": {
                "bits": strategy_data.get("quantization_bits"),
                "lora_rank": strategy_data.get("lora_rank"),
                "pruning_ratio": strategy_data.get("pruning_ratio"),
                "pruning_method": strategy_data.get("pruning_method"),
                "asvd_rank_ratio": strategy_data.get("asvd_rank_ratio"),
                "pipeline_name": strategy_data.get("pipeline_name"),
            },
            "timestamp": strategy_data.get("coordinator_decision_timestamp", ""),
            "skill_recommendations": strategy_data.get("skill_recommendations"),
        }

        # Filter out None params
        decision["params"] = {k: v for k, v in decision["params"].items() if v is not None}

        # Check if this episode is Pareto optimal
        strategy_id = strategy_data.get("strategy_id", "")
        is_pareto = strategy_id in pareto_strategy_ids

        episode = {
            "episode_id": episode_id,
            "decision": decision,
            "strategy": strategy_data,
            "result": result_data,
            "is_pareto": is_pareto,
        }

        episodes.append(episode)

    # Sort by episode_id
    episodes.sort(key=lambda x: x["episode_id"])

    return {
        "episodes": episodes,
        "total": len(episodes),
    }


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
) -> List[JobStatus]:
    """List all jobs with optional filtering."""
    jobs = jobs_storage.list_all()

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply limit
    jobs = jobs[:limit]

    return [JobStatus(**j) for j in jobs]


@app.post("/spec/infer")
async def infer_model_spec(request: ModelSpecRequest) -> Dict[str, Any]:
    """Infer model specification."""
    try:
        spec = infer_spec(request.model_name, request.dataset)
        return spec.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/pareto/{job_id}")
async def get_pareto_frontier(job_id: str) -> Dict[str, Any]:
    """Get Pareto frontier for a job."""
    job = jobs_storage.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job['status']}, not completed"
        )

    # In production, load from storage
    pareto_path = Path(f"./checkpoints/{job_id}/pareto_frontier.json")

    if not pareto_path.exists():
        return {"solutions": [], "message": "No Pareto frontier data available"}

    with open(pareto_path, "r") as f:
        pareto_data = json.load(f)

    return pareto_data


@app.get("/download/{job_id}/{checkpoint_id}")
async def download_checkpoint(job_id: str, checkpoint_id: str):
    """Download compressed model checkpoint."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoint_path = Path(f"./checkpoints/{job_id}/{checkpoint_id}")

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Create tar archive for directory
    import tarfile
    import io

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add(checkpoint_path, arcname=checkpoint_id)

    tar_buffer.seek(0)

    return FileResponse(
        tar_buffer,
        media_type="application/x-tar",
        headers={
            "Content-Disposition": f"attachment; filename={checkpoint_id}.tar.gz"
        }
    )


@app.post("/evaluate")
async def evaluate_model(
    checkpoint_path: str,
    benchmarks: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate a compressed model on benchmarks."""
    from src.evaluation.benchmark_runner import BenchmarkRunner

    try:
        runner = BenchmarkRunner(use_mock=False)

        results = await asyncio.to_thread(
            runner.run_full_evaluation,
            checkpoint_path,
            benchmarks,
            batch_size,
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/methods")
async def list_compression_methods() -> List[str]:
    """List available compression methods."""
    return [method.value for method in CompressionMethod]


@app.get("/benchmarks")
async def list_benchmarks() -> List[str]:
    """List available benchmarks."""
    return [benchmark.value for benchmark in Benchmark]


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy",
        "cuda_available": str(torch.cuda.is_available()),
        "gpu_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
    }


@app.get("/gpu")
async def get_gpu_status() -> Dict[str, Any]:
    """Get detailed GPU status including names and utilization."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "gpus": []}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # Get memory info
        mem_total = props.total_memory
        mem_reserved = torch.cuda.memory_reserved(i)
        mem_allocated = torch.cuda.memory_allocated(i)

        gpus.append({
            "index": i,
            "name": props.name,
            "memory_total_gb": round(mem_total / (1024**3), 2),
            "memory_used_gb": round(mem_allocated / (1024**3), 2),
            "memory_reserved_gb": round(mem_reserved / (1024**3), 2),
            "memory_utilization": round((mem_allocated / mem_total) * 100, 1) if mem_total > 0 else 0,
            "compute_capability": f"{props.major}.{props.minor}",
        })

    return {"available": True, "gpus": gpus}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)