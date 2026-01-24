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
                "best_solution": result["best_solution"] if result else None,
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