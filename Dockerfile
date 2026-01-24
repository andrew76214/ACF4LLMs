# Multi-stage build for production
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    streamlit \
    prometheus-client \
    redis \
    celery

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/mlruns

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command (can be overridden)
CMD ["python3", "scripts/run_pipeline.py"]

# ===== API Stage =====
FROM base AS api

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ===== Dashboard Stage =====
FROM base AS dashboard

EXPOSE 8501
CMD ["streamlit", "run", "src.dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# ===== Worker Stage =====
FROM base AS worker

CMD ["celery", "-A", "src.api.celery_app", "worker", "--loglevel=info", "--concurrency=2"]