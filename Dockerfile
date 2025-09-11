# Dockerfile for LLM Compressor
# Multi-agent system for LLM compression and optimization

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Use system pip directly

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install essential packages first
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    accelerate \
    datasets \
    bitsandbytes

# Install monitoring and visualization libraries
RUN pip install --no-cache-dir \
    plotly \
    pynvml \
    GPUtil \
    psutil \
    scikit-learn \
    pandas \
    numpy \
    pyyaml \
    tqdm \
    click

# Install vLLM (optional, comment out if causing issues)
# RUN pip install --no-cache-dir vllm

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    black \
    isort

# Copy the project files
COPY llm_compressor/ ./llm_compressor/
COPY llm_compressor/configs/ ./configs/
COPY llm_compressor/scripts/ ./scripts/
COPY Makefile .
COPY README.md .

# Create necessary directories
RUN mkdir -p reports artifacts logs

# Set permissions for scripts
RUN chmod +x scripts/*.py scripts/*.sh

# Create a non-root user (use different UID to avoid conflict)
RUN useradd -m -u 1001 llmuser && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import llm_compressor; print('LLM Compressor OK')" || exit 1

# Default command
CMD ["python3", "scripts/run_search.py", "--help"]

# Build arguments for customization
ARG MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
ARG GPU_TYPE="A100"
ARG SEQUENCE_LENGTH="4096"

# Environment variables that can be overridden
ENV MODEL_NAME=${MODEL_NAME}
ENV GPU_TYPE=${GPU_TYPE}
ENV SEQUENCE_LENGTH=${SEQUENCE_LENGTH}

# Labels for metadata
LABEL maintainer="LLM Compressor Team"
LABEL version="0.1.0"
LABEL description="Multi-agent LLM compression and optimization system"
LABEL gpu.required="true"
LABEL gpu.vendor="nvidia"
LABEL gpu.memory.min="40GB"

# Documentation
LABEL org.opencontainers.image.title="LLM Compressor"
LABEL org.opencontainers.image.description="Multi-agent system for LLM compression with Pareto optimization"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.vendor="LLM Compressor Team"
LABEL org.opencontainers.image.source="https://github.com/example/llm-compressor"
LABEL org.opencontainers.image.documentation="https://github.com/example/llm-compressor/blob/main/README.md"