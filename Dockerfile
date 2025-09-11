# Dockerfile for LLM Compressor
# Multi-agent system for LLM compression and optimization

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional ML libraries
RUN pip install \
    transformers>=4.36.0 \
    accelerate \
    datasets \
    auto-gptq[triton] \
    autoawq \
    bitsandbytes \
    flash-attn --no-build-isolation

# Install vLLM (optional, comment out if not needed)
RUN pip install vllm

# Install monitoring and visualization libraries
RUN pip install \
    plotly \
    pynvml \
    GPUtil \
    psutil \
    scikit-learn \
    pandas \
    numpy \
    pyyaml

# Install development tools
RUN pip install \
    pytest \
    black \
    isort \
    flake8 \
    mypy

# Copy the project files
COPY llm_compressor/ ./llm_compressor/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
COPY Makefile .
COPY *.md .

# Create necessary directories
RUN mkdir -p reports artifacts logs

# Set permissions for scripts
RUN chmod +x scripts/*.py scripts/*.sh

# Create a non-root user
RUN useradd -m -u 1000 llmuser && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

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