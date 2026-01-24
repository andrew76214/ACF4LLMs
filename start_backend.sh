#!/bin/bash

# Green AI Backend Startup Script
# ================================

set -e

cd /mnt/ssd_1TB/Green_AI

echo "=========================================="
echo "  Green AI - Backend Startup"
echo "=========================================="

# Activate conda environment
echo "[1/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate greenai

# Check for OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "[!] OPENAI_API_KEY not set. Please set it:"
    echo "    export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    read -p "Enter your OpenAI API Key: " api_key
    export OPENAI_API_KEY="$api_key"
fi

# Set CORS for Cloudflare Pages
export ALLOWED_ORIGINS="https://acf4llms.pages.dev,http://localhost:5173,http://localhost:3000"

echo "[2/4] Environment configured"
echo "  - OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
echo "  - ALLOWED_ORIGINS: $ALLOWED_ORIGINS"

# Start FastAPI backend
echo "[3/4] Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[!] Backend failed to start"
    exit 1
fi

echo "  - Backend running (PID: $BACKEND_PID)"
echo "  - Local: http://localhost:8000"
echo "  - Docs:  http://localhost:8000/docs"

# Start Cloudflare Tunnel
echo "[4/4] Starting Cloudflare Tunnel..."
echo ""
echo "=========================================="
echo "  Copy the tunnel URL below and set it"
echo "  as VITE_API_URL in Cloudflare Pages"
echo "=========================================="
echo ""

cloudflared tunnel --url http://localhost:8000

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
