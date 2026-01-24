#!/bin/bash

# Green AI Backend - Production Startup Script
# =============================================
# 使用正式 Cloudflare Tunnel (固定 URL)

set -e

cd /mnt/ssd_1TB/Green_AI

echo "=========================================="
echo "  Green AI - Production Backend"
echo "=========================================="

# Activate conda environment
echo "[1/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate greenai

# Check for OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "[!] OPENAI_API_KEY not set."
    read -p "Enter your OpenAI API Key: " api_key
    export OPENAI_API_KEY="$api_key"
fi

# Set CORS for Cloudflare Pages
# 修改這裡加入你的域名
export ALLOWED_ORIGINS="https://acf4llms.rstltd.org,https://acf4llms-backend.rstltd.org"
# Allow Cloudflare Pages preview URLs (e.g., https://abc123.acf4llms.pages.dev)
export ALLOWED_ORIGIN_REGEX="https://.*\.acf4llms\.pages\.dev"
export ENVIRONMENT="production"

echo "[2/4] Environment configured"
echo "  - OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
echo "  - ALLOWED_ORIGINS: $ALLOWED_ORIGINS"
echo "  - ALLOWED_ORIGIN_REGEX: $ALLOWED_ORIGIN_REGEX"
echo "  - ENVIRONMENT: $ENVIRONMENT"

# Check if tunnel config exists
if [ ! -f ~/.cloudflared/config.yml ]; then
    echo ""
    echo "[!] Tunnel config not found at ~/.cloudflared/config.yml"
    echo "    Please run these commands first:"
    echo ""
    echo "    1. cloudflared tunnel login"
    echo "    2. cloudflared tunnel create greenai-api"
    echo "    3. cloudflared tunnel route dns greenai-api api.yourdomain.com"
    echo "    4. cp cloudflared-config.yml ~/.cloudflared/config.yml"
    echo "    5. Edit ~/.cloudflared/config.yml with your tunnel ID"
    echo ""
    exit 1
fi

# Start FastAPI backend
echo "[3/4] Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[!] Backend failed to start"
    exit 1
fi

echo "  - Backend running (PID: $BACKEND_PID)"

# Start Cloudflare Tunnel
echo "[4/4] Starting Cloudflare Tunnel (named tunnel)..."
echo ""
echo "=========================================="
echo "  Backend URL: https://acf4llms-backend.rstltd.org"
echo "=========================================="
echo ""

cloudflared tunnel run greenai-api

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
