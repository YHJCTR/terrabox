#!/bin/bash

echo "Starting agent LLM services (using default container entrypoint)..."

# 停止任何现有容器
docker ps -q | xargs -r docker stop 2>/dev/null || true

# Part 1: GPU0
echo "【Part 1】Starting LLM on GPU 0..."
docker run -d \
  --gpus '"device=0"' \
  -p 9100:8000 \
  --name agent_llm_0 \
  terrabox/agent-llm:latest \
  2>&1 | grep -E "error|Error" || echo "✅ Started"

# Part 2: GPU2
echo "【Part 2】Starting LLM on GPU 2..."
docker run -d \
  --gpus '"device=2"' \
  -p 9102:8000 \
  --name agent_llm_2 \
  terrabox/agent-llm:latest \
  2>&1 | grep -E "error|Error" || echo "✅ Started"

echo ""
echo "✅ Services started in background"
echo "   Waiting for initialization..."

