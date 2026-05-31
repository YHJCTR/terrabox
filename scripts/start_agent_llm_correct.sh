#!/bin/bash

echo "Starting agent LLM services with correct image..."

# Part 1: GPU0
echo "【Part 1】Starting LLM on GPU 0 (port 9100)..."
docker run -d \
  --gpus '"device=0"' \
  -p 9100:9100 \
  --name agent_llm_0 \
  terrabox/agent-llm:latest \
  > /dev/null 2>&1

# Part 2: GPU2
echo "【Part 2】Starting LLM on GPU 2 (port 9102)..."
docker run -d \
  --gpus '"device=2"' \
  -p 9102:9102 \
  --name agent_llm_2 \
  terrabox/agent-llm:latest \
  > /dev/null 2>&1

echo "⏳ Waiting for services to start..."
sleep 5

# 检查
echo ""
echo "✅ Agent LLM services started"
echo "   Part 1 (GPU 0): http://localhost:9100"
echo "   Part 2 (GPU 2): http://localhost:9102"

