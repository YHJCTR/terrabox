#!/bin/bash
# 启动项目的agent llm服务（每个GPU一个容器）

echo "Starting agent LLM services..."

# Part 1: GPU0 + GPU1
echo "【Part 1】Starting LLM on GPU 0 (port 9100)..."
docker run -d \
  --gpus '"device=0"' \
  -p 9100:9100 \
  --name agent_llm_0 \
  terrabox-agent-llm:unsloth \
  > /dev/null 2>&1

# Part 2: GPU2 + GPU3  
echo "【Part 2】Starting LLM on GPU 2 (port 9102)..."
docker run -d \
  --gpus '"device=2"' \
  -p 9102:9102 \
  --name agent_llm_2 \
  terrabox-agent-llm:unsloth \
  > /dev/null 2>&1

sleep 3

# 检查服务
echo ""
echo "Checking services..."
for port in 9100 9102; do
  if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
    echo "✅ Port $port: Ready"
  else
    echo "⏳ Port $port: Starting..."
  fi
done

echo "✅ Agent LLM services started"
