#!/bin/bash
# 为4张GPU启动vLLM服务

echo "Starting LLM services on all 4 GPUs..."

# GPU 0 -> Port 9100
echo "Starting LLM on GPU 0 (port 9100)..."
docker run -d \
  --gpus '"device=0"' \
  -p 9100:8000 \
  --name llm_gpu0 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  > /dev/null 2>&1 &

# GPU 1 -> Port 9101
echo "Starting LLM on GPU 1 (port 9101)..."
docker run -d \
  --gpus '"device=1"' \
  -p 9101:8000 \
  --name llm_gpu1 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  > /dev/null 2>&1 &

# GPU 2 -> Port 9102
echo "Starting LLM on GPU 2 (port 9102)..."
docker run -d \
  --gpus '"device=2"' \
  -p 9102:8000 \
  --name llm_gpu2 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  > /dev/null 2>&1 &

# GPU 3 -> Port 9103
echo "Starting LLM on GPU 3 (port 9103)..."
docker run -d \
  --gpus '"device=3"' \
  -p 9103:8000 \
  --name llm_gpu3 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  > /dev/null 2>&1 &

echo "⏳ Waiting for services to start..."
sleep 10

# 检查服务状态
echo ""
echo "Checking service status:"
for port in 9100 9101 9102 9103; do
  if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
    echo "✅ Port $port: Ready"
  else
    echo "⏳ Port $port: Still starting..."
  fi
done

echo ""
echo "✅ LLM services started"
echo "   GPU 0 (Part 1 LLM):  http://localhost:9100"
echo "   GPU 1 (Part 1 Tools): GPU 1"
echo "   GPU 2 (Part 2 LLM):  http://localhost:9102"  
echo "   GPU 3 (Part 2 Tools): GPU 3"

