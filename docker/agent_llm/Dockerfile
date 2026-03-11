FROM vllm/vllm-openai:latest

# Install / upgrade packages needed for Qwen3 text models.
# No qwen-vl-utils needed (this is a pure text model, not a vision model).
RUN pip install --upgrade \
    vllm==0.12.0 \
    transformers==4.57.3 \
    tokenizers==0.22.1 \
    huggingface-hub==0.36.0

# Default CMD — overridden at runtime by AgentLLMDockerManager
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/model", \
     "--trust-remote-code", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.85", \
     "--enforce-eager"]
