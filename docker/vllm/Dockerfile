FROM vllm/vllm-openai:latest

RUN pip install --upgrade \
    vllm==0.12.0 \
    transformers==4.57.3 \
    qwen-vl-utils==0.0.14 \
    tokenizers==0.22.1 \
    huggingface-hub==0.36.0

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/model", \
     "--trust-remote-code", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "2", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.8", \
     "--enforce-eager"]
