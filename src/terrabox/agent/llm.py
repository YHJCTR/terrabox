"""LLM factory: returns a LangChain ChatOpenAI instance for local or remote mode."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from .config import AgentConfig

logger = logging.getLogger(__name__)


def get_llm(config: AgentConfig) -> ChatOpenAI:
    """Return a ChatOpenAI instance for local (subprocess/Docker) or remote API."""
    if config.use_local_llm:
        if config.use_docker:
            # Docker container mode
            from ..managers.docker.agent_llm_manager import AgentLLMDockerManager
            AgentLLMDockerManager.start_service(config)
            api_base = AgentLLMDockerManager._api_base()
            logger.info("Using local LLM (Docker container)")
        else:
            # Subprocess mode — uses the local unsloth conda env
            from ..managers.agent_llm_manager import AgentLLMServiceManager
            AgentLLMServiceManager.start_service(config)
            api_base = f"http://{config.local_llm_host}:{config.local_llm_port}/v1"
            logger.info("Using local LLM (subprocess)")

        # Docker mode mounts the model at /model inside the container, so vLLM
        # serves it as "/model". Subprocess mode serves the model under its host
        # path. Use the correct name accordingly.
        model_name = "/model" if config.use_docker else config.local_llm_model_path
        return ChatOpenAI(
            base_url=api_base,
            api_key="EMPTY",   # vLLM does not require a real key
            model=model_name,
            temperature=0.7,
            streaming=True,
            http_client=httpx.Client(
                trust_env=False,
                transport=httpx.HTTPTransport(retries=3),
            ),
            http_async_client=httpx.AsyncClient(
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=3),
            ),
        )
    else:
        logger.info(f"Using remote LLM: {config.remote_llm_model} via {config.remote_llm_api_base}")
        return ChatOpenAI(
            base_url=config.remote_llm_api_base,
            api_key=config.remote_llm_api_key,
            model=config.remote_llm_model,
            temperature=0.7,
            streaming=True,    # enable SSE token streaming from remote API
        )


def call_llm_json(system: str, user: str) -> Any:
    """Load config, call LLM, parse response as JSON.

    Shared by intent classification and memory extraction to avoid repeating
    load_config() + get_llm() + json.loads() boilerplate.
    Raises on LLM or JSON parse failure — callers should catch and fallback.
    """
    from .config import load_config
    config = load_config()
    llm = get_llm(config)
    result = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return json.loads(result.content)
