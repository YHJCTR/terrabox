"""LLM factory: returns a LangChain ChatOpenAI instance for local or remote mode."""
from __future__ import annotations

import logging

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
            logger.info("Using local LLM (Docker container)")
        else:
            # Subprocess mode — uses the local unsloth conda env
            from ..managers.agent_llm_manager import AgentLLMServiceManager
            AgentLLMServiceManager.start_service(config)
            logger.info("Using local LLM (subprocess)")

        api_base = f"http://{config.local_llm_host}:{config.local_llm_port}/v1"
        # Non-Docker vLLM serves the model under its local path name;
        # Docker vLLM mounts the model at /model. We use the model path
        # so both modes resolve correctly.
        #
        # trust_env=False: bypass http_proxy/https_proxy env vars so that
        # requests to 127.0.0.1 go directly to vLLM instead of an external proxy.
        #
        # retries=3: when the agent calls a long-running tool (e.g., VLM loading
        # takes 10+ minutes), the vLLM server closes the idle HTTP keep-alive
        # connection (default timeout_keep_alive=5s). retries=3 makes httpx
        # transparently re-connect and retry on RemoteProtocolError / ConnectionError
        # so the next LLM call after a long tool execution succeeds.
        return ChatOpenAI(
            base_url=api_base,
            api_key="EMPTY",   # vLLM does not require a real key
            model=config.local_llm_model_path,
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
