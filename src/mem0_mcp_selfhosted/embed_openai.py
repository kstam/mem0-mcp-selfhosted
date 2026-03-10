"""Custom OpenAI embedding provider with gateway header support.

Mirrors mem0ai's built-in OpenAIEmbedding but adds ``default_headers``
to the OpenAI client — required for corporate LLM gateways that need
extra headers (e.g. ``x-wise-llm-gateway-team``).

Includes automatic 401 recovery: when the gateway JWT expires, the client
is rebuilt with a fresh token from ``resolve_token(invalidate_cache=True)``.

Registered as ``"openai"`` via ``EmbedderFactory.provider_to_class``
override, replacing the built-in provider.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

import openai
from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

from mem0_mcp_selfhosted.auth import resolve_token
from mem0_mcp_selfhosted.env import opt_env, parse_custom_headers

logger = logging.getLogger(__name__)


class GatewayOpenAIEmbedding(EmbeddingBase):
    """OpenAI embedding provider with gateway header injection."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        self._base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        # Gateway headers — parsed from ANTHROPIC_CUSTOM_HEADERS (same env
        # var used by the Anthropic LLM provider for corporate gateways).
        self._default_headers: dict[str, str] | None = None
        custom_headers_raw = opt_env("ANTHROPIC_CUSTOM_HEADERS")
        if custom_headers_raw:
            self._default_headers = parse_custom_headers(custom_headers_raw)
            logger.info(
                "OpenAI embedder using gateway headers: %s",
                list(self._default_headers.keys()),
            )

        api_key = (
            self.config.api_key
            or os.getenv("OPENAI_API_KEY")
        )
        self._build_client(api_key)

    def _build_client(self, api_key: str | None) -> None:
        """Build (or rebuild) the OpenAI client with the given API key."""
        self._current_api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            default_headers=self._default_headers,
        )

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> list[float]:
        text = text.replace("\n", " ")
        try:
            return self._do_embed(text)
        except openai.AuthenticationError:
            # Gateway JWT likely expired — refresh and retry once
            new_token = resolve_token(invalidate_cache=True)
            if new_token:
                logger.info("[mem0] Embedder token expired, refreshed via helper script")
                self._build_client(new_token)
                return self._do_embed(text)
            raise

    def _do_embed(self, text: str) -> list[float]:
        return (
            self.client.embeddings.create(
                input=[text],
                model=self.config.model,
                dimensions=self.config.embedding_dims,
            )
            .data[0]
            .embedding
        )
