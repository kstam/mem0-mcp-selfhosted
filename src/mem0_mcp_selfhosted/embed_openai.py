"""Custom OpenAI embedding provider with gateway header support.

Mirrors mem0ai's built-in OpenAIEmbedding but adds ``default_headers``
to the OpenAI client — required for corporate LLM gateways that need
extra headers (e.g. ``x-wise-llm-gateway-team``).

Registered as ``"openai"`` via ``EmbedderFactory.provider_to_class``
override, replacing the built-in provider.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

from mem0_mcp_selfhosted.env import opt_env, parse_custom_headers

logger = logging.getLogger(__name__)


class GatewayOpenAIEmbedding(EmbeddingBase):
    """OpenAI embedding provider with gateway header injection."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = (
            self.config.api_key
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        # Gateway headers — parsed from ANTHROPIC_CUSTOM_HEADERS (same env
        # var used by the Anthropic LLM provider for corporate gateways).
        default_headers: dict[str, str] | None = None
        custom_headers_raw = opt_env("ANTHROPIC_CUSTOM_HEADERS")
        if custom_headers_raw:
            default_headers = parse_custom_headers(custom_headers_raw)
            logger.info(
                "OpenAI embedder using gateway headers: %s",
                list(default_headers.keys()),
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> list[float]:
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[text],
                model=self.config.model,
                dimensions=self.config.embedding_dims,
            )
            .data[0]
            .embedding
        )
