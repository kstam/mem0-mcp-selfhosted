"""Tests for embed_openai.py — GatewayOpenAIEmbedding with gateway headers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGatewayOpenAIEmbedding:
    """Unit tests for the custom OpenAI embedder."""

    # Keys that leak from the real env and affect gateway detection
    _LEAK_KEYS = ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_API_KEY", "OPENAI_BASE_URL")

    def _make_embedder(self, *, env: dict | None = None, config_overrides: dict | None = None):
        """Create a GatewayOpenAIEmbedding with mocked OpenAI client."""
        from mem0.configs.embeddings.base import BaseEmbedderConfig

        config_kwargs = {
            "api_key": "test-key",
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        }
        if config_overrides:
            config_kwargs.update(config_overrides)

        config = BaseEmbedderConfig(**config_kwargs)

        env = env or {}
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in self._LEAK_KEYS:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.embed_openai.OpenAI") as mock_openai_cls:
                from mem0_mcp_selfhosted.embed_openai import GatewayOpenAIEmbedding

                embedder = GatewayOpenAIEmbedding(config)
                return embedder, mock_openai_cls

    def test_basic_client_construction(self):
        """Client constructed with api_key and default base_url."""
        embedder, mock_cls = self._make_embedder()

        mock_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            default_headers=None,
        )

    def test_custom_base_url(self):
        """openai_base_url from config flows to OpenAI client."""
        embedder, mock_cls = self._make_embedder(
            config_overrides={"openai_base_url": "https://gateway.example.com/v1"}
        )

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://gateway.example.com/v1"

    def test_gateway_headers_injected(self):
        """ANTHROPIC_CUSTOM_HEADERS parsed and passed as default_headers."""
        env = {
            "ANTHROPIC_CUSTOM_HEADERS": (
                "x-wise-llm-gateway-team: global-product\n"
                "x-wise-llm-gateway-use-case: claude-code"
            ),
        }
        embedder, mock_cls = self._make_embedder(env=env)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["default_headers"] == {
            "x-wise-llm-gateway-team": "global-product",
            "x-wise-llm-gateway-use-case": "claude-code",
        }

    def test_no_custom_headers(self):
        """Without ANTHROPIC_CUSTOM_HEADERS, default_headers is None."""
        embedder, mock_cls = self._make_embedder()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["default_headers"] is None

    def test_model_defaults(self):
        """Default model and dims when not specified in config."""
        from mem0.configs.embeddings.base import BaseEmbedderConfig

        config = BaseEmbedderConfig(api_key="test-key")

        with patch("mem0_mcp_selfhosted.embed_openai.OpenAI"):
            from mem0_mcp_selfhosted.embed_openai import GatewayOpenAIEmbedding

            embedder = GatewayOpenAIEmbedding(config)
            assert embedder.config.model == "text-embedding-3-small"
            assert embedder.config.embedding_dims == 1536

    def test_embed_calls_openai_api(self):
        """embed() calls client.embeddings.create with correct params."""
        embedder, mock_cls = self._make_embedder()

        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        result = embedder.embed("hello world")

        mock_client.embeddings.create.assert_called_once_with(
            input=["hello world"],
            model="text-embedding-3-small",
            dimensions=1536,
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_strips_newlines(self):
        """embed() replaces newlines with spaces before embedding."""
        embedder, mock_cls = self._make_embedder()

        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_client.embeddings.create.return_value = mock_response

        embedder.embed("line1\nline2\nline3")

        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]["input"] == ["line1 line2 line3"]

    def test_openai_base_url_env_fallback(self):
        """OPENAI_BASE_URL env var used when config.openai_base_url is None."""
        env = {"OPENAI_BASE_URL": "https://env-gateway.example.com/v1"}
        embedder, mock_cls = self._make_embedder(env=env)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://env-gateway.example.com/v1"

    def test_openai_api_key_env_fallback(self):
        """OPENAI_API_KEY env var used when config.api_key is None."""
        from mem0.configs.embeddings.base import BaseEmbedderConfig

        config = BaseEmbedderConfig(model="text-embedding-3-small", embedding_dims=1536)

        env = {"OPENAI_API_KEY": "sk-from-env"}
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in self._LEAK_KEYS:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.embed_openai.OpenAI") as mock_cls:
                from mem0_mcp_selfhosted.embed_openai import GatewayOpenAIEmbedding

                embedder = GatewayOpenAIEmbedding(config)

                call_kwargs = mock_cls.call_args[1]
                assert call_kwargs["api_key"] == "sk-from-env"
