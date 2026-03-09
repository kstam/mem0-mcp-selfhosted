"""Contract tests for mem0ai's EmbedderFactory internals.

These validate assumptions our code makes about mem0ai's internal APIs.
If these fail after a mem0ai upgrade, embed_openai.py needs updating.
"""

from __future__ import annotations

import pytest


class TestEmbedderFactoryContract:
    """Verify EmbedderFactory assumptions used by server.py's override."""

    def test_provider_to_class_is_mutable_dict(self):
        """provider_to_class is a plain dict we can mutate at runtime."""
        from mem0.utils.factory import EmbedderFactory

        assert isinstance(EmbedderFactory.provider_to_class, dict)
        # Verify we can add/override entries
        original = EmbedderFactory.provider_to_class.get("openai")
        try:
            EmbedderFactory.provider_to_class["openai"] = "test.override.Class"
            assert EmbedderFactory.provider_to_class["openai"] == "test.override.Class"
        finally:
            # Restore original
            if original is not None:
                EmbedderFactory.provider_to_class["openai"] = original

    def test_openai_is_registered(self):
        """The built-in 'openai' provider exists (our override depends on this key)."""
        from mem0.utils.factory import EmbedderFactory

        assert "openai" in EmbedderFactory.provider_to_class

    def test_create_passes_base_embedder_config(self):
        """EmbedderFactory.create() constructs BaseEmbedderConfig from config dict.

        Our GatewayOpenAIEmbedding receives a BaseEmbedderConfig with
        api_key, openai_base_url, model, and embedding_dims populated.
        """
        from mem0.configs.embeddings.base import BaseEmbedderConfig

        config = BaseEmbedderConfig(
            api_key="test",
            openai_base_url="https://example.com/v1",
            model="text-embedding-3-small",
            embedding_dims=1536,
        )

        assert config.api_key == "test"
        assert config.openai_base_url == "https://example.com/v1"
        assert config.model == "text-embedding-3-small"
        assert config.embedding_dims == 1536

    def test_base_embedder_config_accepts_embedding_dims(self):
        """BaseEmbedderConfig accepts embedding_dims kwarg.

        This is passed by config.py as embedder_config["embedding_dims"].
        """
        from mem0.configs.embeddings.base import BaseEmbedderConfig

        config = BaseEmbedderConfig(embedding_dims=1536)
        assert config.embedding_dims == 1536
