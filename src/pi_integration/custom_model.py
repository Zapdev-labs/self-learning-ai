"""Custom model provider that registers ASSBRAIN as a backend in pi.

Usage in pi config:
  models:
    assbrain-local:
      provider: custom
      module: src.pi_integration.custom_model
      class: CustomModelProvider
      config:
        model_path: ./models/assbrain
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..core.config import Config
from ..core.llm_engine import LLMEngine
from .pi_adapter import PiAdapter

logger = logging.getLogger(__name__)


class CustomModelProvider:
    """
    Pi-compatible model provider interface.
    Can be loaded dynamically by pi's model registry.
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_config = config
        self._llm: Optional[LLMEngine] = None
        self._adapter: Optional[PiAdapter] = None

    async def initialize(self) -> None:
        """Load the local model and initialize adapter."""
        cfg = Config(self.model_config)
        self._llm = LLMEngine(cfg)
        # Memory + adapter initialized lazily
        from ..learning.memory_store import MemoryStore
        memory = MemoryStore(cfg)
        self._adapter = PiAdapter(cfg, self._llm, memory)
        logger.info("CustomModelProvider initialized with local LLM")

    async def complete(self, prompt: str, **kwargs) -> str:
        if not self._llm:
            raise RuntimeError("Provider not initialized")
        return self._llm.generate(prompt, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self._adapter:
            raise RuntimeError("Provider not initialized")
        return await self._adapter.chat(messages, **kwargs)

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        if not self._adapter:
            raise RuntimeError("Provider not initialized")
        async for token in self._adapter.stream_chat(messages, **kwargs):
            yield token

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_config.get("model_id", "assbrain-local"),
            "provider": "assbrain-local",
            "context_window": self.model_config.get("context_window", 4096),
            "supports_streaming": True,
            "supports_functions": False,
        }
