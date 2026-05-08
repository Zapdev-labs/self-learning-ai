"""Adapter that exposes ASSBRAIN as a custom model provider for the pi harness.

This allows pi to route requests to the local LLMEngine instead of cloud APIs.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..core.config import Config
from ..core.llm_engine import LLMEngine
from ..core.types import AgentState
from ..learning.memory_store import MemoryStore
from ..learning.self_rl_trainer import SelfRLTrainer

logger = logging.getLogger(__name__)


class PiAdapter:
    """
    Bridges ASSBRAIN's local LLM with the pi agent harness.
    Provides chat/completion endpoints and exposes agent state.
    """

    def __init__(
        self,
        config: Config,
        llm: LLMEngine,
        memory: MemoryStore,
        trainer: Optional[SelfRLTrainer] = None,
    ):
        self.cfg = config.pi
        self.llm = llm
        self.memory = memory
        self.trainer = trainer
        self.state = AgentState.IDLE
        self._stats: Dict[str, Any] = {}

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """OpenAI-style chat completion for pi."""
        self.state = AgentState.GENERATING
        try:
            # Retrieve relevant memories for context augmentation
            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break

            memory_context = ""
            if last_user_msg:
                similar = self.memory.search(last_user_msg, limit=2)
                if similar:
                    memory_context = "Relevant past experiences:\n"
                    for s in similar:
                        memory_context += f"- {s['document'][:200]}...\n"

            # Build messages with memory context
            augmented = messages.copy()
            if memory_context:
                augmented.insert(0, {
                    "role": "system",
                    "content": f"You are ASSBRAIN, a self-learning AI. {memory_context}",
                })

            response = self.llm.chat(augmented, **kwargs)
            self._stats["last_response_length"] = len(response)
            return response
        finally:
            self.state = AgentState.IDLE

    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Streaming chat for pi with real-time token output."""
        self.state = AgentState.GENERATING
        try:
            prompt_parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("Assistant:")
            full_prompt = "\n\n".join(prompt_parts)

            streamer = self.llm.generate(full_prompt, stream=True, **kwargs)
            for token in streamer:
                yield token
        finally:
            self.state = AgentState.IDLE

    async def get_status(self) -> Dict[str, Any]:
        """Return current agent status for pi dashboard."""
        mem_stats = self.memory.get_stats()
        status = {
            "model_id": self.llm.model_id,
            "device": self.llm.device,
            "lora_active": self.llm._lora_active,
            "state": self.state.value,
            "memory": mem_stats,
            "stats": self._stats,
        }
        if self.trainer:
            status["trainer"] = self.trainer.get_curriculum_progress([])
        return status

    async def run_self_learning_step(self) -> Dict[str, Any]:
        """Trigger one self-learning episode from pi."""
        if not self.trainer:
            return {"error": "Self-RL trainer not initialized"}
        from ..learning.curriculum import Curriculum
        from ..generation.code_generator import CodeGenerator
        from ..evaluation.evaluator import Evaluator

        # This is a simplified hook — normally the orchestrator wires these
        return {"message": "Use the orchestrator CLI to run learning episodes"}
