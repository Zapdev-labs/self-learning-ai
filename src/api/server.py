"""FastAPI server for external access to ASSBRAIN."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..main import AssBrainOrchestrator

logger = logging.getLogger(__name__)

app: Optional[FastAPI] = None


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class GenerateRequest(BaseModel):
    description: str
    task_type: str = "code_generation"


class EpisodeResult(BaseModel):
    status: str
    reward: float = 0.0
    attempts: int = 0
    task_id: str = ""
    lesson: str = ""


def create_app(orch: AssBrainOrchestrator) -> FastAPI:
    global app
    app = FastAPI(title="ASSBRAIN API", version="0.1.0")

    @app.get("/")
    def root():
        return {"app": "ASSBRAIN", "version": "0.1.0", "status": orch.get_status()}

    @app.get("/status")
    def status():
        return orch.get_status()

    @app.post("/chat")
    async def chat(req: ChatRequest):
        try:
            response = await orch.pi_adapter.chat(req.messages, temperature=req.temperature, max_new_tokens=req.max_tokens)
            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        result = await orch.generate_for_task(req.description, req.task_type)
        return result

    @app.post("/learn")
    async def learn(episodes: int = 1):
        results = await orch.run_curriculum(max_episodes=episodes)
        return {"results": results}

    @app.post("/episode")
    async def episode():
        result = await orch.run_episode()
        return result

    @app.get("/memory/search")
    async def memory_search(query: str, limit: int = 5):
        results = orch.memory.search(query, limit=limit)
        return {"results": results}

    @app.get("/memory/stats")
    async def memory_stats():
        return orch.memory.get_stats()

    @app.post("/train")
    async def train():
        batch = orch.memory.get_training_batch(batch_size=32)
        if len(batch) < 5:
            return {"status": "skipped", "reason": "Not enough experiences"}
        adapter_path = orch.llm.train_on_experiences(batch)
        return {"status": "trained", "adapter_path": adapter_path}

    return app
