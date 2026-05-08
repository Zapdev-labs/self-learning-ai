"""Episodic and semantic memory using vector store + local DB."""

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..core.config import Config
from ..core.types import Experience, Task

logger = logging.getLogger(__name__)


class MemoryStore:
    """Stores and retrieves experiences with semantic search."""

    def __init__(self, config: Config):
        self.cfg = config.memory
        self.dir = config.memory_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        # SQLite for structured metadata
        self.db_path = self.dir / "experiences.db"
        self._init_sqlite()

        # Chroma for vector search
        chroma_dir = str(self.dir / "chroma")
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=chroma_dir,
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.get("collection_name", "assbrain_memory"),
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding model
        emb_model = self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(emb_model)

        self.similarity_threshold = self.cfg.get("similarity_threshold", 0.75)
        self.max_context_memories = self.cfg.get("max_context_memories", 5)

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    task_type TEXT,
                    difficulty TEXT,
                    solution_id TEXT,
                    reward REAL,
                    lesson TEXT,
                    tags TEXT,
                    used_for_training INTEGER DEFAULT 0,
                    created_at TEXT,
                    raw_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    task_type TEXT,
                    difficulty TEXT,
                    status TEXT,
                    created_at TEXT,
                    raw_json TEXT
                )
                """
            )
            conn.commit()

    def add_experience(self, exp: Experience) -> str:
        """Store an experience with embedding. Returns the id."""
        exp_id = exp.id or uuid.uuid4().hex
        exp.id = exp_id

        # Build embedding text from task + solution + lesson
        embed_text = (
            f"Task: {exp.task.description}. "
            f"Type: {exp.task.task_type.value}. "
            f"Solution: {exp.solution.code[:500]}. "
            f"Lesson: {exp.lesson}"
        )
        embedding = self.embedder.encode(embed_text).tolist()

        # Add to Chroma
        self.collection.add(
            ids=[exp_id],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[
                {
                    "task_id": exp.task.id,
                    "task_type": exp.task.task_type.value,
                    "difficulty": exp.task.difficulty.value,
                    "reward": exp.reward,
                    "success": exp.evaluation.success,
                    "tags": ",".join(exp.tags),
                }
            ],
        )

        # Add to SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiences
                (id, task_id, task_type, difficulty, solution_id, reward, lesson, tags, used_for_training, created_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    exp.task.id,
                    exp.task.task_type.value,
                    exp.task.difficulty.value,
                    exp.solution.id,
                    exp.reward,
                    exp.lesson,
                    ",".join(exp.tags),
                    int(exp.used_for_training),
                    exp.created_at.isoformat(),
                    json.dumps(exp, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)),
                ),
            )
            conn.commit()

        exp.embedding_id = exp_id
        logger.info(f"Stored experience {exp_id} with reward {exp.reward:.2f}")
        return exp_id

    def search(
        self,
        query: str,
        task_type: Optional[str] = None,
        min_reward: Optional[float] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search over experiences."""
        embedding = self.embedder.encode(query).tolist()
        where_filter = {}
        if task_type:
            where_filter["task_type"] = task_type
        if min_reward is not None:
            where_filter["reward"] = {"$gte": min_reward}

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where_filter if where_filter else None,
        )

        out = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for idx, eid in enumerate(ids):
            sim = 1 - distances[idx] if distances else 0
            if sim < self.similarity_threshold:
                continue
            out.append(
                {
                    "id": eid,
                    "similarity": sim,
                    "metadata": metadatas[idx] if metadatas else {},
                    "document": documents[idx] if documents else "",
                }
            )
        return sorted(out, key=lambda x: x["similarity"], reverse=True)

    def get_similar_successes(self, task: Task, limit: int = 3) -> List[Dict[str, Any]]:
        """Find successful past experiences similar to a given task."""
        return self.search(
            query=task.description,
            task_type=task.task_type.value,
            min_reward=0.7,
            limit=limit,
        )

    def get_training_batch(self, batch_size: int = 32) -> List[Dict[str, str]]:
        """Get high-reward experiences formatted for training."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT raw_json FROM experiences
                WHERE reward >= 0.7 AND used_for_training = 0
                ORDER BY reward DESC, created_at DESC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()

        batch = []
        for row in rows:
            raw = json.loads(row["raw_json"])
            # Format as prompt-response pairs
            task_desc = raw.get("task", {}).get("description", "")
            code = raw.get("solution", {}).get("code", "")
            lesson = raw.get("lesson", "")
            prompt = f"Task: {task_desc}\n\nPrevious mistakes and lessons: {lesson}\n\nWrite a solution:"
            batch.append({"prompt": prompt, "chosen": code})

        # Mark as used
        if rows:
            ids = [json.loads(r["raw_json"]).get("id") for r in rows]
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "UPDATE experiences SET used_for_training = 1 WHERE id = ?",
                    [(i,) for i in ids if i],
                )
                conn.commit()

        return batch

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            avg_reward = conn.execute("SELECT AVG(reward) FROM experiences").fetchone()[0]
            untrained = conn.execute(
                "SELECT COUNT(*) FROM experiences WHERE used_for_training = 0 AND reward >= 0.7"
            ).fetchone()[0]
        return {
            "total_experiences": total,
            "avg_reward": round(avg_reward or 0, 3),
            "untrained_high_reward": untrained,
            "vector_count": self.collection.count(),
        }

    def store_task(self, task: Task, status: str = "pending") -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (id, description, task_type, difficulty, status, created_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.description,
                    task.task_type.value,
                    task.difficulty.value,
                    status,
                    task.created_at.isoformat(),
                    json.dumps(task, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)),
                ),
            )
            conn.commit()
        return task.id
