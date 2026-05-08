"""Self-play reinforcement learning via self-imitation + outcome reward."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.llm_engine import LLMEngine
from ..core.types import EvaluationResult, Experience, Solution, Task
from .feedback_loop import FeedbackLoop
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)


class SelfRLTrainer:
    """
    Orchestrates the self-learning loop:
    1. Generate solution from task
    2. Evaluate solution in sandbox
    3. Compute reward from outcome
    4. Self-critique on failure
    5. Retry with feedback
    6. Store experience in memory
    7. Periodically fine-tune on high-reward experiences
    """

    def __init__(
        self,
        config: Config,
        llm: LLMEngine,
        memory: MemoryStore,
        feedback: FeedbackLoop,
    ):
        self.cfg = config.learning
        self.llm = llm
        self.memory = memory
        self.feedback = feedback
        self.training_freq = self.cfg.get("training_frequency", 10)
        self.explore_rate = self.cfg.get("explore_vs_exploit", 0.3)
        self._experience_count = 0

    async def run_episode(
        self,
        task: Task,
        generator,
        evaluator,
    ) -> Experience:
        """Run one full learning episode for a task."""
        logger.info(f"🎯 Episode start: {task.id} — {task.description[:60]}...")
        self.memory.store_task(task, status="running")

        best_exp: Optional[Experience] = None
        best_reward = -1.0

        for attempt in range(1, task.max_attempts + 1):
            logger.info(f"  Attempt {attempt}/{task.max_attempts}")

            # 1. Generate solution
            solution = await generator.generate(task, attempt, best_exp)
            solution.attempt_number = attempt

            # 2. Evaluate
            result = await evaluator.evaluate(solution, task)
            result.task_id = task.id
            result.solution_id = solution.id

            # 3. Compute reward
            reward = self.feedback.compute_reward(result)
            result.reward = reward
            result.score = reward

            # 4. Self-critique
            critique, fix_hint = "", ""
            if self.cfg.get("self_critique_enabled", True):
                critique, fix_hint = self.feedback.generate_critique(task, solution, result)
                solution.critique = critique

            # 5. Build experience
            lesson = self._extract_lesson(critique, result)
            exp = Experience(
                id=uuid.uuid4().hex,
                task=task,
                solution=solution,
                evaluation=result,
                reward=reward,
                lesson=lesson,
                tags=task.tags + (["success"] if result.success else ["failure", f"attempt_{attempt}"]),
            )

            # Store immediately
            self.memory.add_experience(exp)
            self._experience_count += 1

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_exp = exp

            logger.info(f"  Reward: {reward:.3f} | Success: {result.success}")

            # 6. Decide whether to continue
            if not self.feedback.should_continue(attempt, reward):
                break

            # Explore: sometimes try a wildly different approach
            if attempt < task.max_attempts and self._should_explore():
                logger.info("  🎲 Exploring alternative approach...")

        if best_exp:
            self.memory.store_task(task, status="completed")
            logger.info(f"✅ Episode complete. Best reward: {best_reward:.3f}")

            # 7. Trigger training if enough new experiences
            if self._experience_count % self.training_freq == 0:
                await self._trigger_training()

            return best_exp

        raise RuntimeError("No experience generated")

    def _should_explore(self) -> bool:
        """Epsilon-greedy exploration."""
        import random
        return random.random() < self.explore_rate

    def _extract_lesson(self, critique: str, result: EvaluationResult) -> str:
        """Extract a concise lesson from critique and result."""
        if result.success:
            return "Success pattern: follow the same structure for similar tasks."
        if critique and len(critique) > 10:
            # Take first sentence as lesson
            first = critique.split(".")[0]
            return first[:200]
        if result.runtime_errors:
            return f"Runtime error: {result.runtime_errors[0][:100]}"
        if result.compile_errors:
            return f"Compile error: {result.compile_errors[0][:100]}"
        return "Unknown failure — needs more investigation."

    async def _trigger_training(self) -> Optional[str]:
        """Fine-tune LoRA on high-reward experiences."""
        stats = self.memory.get_stats()
        if stats["untrained_high_reward"] < 5:
            logger.info("Not enough high-reward experiences to train yet.")
            return None

        batch = self.memory.get_training_batch(batch_size=32)
        if len(batch) < 5:
            return None

        logger.info(f"🧠 Training on {len(batch)} experiences...")
        try:
            adapter_path = self.llm.train_on_experiences(batch)
            logger.info(f"🧠 Training complete. Adapter: {adapter_path}")
            return adapter_path
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None

    def get_curriculum_progress(self, curriculum_steps: List[Any]) -> Dict[str, Any]:
        """Report progress through curriculum."""
        # Query memory for success rate per task type
        stats = self.memory.get_stats()
        return {
            "total_experiences": stats["total_experiences"],
            "avg_reward": stats["avg_reward"],
            "episodes_since_last_train": self._experience_count % self.training_freq,
            "exploration_rate": self.explore_rate,
        }
