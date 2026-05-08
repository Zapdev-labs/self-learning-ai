"""Progressive curriculum generator for self-learning."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..core.types import CurriculumStep, Task, TaskDifficulty, TaskType

logger = logging.getLogger(__name__)


# Built-in curriculum of ML/AI + Next.js tasks
DEFAULT_CURRICULUM: List[CurriculumStep] = [
    # Python basics
    CurriculumStep(1, TaskType.CODE_GENERATION, TaskDifficulty.BEGINNER,
        "Write a Python function that takes a list of numbers and returns the mean and standard deviation."),
    CurriculumStep(2, TaskType.CODE_GENERATION, TaskDifficulty.BEGINNER,
        "Write a Python class `DataLoader` that reads a CSV file and provides batches of rows."),
    CurriculumStep(3, TaskType.TEST_WRITING, TaskDifficulty.BEGINNER,
        "Write pytest unit tests for a `Calculator` class with add, subtract, multiply, divide methods.", prerequisites=[1]),
    
    # ML fundamentals
    CurriculumStep(4, TaskType.CODE_GENERATION, TaskDifficulty.INTERMEDIATE,
        "Implement a simple linear regression from scratch using only NumPy. Include training via gradient descent.", prerequisites=[1, 2]),
    CurriculumStep(5, TaskType.CODE_GENERATION, TaskDifficulty.INTERMEDIATE,
        "Implement a 2-layer neural network classifier for the Iris dataset using PyTorch. Include training loop and accuracy metric.", prerequisites=[4]),
    CurriculumStep(6, TaskType.ML_MODEL, TaskDifficulty.INTERMEDIATE,
        "Build a text tokenizer that splits text into subword tokens using a simple BPE algorithm.", prerequisites=[2]),
    
    # Data pipelines
    CurriculumStep(7, TaskType.DATA_PIPELINE, TaskDifficulty.INTERMEDIATE,
        "Create a data pipeline that downloads the IMDB dataset from HuggingFace, tokenizes reviews, and creates a PyTorch DataLoader.", prerequisites=[5, 6]),
    
    # Bug fixing
    CurriculumStep(8, TaskType.BUG_FIX, TaskDifficulty.INTERMEDIATE,
        "Fix the following code: a training loop that has a gradient accumulation bug causing incorrect loss scaling.", prerequisites=[5]),
    CurriculumStep(9, TaskType.BUG_FIX, TaskDifficulty.INTERMEDIATE,
        "Fix a PyTorch model that crashes on GPU due to tensor device mismatches.", prerequisites=[5]),
    
    # Refactoring
    CurriculumStep(10, TaskType.REFACTOR, TaskDifficulty.INTERMEDIATE,
        "Refactor a messy 200-line training script into clean modules: model.py, train.py, config.py, utils.py.", prerequisites=[5, 7]),
    
    # Next.js basics
    CurriculumStep(11, TaskType.NEXTJS_APP, TaskDifficulty.BEGINNER,
        "Create a Next.js 14 app with a single page that fetches data from an API and displays it in a responsive grid.", prerequisites=[1]),
    CurriculumStep(12, TaskType.NEXTJS_APP, TaskDifficulty.INTERMEDIATE,
        "Build a Next.js dashboard with React Server Components, Tailwind CSS, and a dark mode toggle.", prerequisites=[11]),
    
    # Advanced ML
    CurriculumStep(13, TaskType.ML_MODEL, TaskDifficulty.ADVANCED,
        "Implement a transformer encoder block from scratch with multi-head self-attention and feed-forward layers.", prerequisites=[5, 6]),
    CurriculumStep(14, TaskType.DATA_PIPELINE, TaskDifficulty.ADVANCED,
        "Create a distributed data loading pipeline that shards a large dataset across multiple workers.", prerequisites=[7]),
    
    # Complex Next.js
    CurriculumStep(15, TaskType.NEXTJS_APP, TaskDifficulty.ADVANCED,
        "Build a full-stack Next.js app with authentication (JWT), a PostgreSQL API route, and real-time updates via Server-Sent Events.", prerequisites=[12]),
    
    # Exploration / open-ended
    CurriculumStep(16, TaskType.EXPLORATION, TaskDifficulty.EXPERT,
        "Research and implement a novel attention mechanism variant, benchmark it against standard attention on a toy task.", prerequisites=[13]),
]


class Curriculum:
    """Manages progressive learning tasks."""

    def __init__(self, steps: Optional[List[CurriculumStep]] = None):
        self.steps = steps or DEFAULT_CURRICULUM
        self._completed: set = set()
        self._current_index = 0

    def next_task(self) -> Optional[Task]:
        """Get the next available task based on prerequisites."""
        for step in self.steps[self._current_index:]:
            if step.step_number in self._completed:
                continue
            prereqs_met = all(p in self._completed for p in step.prerequisites)
            if not prereqs_met:
                logger.debug(f"Prerequisites not met for step {step.step_number}")
                continue
            return self._step_to_task(step)
        return None

    def mark_complete(self, step_number: int) -> None:
        self._completed.add(step_number)
        if self._current_index < len(self.steps) and self.steps[self._current_index].step_number == step_number:
            self._current_index += 1

    def get_progress(self) -> Dict[str, Any]:
        total = len(self.steps)
        completed = len(self._completed)
        return {
            "total_steps": total,
            "completed": completed,
            "remaining": total - completed,
            "percent": round(100 * completed / total, 1) if total else 0,
            "current_step": self._current_index + 1,
        }

    def add_custom_step(
        self,
        description: str,
        task_type: TaskType,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        prerequisites: Optional[List[int]] = None,
    ) -> CurriculumStep:
        next_num = max(s.step_number for s in self.steps) + 1 if self.steps else 1
        step = CurriculumStep(
            step_number=next_num,
            task_type=task_type,
            difficulty=difficulty,
            description_template=description,
            prerequisites=prerequisites or [],
        )
        self.steps.append(step)
        self.steps.sort(key=lambda s: s.step_number)
        return step

    def _step_to_task(self, step: CurriculumStep) -> Task:
        return Task(
            id=uuid.uuid4().hex,
            description=step.description_template,
            task_type=step.task_type,
            difficulty=step.difficulty,
            tags=[f"curriculum_step_{step.step_number}"],
            source="curriculum",
        )
