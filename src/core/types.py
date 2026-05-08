"""Core type definitions for ASSBRAIN."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    TEST_WRITING = "test_writing"
    NEXTJS_APP = "nextjs_app"
    ML_MODEL = "ml_model"
    DATA_PIPELINE = "data_pipeline"
    EXPLORATION = "exploration"


class TaskDifficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SolutionStatus(str, Enum):
    GENERATED = "generated"
    BUILDING = "building"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    FIXED = "fixed"
    ABANDONED = "abandoned"


@dataclass
class Task:
    """A learning task for the agent."""
    id: str
    description: str
    task_type: TaskType
    difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    test_cases: List[str] = field(default_factory=list)
    max_attempts: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    source: str = "curriculum"  # curriculum | self_discovered | user | mistake


@dataclass
class Solution:
    """A generated solution attempt."""
    id: str
    task_id: str
    attempt_number: int
    code: str
    language: str = "python"
    files: Dict[str, str] = field(default_factory=dict)
    build_commands: List[str] = field(default_factory=list)
    status: SolutionStatus = SolutionStatus.GENERATED
    critique: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a solution."""
    solution_id: str
    task_id: str
    success: bool
    score: float  # 0.0 - 1.0
    test_passed: int = 0
    test_failed: int = 0
    lint_score: float = 0.0
    type_check_passed: bool = False
    build_success: bool = False
    runtime_errors: List[str] = field(default_factory=list)
    compile_errors: List[str] = field(default_factory=list)
    test_output: str = ""
    browser_screenshot_path: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Experience:
    """A stored learning experience for replay / fine-tuning."""
    id: str
    task: Task
    solution: Solution
    evaluation: EvaluationResult
    reward: float
    lesson: str = ""  # what the agent learned from this
    embedding_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    used_for_training: bool = False


@dataclass
class CurriculumStep:
    """One step in the learning curriculum."""
    step_number: int
    task_type: TaskType
    difficulty: TaskDifficulty
    description_template: str
    prerequisites: List[int] = field(default_factory=list)
    success_threshold: float = 0.8
    required_consecutive_passes: int = 2


class AgentState(str, Enum):
    IDLE = "idle"
    GENERATING = "generating"
    BUILDING = "building"
    TESTING = "testing"
    LEARNING = "learning"
    TRAINING = "training"
    REFLECTING = "reflecting"
