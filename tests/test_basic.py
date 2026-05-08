"""Basic smoke tests for ASSBRAIN imports and config."""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """All core modules should import cleanly."""
    from core.config import load_config
    from core.types import Task, TaskType, Solution, EvaluationResult, Experience
    from learning.memory_store import MemoryStore
    from learning.feedback_loop import FeedbackLoop
    from learning.self_rl_trainer import SelfRLTrainer
    from learning.curriculum import Curriculum
    from generation.code_generator import CodeGenerator
    from generation.project_scaffold import ProjectScaffold
    from generation.nextjs_builder import NextJSBuilder
    from evaluation.code_executor import CodeExecutor
    from evaluation.test_runner import TestRunner
    from evaluation.linter import Linter
    from evaluation.browser_tester import BrowserTester
    from evaluation.evaluator import Evaluator
    from tools.browser_tool import BrowserTool
    from tools.mcp_client import MCPClient
    from tools.lsp_client import LSPClient
    from tools.huggingface_loader import HuggingFaceLoader
    from pi_integration.pi_adapter import PiAdapter
    from pi_integration.custom_model import CustomModelProvider


def test_config_loading():
    """Config should load from default.yaml."""
    from core.config import load_config
    cfg = load_config("./config/default.yaml")
    assert cfg.get("app.name") == "ASSBRAIN"
    assert cfg.get("llm.model_id") is not None
    assert cfg.data_dir.exists()


def test_curriculum_steps():
    """Curriculum should have default steps."""
    from learning.curriculum import Curriculum
    c = Curriculum()
    assert len(c.steps) > 0
    progress = c.get_progress()
    assert progress["total_steps"] > 0
