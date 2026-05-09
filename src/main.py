"""ASSBRAIN Main Orchestrator — Self-learning AI agent loop."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .core.config import load_config
from .core.llm_engine import LLMEngine
from .core.types import AgentState, Task
from .evaluation.browser_tester import BrowserTester
from .evaluation.code_executor import CodeExecutor
from .evaluation.evaluator import Evaluator
from .evaluation.linter import Linter
from .evaluation.test_runner import TestRunner
from .generation.code_generator import CodeGenerator
from .generation.nextjs_builder import NextJSBuilder
from .generation.project_scaffold import ProjectScaffold
from .learning.curriculum import Curriculum
from .learning.feedback_loop import FeedbackLoop
from .learning.memory_store import MemoryStore
from .learning.self_rl_trainer import SelfRLTrainer
from .pi_integration.pi_adapter import PiAdapter
from .tools.browser_tool import BrowserTool
from .tools.huggingface_loader import HuggingFaceLoader
from .tools.mcp_client import MCPClient
from .core.gpu_monitor import GPUMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("assbrain")
console = Console()
app = typer.Typer(help="ASSBRAIN — Self-Learning AI Agent")


class AssBrainOrchestrator:
    """
    Central orchestrator that wires all components together
    and runs the self-learning loop.
    """

    def __init__(self, config_path: Optional[str] = None):
        console.print(Panel.fit("🧠 ASSBRAIN Initializing...", style="bold magenta"))

        self.config = load_config(config_path)
        self.state = AgentState.IDLE

        # Core
        self.llm = LLMEngine(self.config)
        self.memory = MemoryStore(self.config)

        # Generation
        self.generator = CodeGenerator(self.config, self.llm, self.memory)
        self.scaffold = ProjectScaffold(self.config)
        self.nextjs = NextJSBuilder(self.config, self.scaffold)

        # Evaluation
        self.executor = CodeExecutor(self.config)
        self.test_runner = TestRunner(self.config)
        self.linter = Linter(self.config)
        self.browser = BrowserTester(self.config)
        self.evaluator = Evaluator(
            self.config,
            self.executor,
            self.test_runner,
            self.linter,
            self.browser,
            self.nextjs,
            self.scaffold,
        )

        # Learning
        self.feedback = FeedbackLoop(self.config, self.llm)
        self.trainer = SelfRLTrainer(self.config, self.llm, self.memory, self.feedback)
        self.curriculum = Curriculum()

        # Tools
        self.mcp = MCPClient(self.config)
        self.browser_tool = BrowserTool(self.config)
        self.hf_loader = HuggingFaceLoader(self.config)

        # Pi
        self.pi_adapter = PiAdapter(self.config, self.llm, self.memory, self.trainer)

        self._episodes_completed = 0
        self.gpu_monitor = GPUMonitor(interval=2.0)
        console.print("[green]✓[/green] All components loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_episode(self, task: Optional[Task] = None) -> Dict[str, Any]:
        """Run one full learning episode."""
        if task is None:
            task = self.curriculum.next_task()
            if task is None:
                return {"status": "complete", "message": "All curriculum steps finished!"}

        console.print(f"\n[bold cyan]▶ Episode {self._episodes_completed + 1}[/bold cyan]: {task.description[:60]}...")

        self.state = AgentState.LEARNING
        try:
            exp = await self.trainer.run_episode(task, self.generator, self.evaluator)
            self._episodes_completed += 1

            # Mark curriculum progress
            if task.source == "curriculum":
                tag = next((t for t in task.tags if t.startswith("curriculum_step_")), None)
                if tag:
                    step_num = int(tag.split("_")[-1])
                    self.curriculum.mark_complete(step_num)

            color = "green" if exp.evaluation.success else "yellow"
            console.print(f"  [{color}]Reward: {exp.reward:.3f} | Success: {exp.evaluation.success}[/{color}]")

            return {
                "status": "success" if exp.evaluation.success else "partial",
                "reward": exp.reward,
                "attempts": exp.solution.attempt_number,
                "task_id": task.id,
                "lesson": exp.lesson,
            }
        except Exception as e:
            logger.exception("Episode failed")
            return {"status": "error", "message": str(e)}
        finally:
            self.state = AgentState.IDLE

    async def run_curriculum(self, max_episodes: int = 10) -> List[Dict[str, Any]]:
        """Run through curriculum episodes."""
        self.gpu_monitor.start()
        results = []
        try:
            for i in range(max_episodes):
                result = await self.run_episode()
                results.append(result)
                if result.get("status") == "complete":
                    break
        finally:
            self.gpu_monitor.stop()
            console.print()  # newline after monitor
            self.gpu_monitor.print_summary()
        return results

    async def generate_for_task(
        self,
        description: str,
        task_type: str = "code_generation",
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """One-shot generation for a user-provided task. Supports images."""
        from .core.types import TaskType, TaskDifficulty
        task = Task(
            id="user_task",
            description=description,
            task_type=TaskType(task_type),
            difficulty=TaskDifficulty.INTERMEDIATE,
            source="user",
        )
        # If images provided, inject them into the generator context
        if images and self.llm.mode == "scratch":
            # For now, images are passed via the LLM engine directly in generate()
            sol = await self.generator.generate(task, attempt=1)
            # Re-generate with image context
            raw = self.llm.generate(
                description,
                images=images,
                system_prompt="You are an expert software engineer. Write clean, correct code.",
            )
            from .core.types import SolutionStatus
            sol.code = raw
            sol.language = "python" if "import" in raw else "typescript"
        else:
            sol = await self.generator.generate(task, attempt=1)
        result = await self.evaluator.evaluate(sol, task)
        return {
            "code": sol.code,
            "language": sol.language,
            "success": result.success,
            "reward": self.feedback.compute_reward(result),
            "feedback": result.feedback,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "episodes": self._episodes_completed,
            "model": self.llm.get_stats(),
            "memory": self.memory.get_stats(),
            "curriculum": self.curriculum.get_progress(),
        }

    def print_status(self) -> None:
        stats = self.get_status()
        table = Table(title="ASSBRAIN Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("State", stats["state"])
        table.add_row("Episodes", str(stats["episodes"]))
        model_stats = stats["model"]
        if "params_B" in model_stats:
            table.add_row("Model", f"Custom {model_stats['params_B']}B multimodal")
            table.add_row("Vision", f"{model_stats.get('img_size', 336)}px ViT")
            table.add_row("Tools", str(model_stats.get("use_tools", False)))
        else:
            table.add_row("Model", model_stats["model_id"])
        table.add_row("Device", model_stats["device"])
        table.add_row("LoRA Active", str(model_stats["lora_active"]))
        table.add_row("Experiences", str(stats["memory"]["total_experiences"]))
        table.add_row("Avg Reward", str(stats["memory"]["avg_reward"]))
        table.add_row("Curriculum", f"{stats['curriculum']['completed']}/{stats['curriculum']['total_steps']}")
        console.print(table)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

_orchestrator: Optional[AssBrainOrchestrator] = None


def _get_orch() -> AssBrainOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AssBrainOrchestrator()
    return _orchestrator


@app.command()
def status():
    """Show current agent status."""
    orch = _get_orch()
    orch.print_status()


@app.command()
def learn(
    episodes: int = typer.Option(5, "--episodes", "-n", help="Number of episodes to run"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Run self-learning episodes."""
    global _orchestrator
    if config:
        _orchestrator = AssBrainOrchestrator(config)
    orch = _get_orch()
    asyncio.run(orch.run_curriculum(max_episodes=episodes))
    orch.print_status()


@app.command()
def generate(
    description: str = typer.Argument(..., help="Task description"),
    task_type: str = typer.Option("code_generation", "--type", "-t", help="Task type"),
):
    """Generate a solution for a one-off task."""
    orch = _get_orch()
    result = asyncio.run(orch.generate_for_task(description, task_type))
    console.print(Panel(result["code"], title=f"Generated ({result['language']}) — Success: {result['success']}"))


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send"),
):
    """Chat with the local model."""
    orch = _get_orch()
    response = orch.llm.generate(message)
    console.print(Panel(response, title="ASSBRAIN"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the FastAPI server."""
    from .api.server import create_app
    orch = _get_orch()
    app = create_app(orch)
    import uvicorn
    uvicorn.run(app, host=host, port=port)


@app.command()
def load_datasets(
    name: Optional[str] = typer.Option(None, "--name", help="Specific dataset to load"),
):
    """Load HuggingFace datasets for initial training."""
    orch = _get_orch()
    if name:
        ds = orch.hf_loader.load_dataset(name)
        console.print(f"Loaded [green]{name}[/green]: {len(ds)} samples")
    else:
        datasets = orch.hf_loader.load_code_datasets()
        for k, v in datasets.items():
            sz = len(v) if hasattr(v, "__len__") else "?"
            console.print(f"  {k}: {sz} samples")


@app.command()
def train(
    adapter_name: str = typer.Option("latest", "--adapter", help="Adapter save name"),
):
    """Trigger fine-tuning on accumulated high-reward experiences."""
    orch = _get_orch()
    batch = orch.memory.get_training_batch(batch_size=32)
    if len(batch) < 5:
        console.print("[yellow]Not enough high-reward experiences to train.[/yellow]")
        return
    console.print(f"Training on {len(batch)} experiences...")
    adapter_path = orch.llm.train_on_experiences(batch, output_dir=f"./models/{adapter_name}")
    console.print(f"[green]Adapter saved: {adapter_path}[/green]")


if __name__ == "__main__":
    app()
