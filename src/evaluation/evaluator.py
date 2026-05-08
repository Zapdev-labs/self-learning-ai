"""Unified evaluator that orchestrates execution, testing, linting, and browser validation."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import Config
from ..core.types import EvaluationResult, Solution, Task, TaskType
from ..generation.nextjs_builder import NextJSBuilder
from ..generation.project_scaffold import ProjectScaffold
from .browser_tester import BrowserTester
from .code_executor import CodeExecutor
from .linter import Linter
from .test_runner import TestRunner

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a solution across all relevant dimensions."""

    def __init__(
        self,
        config: Config,
        executor: CodeExecutor,
        test_runner: TestRunner,
        linter: Linter,
        browser: BrowserTester,
        nextjs_builder: NextJSBuilder,
        scaffold: ProjectScaffold,
    ):
        self.cfg = config.evaluation
        self.executor = executor
        self.test_runner = test_runner
        self.linter = linter
        self.browser = browser
        self.nextjs = nextjs_builder
        self.scaffold = scaffold

    async def evaluate(self, solution: Solution, task: Task) -> EvaluationResult:
        """Full evaluation pipeline for a solution."""
        result = EvaluationResult(
            solution_id=solution.id,
            task_id=task.id,
            success=False,
            score=0.0,
        )

        project_dir: Optional[Path] = None

        try:
            if task.task_type == TaskType.NEXTJS_APP:
                result = await self._evaluate_nextjs(solution, task, result)
            else:
                result = await self._evaluate_python(solution, task, result)
        except Exception as e:
            logger.exception("Evaluation failed")
            result.runtime_errors.append(str(e))

        return result

    async def _evaluate_python(
        self, solution: Solution, task: Task, result: EvaluationResult
    ) -> EvaluationResult:
        # 1. Lint
        lint_score, type_ok, lint_out = self.linter.lint(solution)
        result.lint_score = lint_score
        result.type_check_passed = type_ok
        result.compile_errors.append(lint_out) if lint_out else None

        # 2. Execute
        exec_ok, exec_out, exec_err, exec_time = self.executor.execute(solution)
        result.runtime_errors.extend(exec_err)
        result.performance_metrics["execution_time"] = exec_time

        # 3. Tests
        test_ok, passed, failed, test_out = self.test_runner.run_tests(solution)
        result.test_passed = passed
        result.test_failed = failed
        result.test_output = test_out

        # 4. Determine success
        result.build_success = exec_ok
        result.success = exec_ok and test_ok and failed == 0 and not exec_err
        result.score = result.success
        result.feedback = self._build_feedback(result)
        return result

    async def _evaluate_nextjs(
        self, solution: Solution, task: Task, result: EvaluationResult
    ) -> EvaluationResult:
        # 1. Prepare project
        project_dir = self.nextjs.prepare_solution(task, solution)

        # 2. Build
        build_ok, build_out = await self.nextjs.build_project(project_dir)
        result.build_success = build_ok
        if not build_ok:
            result.compile_errors.append(build_out[:1000])

        # 3. Lint / Type check
        lint_score, type_ok, lint_out = self.linter.lint(solution, project_dir)
        result.lint_score = lint_score
        result.type_check_passed = type_ok

        # 4. Browser validation (if build succeeded)
        if build_ok:
            # Start dev server
            proc = await self.nextjs.start_dev_server(project_dir, port=3456)
            try:
                ss_dir = project_dir / "screenshots"
                ss_dir.mkdir(exist_ok=True)
                browser_ok, browser_details, ss_path = await self.browser.validate_nextjs_dev(
                    port=3456, screenshot_dir=ss_dir
                )
                result.success = browser_ok
                result.feedback = browser_details
                if ss_path:
                    result.browser_screenshot_path = str(ss_path)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        else:
            result.success = False
            result.feedback = f"Build failed:\n{build_out[:500]}"

        return result

    def _build_feedback(self, result: EvaluationResult) -> str:
        parts = []
        if result.test_passed > 0:
            parts.append(f"Tests: {result.test_passed} passed")
        if result.test_failed > 0:
            parts.append(f"Tests: {result.test_failed} failed")
        if result.runtime_errors:
            parts.append(f"Runtime errors: {len(result.runtime_errors)}")
        if result.compile_errors:
            parts.append("Compile/type errors present")
        return "; ".join(parts) if parts else "No issues detected"
