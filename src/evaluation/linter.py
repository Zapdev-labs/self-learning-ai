"""Static analysis — pylint, mypy, ruff for Python; eslint for JS/TS."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.types import Solution

logger = logging.getLogger(__name__)


class Linter:
    """Runs static analysis tools on generated code."""

    def __init__(self, config: Config):
        self.cfg = config.evaluation

    def lint(self, solution: Solution, project_dir: Optional[Path] = None) -> Tuple[float, bool, str]:
        """
        Run linters and type checkers.
        Returns (lint_score, type_check_passed, combined_output).
        """
        if solution.language == "python":
            return self._lint_python(solution, project_dir)
        elif solution.language in ("typescript", "javascript"):
            return self._lint_ts(solution, project_dir)
        return 0.0, False, "No linter available for language"

    def _lint_python(
        self, solution: Solution, project_dir: Optional[Path]
    ) -> Tuple[float, bool, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(solution.code)
            temp_path = f.name

        outputs: List[str] = []
        score = 10.0
        type_ok = False

        # Ruff (fast, modern)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                outputs.append("Ruff:\n" + result.stdout)
                error_count = result.stdout.count("\n")
                score -= min(5, error_count * 0.5)
        except FileNotFoundError:
            pass

        # Pylint
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pylint", "--score=n", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                outputs.append("Pylint:\n" + result.stdout)
                error_count = result.stdout.count(":")
                score -= min(3, error_count * 0.3)
        except FileNotFoundError:
            pass

        # MyPy
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--ignore-missing-imports", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            type_ok = result.returncode == 0
            if not type_ok:
                outputs.append("MyPy:\n" + result.stdout + result.stderr)
        except FileNotFoundError:
            type_ok = True  # mypy not installed

        # Cleanup
        try:
            Path(temp_path).unlink()
        except OSError:
            pass

        score = max(0.0, min(10.0, score))
        return score, type_ok, "\n\n".join(outputs)

    def _lint_ts(
        self, solution: Solution, project_dir: Optional[Path]
    ) -> Tuple[float, bool, str]:
        if project_dir is None:
            return 0.0, False, "TypeScript linting requires project context"

        outputs: List[str] = []
        score = 10.0
        type_ok = False

        # ESLint via npm
        ok, out = self._run_npm_cmd(project_dir, ["run", "lint"])
        if not ok and out.strip():
            outputs.append("ESLint:\n" + out)
            score -= 3

        # TypeScript type check
        ok2, out2 = self._run_npm_cmd(project_dir, ["npx", "tsc", "--noEmit"])
        type_ok = ok2
        if not ok2:
            outputs.append("TypeScript:\n" + out2)
            score -= 2

        score = max(0.0, min(10.0, score))
        return score, type_ok, "\n\n".join(outputs)

    def _run_npm_cmd(self, cwd: Path, args: List[str]) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                ["npm"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0, result.stdout + "\n" + result.stderr
        except Exception as e:
            return False, str(e)
