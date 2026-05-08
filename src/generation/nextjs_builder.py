"""Next.js specific build, dev server, and validation logic."""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.types import EvaluationResult, Solution, Task
from .project_scaffold import ProjectScaffold

logger = logging.getLogger(__name__)


class NextJSBuilder:
    """Builds and validates Next.js projects."""

    def __init__(self, config: Config, scaffold: ProjectScaffold):
        self.cfg = config.generation.get("nextjs", {})
        self.scaffold = scaffold
        self.node_version = self.cfg.get("version", "14")

    async def build_project(self, project_dir: Path) -> Tuple[bool, str]:
        """Run `npm install` and `npm run build` in the project."""
        # Ensure node_modules exists
        if not (project_dir / "node_modules").exists():
            install_ok, install_out = await self._run_npm(project_dir, ["install"])
            if not install_ok:
                return False, f"npm install failed:\n{install_out}"

        build_ok, build_out = await self._run_npm(project_dir, ["run", "build"])
        return build_ok, build_out

    async def lint_project(self, project_dir: Path) -> Tuple[bool, str]:
        """Run `npm run lint` if available."""
        return await self._run_npm(project_dir, ["run", "lint"])

    async def typecheck_project(self, project_dir: Path) -> Tuple[bool, str]:
        """Run `npx tsc --noEmit` for type checking."""
        return await self._run_npm(project_dir, ["npx", "tsc", "--noEmit"])

    async def start_dev_server(self, project_dir: Path, port: int = 3000) -> subprocess.Popen:
        """Start the Next.js dev server for browser testing."""
        env = os.environ.copy()
        env["PORT"] = str(port)
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait a bit for server to start
        time.sleep(5)
        return proc

    async def _run_npm(
        self, project_dir: Path, args: List[str], timeout: int = 120
    ) -> Tuple[bool, str]:
        cmd = ["npm"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def prepare_solution(self, task: Task, solution: Solution) -> Path:
        """Scaffold a Next.js project and write solution files into it."""
        project_name = f"nextjs_{task.id[:8]}"
        project_dir = self.scaffold.scaffold(project_name, template="nextjs")
        self.scaffold.write_solution(project_dir, solution)
        return project_dir
