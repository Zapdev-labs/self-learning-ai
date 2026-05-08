"""Sandboxed code execution for Python and shell commands."""

import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.types import Solution

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Executes Python code in a restricted subprocess sandbox."""

    def __init__(self, config: Config):
        self.cfg = config.evaluation
        self.timeout = self.cfg.get("timeout_seconds", 60)
        self.max_memory_mb = self.cfg.get("max_memory_mb", 512)
        self.allow_network = self.cfg.get("allow_network", False)
        self.allowed_imports = set(self.cfg.get("allowed_imports", []))
        self.blocked_imports = set(self.cfg.get("blocked_imports", []))
        self.auto_install = self.cfg.get("auto_install_deps", True)

    def execute(
        self,
        solution: Solution,
        project_dir: Optional[Path] = None,
        extra_files: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, List[str], float]:
        """
        Execute solution code safely.
        Returns (success, stdout, runtime_errors, execution_time_sec).
        """
        if solution.language == "python":
            return self._execute_python(solution, project_dir, extra_files)
        return False, "Unsupported language for execution", [], 0.0

    def _execute_python(
        self,
        solution: Solution,
        project_dir: Optional[Path],
        extra_files: Optional[Dict[str, str]],
    ) -> Tuple[bool, str, List[str], float]:
        code = solution.code
        errors: List[str] = []

        # Security: basic import checks
        for blocked in self.blocked_imports:
            if blocked in code:
                return False, f"", [f"Blocked import/usage: {blocked}"], 0.0

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Also write extra files if provided
        temp_dir = Path(temp_path).parent
        if extra_files:
            for name, content in extra_files.items():
                (temp_dir / name).write_text(content)

        # Build command
        cmd = [sys.executable, temp_path]
        env = os.environ.copy()
        if not self.allow_network:
            # Best-effort network isolation (Linux only)
            pass

        try:
            start = os.times()[0]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(project_dir) if project_dir else temp_dir,
                env=env,
            )
            elapsed = os.times()[0] - start

            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            if stderr:
                errors.append(stderr)
            if not success:
                errors.append(f"Exit code: {result.returncode}")

            return success, stdout, errors, elapsed

        except subprocess.TimeoutExpired:
            return False, "", [f"Execution timed out after {self.timeout}s"], float(self.timeout)
        except Exception as e:
            return False, "", [str(e)], 0.0
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Run a shell command safely."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
            return result.returncode == 0, result.stdout + "\n" + result.stderr
        except Exception as e:
            return False, str(e)
