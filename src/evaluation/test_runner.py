"""Automated test discovery and execution."""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.types import Solution

logger = logging.getLogger(__name__)


class TestRunner:
    """Runs pytest on generated code and collects results."""

    def __init__(self, config: Config):
        self.cfg = config.evaluation
        self.timeout = self.cfg.get("timeout_seconds", 60)

    def run_tests(
        self,
        solution: Solution,
        project_dir: Optional[Path] = None,
        test_code: Optional[str] = None,
    ) -> Tuple[bool, int, int, str]:
        """
        Run tests on a solution.
        Returns (all_passed, passed_count, failed_count, output).
        """
        if solution.language != "python":
            return False, 0, 0, "Test runner only supports Python"

        # Create temp project if needed
        if project_dir is None:
            project_dir = Path(tempfile.mkdtemp(prefix="assbrain_test_"))
            (project_dir / "solution.py").write_text(solution.code)
        else:
            # Write solution code into project
            sol_path = project_dir / "solution.py"
            sol_path.write_text(solution.code)

        # Write or discover tests
        test_dir = project_dir / "tests"
        test_dir.mkdir(exist_ok=True)

        if test_code:
            (test_dir / "test_generated.py").write_text(test_code)
        else:
            # Generate a basic import test
            basic_test = f"""
import sys
sys.path.insert(0, str({repr(str(project_dir))}))
import solution

def test_imports():
    assert solution is not None
"""
            (test_dir / "test_basic.py").write_text(basic_test)

        # Run pytest
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
            "--no-header",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + "\n" + result.stderr
            passed, failed = self._parse_pytest_output(output)
            all_passed = result.returncode == 0 and failed == 0
            return all_passed, passed, failed, output
        except subprocess.TimeoutExpired:
            return False, 0, 0, f"Tests timed out after {self.timeout}s"
        except Exception as e:
            return False, 0, 0, str(e)

    def _parse_pytest_output(self, output: str) -> Tuple[int, int]:
        """Parse pytest output to count passes and failures."""
        passed = 0
        failed = 0
        for line in output.split("\n"):
            if line.startswith("passed") or " passed" in line.lower():
                parts = line.split(",")
                for p in parts:
                    p = p.strip()
                    if "passed" in p:
                        try:
                            passed = int(p.split()[0])
                        except (ValueError, IndexError):
                            pass
                    if "failed" in p:
                        try:
                            failed = int(p.split()[0])
                        except (ValueError, IndexError):
                            pass
        return passed, failed
