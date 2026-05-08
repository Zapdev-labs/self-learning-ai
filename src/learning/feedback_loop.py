"""Error analysis, self-critique, and reward computation."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.llm_engine import LLMEngine
from ..core.types import EvaluationResult, Solution, Task

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Computes rewards from evaluation results, generates self-critique,
    and suggests fixes without any human in the loop.
    """

    def __init__(self, config: Config, llm: LLMEngine):
        self.cfg = config.learning
        self.llm = llm
        self.error_analysis_depth = self.cfg.get("error_analysis_depth", "deep")
        self.max_attempts = self.cfg.get("max_attempts_per_task", 5)

    def compute_reward(self, result: EvaluationResult) -> float:
        """
        Compute a scalar reward from evaluation metrics.
        Weighted combination of test success, build success, lint score,
        type check, and absence of runtime errors.
        """
        if result.success and result.test_passed > 0 and result.test_failed == 0:
            # Perfect pass with tests
            return 1.0

        weights = {
            "tests": 0.35,
            "build": 0.20,
            "lint": 0.15,
            "type_check": 0.10,
            "no_runtime_errors": 0.20,
        }

        # Test component
        total_tests = result.test_passed + result.test_failed
        test_score = result.test_passed / max(total_tests, 1)

        # Build component
        build_score = 1.0 if result.build_success else 0.0

        # Lint component (normalized to 0-1)
        lint_score = max(0.0, min(1.0, result.lint_score / 10.0))

        # Type check
        type_score = 1.0 if result.type_check_passed else 0.0

        # No runtime errors
        error_score = 1.0 if not result.runtime_errors else max(0.0, 1.0 - len(result.runtime_errors) * 0.25)

        reward = (
            weights["tests"] * test_score +
            weights["build"] * build_score +
            weights["lint"] * lint_score +
            weights["type_check"] * type_score +
            weights["no_runtime_errors"] * error_score
        )
        return round(min(1.0, max(0.0, reward)), 4)

    def generate_critique(
        self,
        task: Task,
        solution: Solution,
        result: EvaluationResult,
    ) -> Tuple[str, str]:
        """
        Ask the model to critique its own solution and suggest a fix.
        Returns (critique, suggested_fix_summary).
        """
        if result.success and result.reward >= 0.9:
            return "No issues — solution passed all checks.", ""

        # Build error context
        error_context = self._build_error_context(result)
        code_preview = solution.code[:2000] if len(solution.code) > 2000 else solution.code

        prompt = f"""You are an expert code reviewer. Analyze your own solution and identify why it failed.

Task: {task.description}
Language: {solution.language}
Attempt: {solution.attempt_number}

Your Code:
```{solution.language}
{code_preview}
```

Test Results: {result.test_passed} passed, {result.test_failed} failed.
Build Success: {result.build_success}
Lint Score: {result.lint_score}/10
Type Check: {result.type_check_passed}

Errors:
{error_context}

Provide:
1. CRITIQUE: What went wrong and why (be specific).
2. FIX: A concise description of how to fix it.
3. LESSON: A general lesson to remember for similar future tasks.
"""
        try:
            response = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.4)
        except Exception as e:
            logger.warning(f"Critique generation failed: {e}")
            return "Auto-critique unavailable.", "Retry with more careful attention to requirements."

        critique = self._extract_section(response, "CRITIQUE")
        fix = self._extract_section(response, "FIX")
        lesson = self._extract_section(response, "LESSON")

        full_critique = f"{critique}\n\nLesson: {lesson}".strip()
        return full_critique, fix

    def build_retry_prompt(
        self,
        task: Task,
        previous_solution: Solution,
        result: EvaluationResult,
        critique: str,
        similar_successes: List[Dict[str, Any]],
    ) -> str:
        """Build a rich prompt for the retry attempt using error feedback."""
        parts = [
            f"Task: {task.description}",
            f"Requirements: {', '.join(task.requirements)}",
            "",
            "Your previous attempt FAILED. Here is the feedback:",
            f"Critique: {critique}",
            "",
            "Errors encountered:",
            self._build_error_context(result),
            "",
        ]

        if similar_successes:
            parts.append("Hints from similar successful solutions:")
            for i, succ in enumerate(similar_successes[:2], 1):
                doc = succ.get("document", "")
                # Truncate to avoid overwhelming context
                parts.append(f"  {i}. {doc[:300]}...")
            parts.append("")

        parts.append(
            "Now write a corrected, complete solution. Do not include explanations outside the code block."
        )

        return "\n".join(parts)

    def _build_error_context(self, result: EvaluationResult) -> str:
        lines = []
        if result.compile_errors:
            lines.append("Compile Errors:")
            for e in result.compile_errors[:5]:
                lines.append(f"  - {e[:200]}")
        if result.runtime_errors:
            lines.append("Runtime Errors:")
            for e in result.runtime_errors[:5]:
                lines.append(f"  - {e[:200]}")
        if result.test_output and not result.success:
            lines.append("Test Output:")
            lines.append(result.test_output[-800:])  # last 800 chars
        if not lines:
            lines.append("No specific error details available.")
        return "\n".join(lines)

    def _extract_section(self, text: str, section: str) -> str:
        pattern = rf"{section}[:\s]*(.+?)(?=\n\d+\.\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: look for bold or header markers
        pattern2 = rf"\*\*{section}\*\*[:\s]*(.+?)(?=\n\*\*|\Z)"
        match2 = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return ""

    def should_continue(self, attempt: int, reward: float) -> bool:
        """Decide whether to attempt another fix."""
        if reward >= self.cfg.get("reward_threshold", 0.8):
            return False
        if attempt >= self.max_attempts:
            return False
        return True
