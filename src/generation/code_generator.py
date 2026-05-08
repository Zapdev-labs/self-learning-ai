"""Code generation engine with prompt engineering for ML/AI and web tasks."""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.llm_engine import LLMEngine
from ..core.types import Experience, Solution, Task, TaskType
from ..learning.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates code solutions for tasks using the local LLM."""

    def __init__(self, config: Config, llm: LLMEngine, memory: MemoryStore):
        self.cfg = config.generation
        self.llm = llm
        self.memory = memory
        self.supported_langs = self.cfg.get("supported_languages", ["python", "typescript", "javascript"])

    async def generate(
        self,
        task: Task,
        attempt: int = 1,
        previous_exp: Optional[Experience] = None,
    ) -> Solution:
        """Generate a solution for the given task."""
        logger.info(f"📝 Generating solution for task {task.id} (attempt {attempt})")

        # Retrieve relevant past experiences for context
        similar = self.memory.get_similar_successes(task, limit=3)

        # Build the generation prompt
        prompt = self._build_generation_prompt(task, attempt, previous_exp, similar)

        # Generate with adjusted temperature for exploration
        temp = self._temperature_for_attempt(attempt)
        raw = self.llm.generate(prompt, max_new_tokens=2048, temperature=temp)

        # Parse code from response
        code, files = self._extract_code(raw, task)
        language = self._detect_language(code, task)

        sol = Solution(
            id=uuid.uuid4().hex,
            task_id=task.id,
            attempt_number=attempt,
            code=code,
            language=language,
            files=files,
            metadata={
                "prompt": prompt,
                "raw_response": raw,
                "temperature": temp,
                "similar_experiences": len(similar),
            },
        )
        return sol

    def _build_generation_prompt(
        self,
        task: Task,
        attempt: int,
        previous_exp: Optional[Experience],
        similar: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []

        # System context
        parts.append(
            "You are ASSBRAIN, an expert software engineer and ML researcher. "
            "You write clean, correct, well-tested code. You think step by step."
        )
        parts.append("")

        # Task description
        parts.append(f"Task: {task.description}")
        if task.requirements:
            parts.append("Requirements:")
            for req in task.requirements:
                parts.append(f"  - {req}")
        parts.append("")

        # Previous attempt feedback
        if previous_exp and previous_exp.solution.critique:
            parts.append("Previous attempt failed. Feedback:")
            parts.append(previous_exp.solution.critique)
            parts.append("")

        # Similar successful patterns
        if similar:
            parts.append("Successful patterns from memory:")
            for s in similar[:2]:
                doc = s.get("document", "")
                parts.append(f"  - {doc[:250]}...")
            parts.append("")

        # Language hint based on task type
        lang_hint = self._lang_hint_for_task(task)
        parts.append(f"Write your solution in {lang_hint}.")

        # Output format instructions
        parts.append(
            "Respond ONLY with code. Wrap your solution in markdown code blocks. "
            "If multiple files are needed, use this format:\n"
            "```filename: path/to/file.ext\n"
            "...code...\n"
            "```"
        )

        return "\n".join(parts)

    def _extract_code(self, raw: str, task: Task) -> tuple:
        """Extract code blocks from LLM response."""
        code_pattern = r"```(?:\w+)?\n(.*?)```"
        file_pattern = r"```(?:\w+)?\s*filename:\s*(.+?)\n(.*?)```"

        files: Dict[str, str] = {}
        main_code = ""

        # Look for file-tagged blocks first
        file_matches = list(re.finditer(file_pattern, raw, re.DOTALL))
        if file_matches:
            for m in file_matches:
                fname = m.group(1).strip()
                fcode = m.group(2).strip()
                files[fname] = fcode
            # Use first file as main code
            main_code = next(iter(files.values()))
        else:
            # Fall back to plain code blocks
            matches = re.findall(code_pattern, raw, re.DOTALL)
            if matches:
                main_code = matches[0].strip()
                # If multiple blocks, store extras
                for i, m in enumerate(matches[1:], 1):
                    files[f"extra_{i}.py"] = m.strip()
            else:
                # No code blocks found — treat entire response as code
                main_code = raw.strip()

        return main_code, files

    def _detect_language(self, code: str, task: Task) -> str:
        if task.task_type == TaskType.NEXTJS_APP:
            return "typescript"
        if "import torch" in code or "import numpy" in code:
            return "python"
        if "import React" in code or "export default" in code:
            return "typescript"
        if task.task_type in (TaskType.CODE_GENERATION, TaskType.BUG_FIX, TaskType.REFACTOR, TaskType.TEST_WRITING, TaskType.ML_MODEL, TaskType.DATA_PIPELINE):
            return "python"
        return "python"

    def _lang_hint_for_task(self, task: Task) -> str:
        mapping = {
            TaskType.NEXTJS_APP: "TypeScript / React for Next.js 14",
            TaskType.CODE_GENERATION: "Python",
            TaskType.BUG_FIX: "Python",
            TaskType.REFACTOR: "Python",
            TaskType.TEST_WRITING: "Python (pytest)",
            TaskType.ML_MODEL: "Python (PyTorch / NumPy)",
            TaskType.DATA_PIPELINE: "Python",
            TaskType.EXPLORATION: "Python",
        }
        return mapping.get(task.task_type, "Python")

    def _temperature_for_attempt(self, attempt: int) -> float:
        """Increase temperature on later attempts for more creative fixes."""
        base = 0.6
        return min(0.95, base + (attempt - 1) * 0.1)
