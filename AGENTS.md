# AGENTS.md — ASSBRAIN

## Project

ASSBRAIN is a self-learning AI agent built around local LLM inference (HuggingFace Transformers). It generates code, evaluates it in sandboxes, and fine-tunes itself via LoRA on successful experiences.

## Repo Structure

```
config/default.yaml    # Primary configuration file
src/
  core/                # Config, LLMEngine, shared types
  learning/            # Memory store, curriculum, self-RL trainer, feedback loop
  generation/          # Code generator, project scaffold, Next.js builder
  evaluation/          # Code executor, test runner, linter, browser tester, Evaluator
tests/                 # Empty — no tests written yet
data/ memory/ models/ logs/ projects/   # Runtime directories (created on demand)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: set HuggingFace token for gated models
export HF_TOKEN=...
```

## Configuration

- Config loads from `config/default.yaml` by default.
- Override path with `ASSBRAIN_CONFIG` env var, or create `config/local.yaml` (checked second).
- Config supports dot-notation access: `config.get('llm.temperature')`.
- `HF_TOKEN` or `HUGGINGFACE_TOKEN` env vars are automatically injected into `llm.huggingface_token` at load time.
- All runtime directories (`data`, `memory`, `models`, `logs`, `projects`) are relative to the repo root by default.

## Key Architecture Notes

- **No setup.py / pyproject.toml** — this is a plain Python project; imports are relative within `src/`.
- **LLMEngine** (`src/core/llm_engine.py`) is the main entrypoint for inference and fine-tuning. It auto-loads the model on instantiation.
- **Evaluator** (`src/evaluation/evaluator.py`) orchestrates the full validation pipeline: lint → execute → test. For Next.js tasks it adds build → browser validation.
- **Code execution sandbox** is configurable: `subprocess` (default) or `docker`. See `evaluation.sandbox_type` in config.
- **LoRA fine-tuning** is applied on-demand via `LLMEngine.apply_lora()` and trained on successful experiences via `LLMEngine.train_on_experiences()`.

## Running the Project

There is no top-level CLI or main entrypoint yet. Instantiate and use classes directly:

```python
from src.core.config import load_config
from src.core.llm_engine import LLMEngine
from src.evaluation.evaluator import Evaluator

config = load_config()
engine = LLMEngine(config)
result = engine.generate("Write a Python function to...")
```

## Testing

- `tests/` is empty. No test framework is configured yet, though `pytest` and `pytest-asyncio` are in `requirements.txt`.
- When adding tests, run with: `pytest tests/`

## Code Quality

- Linting/formatting tools are in `requirements.txt` (`black`, `ruff`, `mypy`, `pylint`) but no config files or scripts exist yet.
- When adding config, prefer `pyproject.toml` for tool settings.

## Environment & Runtime Quirks

- **Model download**: First run will download the default model (`microsoft/DialoGPT-medium`) from HuggingFace Hub into the local cache. Ensure sufficient disk space and network access.
- **Device selection**: `device: auto` in config picks CUDA → MPS → CPU automatically.
- **Quantization**: 4-bit/8-bit via `bitsandbytes` is supported; enable in config (`load_in_4bit` / `load_in_8bit`).
- **Browser testing**: Requires Playwright browsers to be installed (`playwright install`) if using browser validation for Next.js apps.
- **Next.js builds**: The Next.js builder expects `node`/`npm`/`npx` to be available on PATH. It scaffolds projects under `./projects/`.
- **Sandbox security**: The default subprocess sandbox allows only whitelisted imports (see `evaluation.allowed_imports` in config). Blocked imports include `os.system`, `subprocess.call`, `eval`, `exec`, etc.

## Things That Are Missing / To Be Built

- No CLI entrypoint (`__main__.py` or script in repo root).
- No CI/CD workflows.
- No README.
- No actual tests.
- No lint/format config files (`.pre-commit-config.yaml`, `pyproject.toml`, etc.).
