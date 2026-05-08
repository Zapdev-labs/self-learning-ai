# 🧠 ASSBRAIN — Self-Learning AI Agent

ASSBRAIN is a **locally-run, self-improving AI coding agent** built in Python. It generates ML/AI code and Next.js applications, tests them in sandboxes, learns from failures via self-critique, and fine-tunes itself using LoRA — all without human feedback.

## Architecture Overview

```
ASSBRAIN
├── Core Engine          (Local LLM via HuggingFace Transformers)
├── Learning Loop        (Self-RL, Memory, Curriculum, Feedback)
├── Generation           (Code Generator, Project Scaffold, Next.js Builder)
├── Evaluation           (Sandbox Exec, Pytest, Lint, Browser Validation)
├── Tools                (Browser, MCP, LSP, HuggingFace Datasets)
└── Pi Integration       (Custom model provider for the pi harness)
```

## Key Features

- **🤖 Local LLM Inference** — Loads any HuggingFace causal LM (DialoGPT, Mistral, CodeLlama, etc.)
- **🔄 Self-Learning Loop** — Generates code → Tests → Self-critiques → Retries → Stores experience → Fine-tunes LoRA
- **🧠 Vector Memory** — Semantic search over past experiences using Chroma + sentence-transformers
- **📚 Progressive Curriculum** — Built-in ML/AI + Next.js tasks from beginner to expert
- **🌐 Browser Automation** — Playwright-based validation for Next.js apps (screenshots, console errors, DOM checks)
- **🔧 Tool Use** — MCP client, LSP integration, HuggingFace dataset loader
- **⚡ Pi Integration** — Exposes ASSBRAIN as a custom model provider inside the pi agent harness
- **🚀 FastAPI Server** — REST API for chat, generation, learning episodes, and memory search

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install
```

### 2. Set Your HuggingFace Token

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 3. Run the CLI

```bash
# Check status
python run.py status

# Chat with the local model
python run.py chat "Write a Python function for quicksort"

# Run self-learning episodes
python run.py learn --episodes 5

# Generate a one-off solution
python run.py generate "Build a Next.js dashboard with Tailwind" --type nextjs_app

# Load initial datasets from HuggingFace
python run.py load-datasets

# Trigger fine-tuning on accumulated experiences
python run.py train

# Start the API server
python run.py serve --port 8000
```

### 4. Use with Pi

Configure pi to use ASSBRAIN as a custom model:

```yaml
# In your pi config
models:
  assbrain-local:
    provider: custom
    module: src.pi_integration.custom_model
    class: CustomModelProvider
    config:
      model_id: microsoft/DialoGPT-medium
      device: auto
```

## How Self-Learning Works

1. **Curriculum** provides progressively harder tasks (Python basics → ML models → Next.js apps)
2. **Code Generator** creates a solution using the local LLM + relevant memories
3. **Evaluator** runs the code in a sandbox:
   - Python: executes, runs pytest, lints with ruff/pylint/mypy
   - Next.js: `npm install` → `npm run build` → browser screenshot validation
4. **Feedback Loop** computes a scalar reward from all metrics
5. **Self-Critique** asks the model to analyze its own failure and suggest fixes
6. **Retry** with error context + similar successful past solutions
7. **Store** the best experience in vector memory (Chroma)
8. **Train** LoRA adapters periodically on high-reward experiences

## Configuration

Edit `config/default.yaml` or create `config/local.yaml`:

```yaml
llm:
  model_id: "microsoft/DialoGPT-medium"  # or "mistralai/Mistral-7B-Instruct-v0.2"
  device: "auto"  # auto | cpu | cuda | mps
  load_in_4bit: false
  temperature: 0.7

learning:
  max_attempts_per_task: 5
  reward_threshold: 0.8
  training_frequency: 10  # train after N new experiences

memory:
  backend: "chroma"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

evaluation:
  sandbox_type: "subprocess"
  timeout_seconds: 60

browser:
  headless: true
  screenshot_on_fail: true
```

## Project Structure

```
├── config/
│   └── default.yaml
├── src/
│   ├── core/           # Config, LLMEngine, Types
│   ├── learning/       # Memory, Self-RL, Feedback, Curriculum
│   ├── generation/     # CodeGenerator, ProjectScaffold, NextJSBuilder
│   ├── evaluation/     # CodeExecutor, TestRunner, Linter, BrowserTester, Evaluator
│   ├── tools/          # BrowserTool, MCPClient, LSPClient, HuggingFaceLoader
│   ├── pi_integration/ # PiAdapter, CustomModelProvider
│   ├── api/            # FastAPI server
│   └── main.py         # CLI orchestrator
├── data/               # Datasets, cache
├── memory/             # Vector DB + SQLite
├── models/             # Downloaded models + LoRA checkpoints
├── projects/           # Generated Next.js/Python projects
├── logs/               # Runtime logs
├── requirements.txt
└── run.py
```

## API Endpoints

When running `python run.py serve`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Agent status |
| POST | `/chat` | Chat completion |
| POST | `/generate` | One-shot code generation |
| POST | `/learn?episodes=5` | Run learning episodes |
| POST | `/episode` | Single episode |
| GET | `/memory/search?query=...` | Semantic memory search |
| POST | `/train` | Trigger LoRA fine-tuning |

## Extending ASSBRAIN

### Add a New Curriculum Task

```python
from src.learning.curriculum import Curriculum
from src.core.types import TaskType, TaskDifficulty

curriculum = Curriculum()
curriculum.add_custom_step(
    description="Implement a GAN training loop in PyTorch",
    task_type=TaskType.ML_MODEL,
    difficulty=TaskDifficulty.EXPERT,
    prerequisites=[13, 14],
)
```

### Add a Custom Tool (MCP)

Edit `config/default.yaml`:

```yaml
mcp:
  servers:
    - name: "filesystem"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "./projects"]
```

## License

MIT — Built for autonomous AI research.
