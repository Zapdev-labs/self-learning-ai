#!/usr/bin/env python3
"""
ASSBRAIN — One-shot self-learning launcher.

Just run:
    python run.py

This script bootstraps itself: creates a venv, installs CUDA packages,
detects your A40 GPU, writes an optimized config, and starts the
self-learning loop with all features enabled.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
VENV_DIR = REPO_ROOT / ".venv"
CONFIG_DIR = REPO_ROOT / "config"
LOCAL_CONFIG = CONFIG_DIR / "local.yaml"


def log(msg: str):
    print(f"[ASSBRAIN] {msg}")


def run(cmd: list[str], **kwargs):
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        log(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or bool(os.getenv("VIRTUAL_ENV"))
    )


def ensure_venv():
    if in_venv():
        return
    if not VENV_DIR.exists():
        log("Creating virtualenv at .venv ...")
        venv.create(VENV_DIR, with_pip=True)
    python = VENV_DIR / "bin" / "python"
    log(f"Re-launching inside venv: {python}")
    os.execv(str(python), [str(python), __file__] + sys.argv[1:])


def get_gpu_info() -> dict | None:
    """Use nvidia-smi to get GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=json,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)[0]
    except Exception:
        pass
    return None


def install_deps():
    """Install all Python dependencies with CUDA support."""
    pip = [sys.executable, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "wheel", "setuptools"])

    gpu = get_gpu_info()
    has_gpu = gpu is not None
    gpu_name = gpu.get("name", "Unknown") if gpu else "None"
    log(f"Detected GPU: {gpu_name}")

    # PyTorch with CUDA
    if has_gpu:
        log("Installing PyTorch + CUDA 12.4 ...")
        run(pip + [
            "install", "torch>=2.3.0", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])
    else:
        log("Installing CPU-only PyTorch ...")
        run(pip + [
            "install", "torch>=2.3.0", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])

    # FAISS
    if has_gpu:
        log("Installing FAISS GPU ...")
        run(pip + ["install", "faiss-gpu-cu12>=1.14.0"])
    else:
        log("Installing FAISS CPU ...")
        run(pip + ["install", "faiss-cpu>=1.7.4"])

    # Everything else
    req_file = REPO_ROOT / "requirements.txt"
    lines = req_file.read_text().splitlines()
    filtered = []
    for line in lines:
        s = line.strip().lower()
        if any(pkg in s for pkg in ["torch", "torchvision", "torchaudio", "faiss-cpu", "faiss-gpu"]):
            if not s.startswith("#"):
                continue
        filtered.append(line)
    tmp = REPO_ROOT / ".__req_tmp.txt"
    tmp.write_text("\n".join(filtered))
    try:
        run(pip + ["install", "-r", str(tmp)])
    finally:
        tmp.unlink()

    # Playwright browsers
    log("Installing Playwright Chromium ...")
    run([sys.executable, "-m", "playwright", "install", "chromium"])

    return gpu


def validate_gpu():
    log("Validating GPU setup ...")
    script = """
import torch, sys
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem:.1f} GB")
    x = torch.rand(2000, 2000, device="cuda")
    y = torch.mm(x, x)
    print(f"GPU matmul test: OK")
    sys.exit(0)
else:
    print("WARNING: No CUDA available — falling back to CPU")
    sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        log(line)
    if result.returncode != 0:
        log(f"Validation stderr: {result.stderr}")


def write_a40_config(gpu_info: dict | None):
    """Write a local.yaml optimized for NVIDIA A40 (48GB VRAM) scratch model."""
    if not gpu_info:
        return
    name = gpu_info.get("name", "").upper()
    mem = gpu_info.get("memory.total", "0 MiB")
    try:
        vram_gb = int(mem.split()[0]) / 1024
    except Exception:
        vram_gb = 0

    is_a40 = "A40" in name
    if not is_a40 and vram_gb < 40:
        return

    log(f"Writing A40 scratch-model config (VRAM: {vram_gb:.0f} GB) ...")

    # A40: ~1.7B parameter custom transformer, fits in 48GB with fp16 training
    config_text = f"""# Auto-generated by run.py for {gpu_info.get('name', 'GPU')}
# SCRATCH MODEL — custom transformer trained from scratch

llm:
  mode: "scratch"
  device: "auto"
  compile: true

  # ~1.7B parameter architecture
  vocab_size: 32000
  block_size: 4096
  n_layer: 32
  n_head: 32
  n_embd: 2048
  dropout: 0.0
  ffn_mult: 4.0
  multiple_of: 256
  tie_weights: true

  # Generation
  max_new_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  context_window: 4096
  batch_size: 4

  # Training
  training:
    learning_rate: 1.0e-4
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    num_train_epochs: 1
    max_steps: 500
    warmup_steps: 50
    logging_steps: 5
    save_steps: 100
    output_dir: "./models/checkpoints"

learning:
  max_attempts_per_task: 5
  reward_threshold: 0.8
  curriculum_enabled: true
  self_critique_enabled: true
  imitation_learning_enabled: true
  error_analysis_depth: "deep"
  experience_buffer_size: 100000
  training_frequency: 5
  explore_vs_exploit: 0.3

evaluation:
  sandbox_type: "subprocess"
  timeout_seconds: 120
  max_memory_mb: 4096
  allow_network: false
"""
    LOCAL_CONFIG.write_text(config_text)
    log(f"Config written to {LOCAL_CONFIG}")


def main():
    ensure_venv()

    # Check if we need to install
    try:
        import torch  # noqa: F401
        deps_ready = True
    except ImportError:
        deps_ready = False

    gpu_info = None
    if not deps_ready:
        log("First run detected — installing dependencies ...")
        gpu_info = install_deps()
        validate_gpu()
        write_a40_config(gpu_info)
    else:
        gpu_info = get_gpu_info()
        write_a40_config(gpu_info)

    # Now import and run the actual app
    log("Starting ASSBRAIN self-learning loop ...")
    sys.path.insert(0, str(REPO_ROOT))
    from src.main import AssBrainOrchestrator  # type: ignore

    import asyncio
    orch = AssBrainOrchestrator(config_path=str(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else None)
    orch.print_status()

    # Optional: pre-train on synthetic code if model is fresh (no checkpoint)
    _maybe_warmup_pretrain(orch)

    # Run the self-learning curriculum
    try:
        asyncio.run(orch.run_curriculum(max_episodes=10))
    except KeyboardInterrupt:
        log("Interrupted by user.")
    finally:
        orch.print_status()


def _maybe_warmup_pretrain(orch):
    """If the scratch model has no checkpoints, do a quick warm-up pre-train."""
    from pathlib import Path
    ckpt_dir = Path("./models/checkpoints")
    has_ckpt = any(ckpt_dir.glob("checkpoint-*.pt"))
    if has_ckpt:
        log("Checkpoint found — skipping warm-up pre-training.")
        return

    log("No checkpoint found — running warm-up pre-training ...")
    from src.core.custom_trainer import CustomTrainer, load_hf_code_datasets, seed_corpus_for_tokenizer

    engine = orch.llm
    if engine.mode != "scratch" or engine.model is None:
        log("Not in scratch mode — skipping warm-up.")
        return

    texts = []

    # 1. Try loading real code datasets from HuggingFace
    try:
        hf_texts = load_hf_code_datasets(engine.custom_tokenizer, max_samples=3000)
        texts.extend(hf_texts)
        log(f"Loaded {len(hf_texts)} samples from HuggingFace datasets")
    except Exception as e:
        log(f"HF dataset loading failed: {e}")

    # 2. Fallback to synthetic corpus
    if len(texts) < 100:
        log("Using synthetic code corpus as fallback...")
        corpus_files = seed_corpus_for_tokenizer("./models/corpus")
        for fpath in corpus_files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except Exception:
                pass

    total_chars = sum(len(t) for t in texts)
    if total_chars < 10000:
        log("Not enough training data — skipping warm-up.")
        return

    log(f"Pre-training on {len(texts)} texts ({total_chars:,} chars) ...")
    trainer = CustomTrainer(
        model=engine.model,
        tokenizer=engine.custom_tokenizer,
        device=engine.device,
        learning_rate=5e-4,
        batch_size=4,
        grad_accum_steps=4,
        max_steps=1000,
        warmup_steps=100,
        compile_model=engine.cfg.get("compile", True),
    )
    trainer.pretrain_on_texts(texts, str(ckpt_dir))
    log("Warm-up pre-training complete.")


if __name__ == "__main__":
    main()
