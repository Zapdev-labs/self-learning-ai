"""Training loop for the custom scratch transformer."""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .custom_transformer import AssBrainTransformer, ModelConfig
from .tokenizer_trainer import CodeTokenizer

logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Dataset that chunks token sequences into blocks."""

    def __init__(self, token_ids: List[int], block_size: int):
        self.block_size = block_size
        self.data = token_ids

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


class ExperienceDataset(Dataset):
    """Dataset built from self-learning experiences."""

    def __init__(self, experiences: List[Dict[str, str]], tokenizer: CodeTokenizer, block_size: int):
        self.block_size = block_size
        self.samples = []
        for exp in experiences:
            prompt = exp.get("prompt", "")
            chosen = exp.get("chosen", "")
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
            ids = tokenizer.encode(text)
            # Pad or chunk
            for i in range(0, max(1, len(ids) - block_size), block_size // 2):
                chunk = ids[i : i + block_size + 1]
                if len(chunk) < 2:
                    continue
                x = chunk[:-1] + [tokenizer.pad_token_id] * (block_size - len(chunk) + 1)
                y = chunk[1:] + [-100] * (block_size - len(chunk) + 1)
                self.samples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CustomTrainer:
    """Simplified trainer for the custom transformer."""

    def __init__(
        self,
        model: AssBrainTransformer,
        tokenizer: CodeTokenizer,
        device: str,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        compile_model: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Compile for speed (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")

        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        # AdamW with cosine decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

    def _get_lr(self, step: int, base_lr: float) -> float:
        if step < self.warmup_steps:
            return base_lr * step / self.warmup_steps
        decay_steps = self.max_steps - self.warmup_steps
        step_adjusted = step - self.warmup_steps
        coeff = 0.5 * (1 + math.cos(math.pi * step_adjusted / decay_steps))
        return base_lr * coeff

    def train_on_tokens(self, token_ids: List[int], save_dir: str):
        """Pre-train on raw token sequences."""
        dataset = TokenDataset(token_ids, self.model.config.block_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._train_loop(loader, save_dir, label="pretrain")

    def train_on_experiences(self, experiences: List[Dict[str, str]], save_dir: str):
        """Train on self-learning experiences."""
        dataset = ExperienceDataset(experiences, self.tokenizer, self.model.config.block_size)
        if len(dataset) == 0:
            logger.warning("No training samples from experiences")
            return
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._train_loop(loader, save_dir, label="experience")

    def pretrain_on_texts(self, texts: List[str], save_dir: str):
        """Pre-train on raw text strings (e.g., from HF datasets)."""
        all_ids = []
        for text in texts:
            all_ids.extend(self.tokenizer.encode(text))
            all_ids.append(self.tokenizer.eos_token_id)
        if len(all_ids) < self.model.config.block_size + 1:
            logger.warning("Not enough tokens for pre-training")
            return
        dataset = TokenDataset(all_ids, self.model.config.block_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._train_loop(loader, save_dir, label="pretrain")

    def _train_loop(self, loader: DataLoader, save_dir: str, label: str = "train"):
        self.model.train()
        step = 0
        running_loss = 0.0
        t0 = time.time()

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        while step < self.max_steps:
            for batch_idx, (x, y) in enumerate(loader):
                if step >= self.max_steps:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                lr = self._get_lr(step, self.optimizer.defaults["lr"])
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                use_amp = self.scaler is not None
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(input_ids=x, labels=y)
                    loss = outputs["loss"] / self.grad_accum_steps

                if use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * self.grad_accum_steps

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    step += 1
                    if step % 10 == 0:
                        dt = time.time() - t0
                        loss_avg = running_loss / 10
                        mfu = self.model.estimate_mfu(self.grad_accum_steps, dt / 10) if hasattr(self.model, "estimate_mfu") else 0
                        logger.info(
                            f"{label} | step {step}/{self.max_steps} | loss {loss_avg:.4f} | lr {lr:.2e} | mfu {mfu*100:.1f}%"
                        )
                        running_loss = 0.0
                        t0 = time.time()

                    if step % 100 == 0:
                        self._save_checkpoint(save_dir, step)

        self._save_checkpoint(save_dir, step, final=True)
        logger.info(f"Training complete. Saved to {save_dir}")

    def _save_checkpoint(self, save_dir: str, step: int, final: bool = False):
        suffix = "final" if final else f"step-{step}"
        path = Path(save_dir) / f"checkpoint-{suffix}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    "vocab_size": self.model.config.vocab_size,
                    "block_size": self.model.config.block_size,
                    "n_layer": self.model.config.n_layer,
                    "n_head": self.model.config.n_head,
                    "n_embd": self.model.config.n_embd,
                },
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")


def load_hf_code_datasets(tokenizer: CodeTokenizer, max_samples: int = 5000) -> List[str]:
    """Load real code instruction datasets from HuggingFace for pre-training."""
    texts = []
    try:
        from datasets import load_dataset
        logger.info("Loading code datasets from HuggingFace for pre-training...")

        datasets_to_try = [
            ("sahil2801/CodeAlpaca-20k", None, "train", "alpaca"),
            ("iamtarun/python_code_instructions_18k_alpaca", None, "train", "alpaca"),
            ("timdettmers/openassistant-guanaco", None, "train", "guanaco"),
        ]

        for name, config, split, fmt in datasets_to_try:
            try:
                ds = load_dataset(name, config, split=f"{split}[:{max_samples}]", trust_remote_code=True)
                for item in ds:
                    if fmt == "alpaca":
                        instruction = item.get("instruction", "")
                        inp = item.get("input", "")
                        output = item.get("output", "")
                        text = f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n{output}"
                    elif fmt == "guanaco":
                        text = item.get("text", "")
                    else:
                        text = item.get("text", item.get("content", ""))
                    texts.append(text)
                logger.info(f"  Loaded {len(ds)} samples from {name}")
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")

    except ImportError:
        logger.warning("datasets library not available — skipping HF dataset loading")

    return texts


def seed_corpus_for_tokenizer(save_path: str, min_size_mb: int = 10) -> List[str]:
    """Generate a seed corpus for tokenizer training from local Python files."""
    import tempfile

    files = []
    # Search repo itself for code
    repo_root = Path(__file__).parent.parent.parent
    for ext in [".py", ".yaml", ".json", ".md"]:
        for p in repo_root.rglob(f"*{ext}"):
            if ".venv" in str(p) or "__pycache__" in str(p):
                continue
            files.append(str(p))

    # Also create synthetic code samples
    synthetic_dir = Path(save_path) / "synthetic_corpus"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    python_snippets = [
        # ... hundreds of diverse Python patterns
    ]

    # Write a large synthetic corpus
    corpus_file = synthetic_dir / "corpus.txt"
    with open(corpus_file, "w") as f:
        for _ in range(5000):
            f.write(generate_synthetic_code())
            f.write("\n\n")
    files.append(str(corpus_file))

    return files


def generate_synthetic_code() -> str:
    """Generate a random Python-like code snippet for tokenizer training."""
    import random

    templates = [
        "def {name}({args}):\n    {body}\n",
        "class {Name}:\n    def __init__(self, {args}):\n        {init_body}\n",
        "import {module}\n",
        "from {module} import {name}\n",
        "for {var} in {iterable}:\n    {body}\n",
        "if {condition}:\n    {body}\nelse:\n    {else_body}\n",
        "try:\n    {body}\nexcept {exc}:\n    {handler}\n",
        "async def {name}({args}):\n    {body}\n",
        "@decorator\ndef {name}({args}):\n    {body}\n",
        "{var} = {expr}\n",
        "return {expr}\n",
        "raise {exc}({msg})\n",
        "with open({path}) as f:\n    {body}\n",
        "[{expr} for {var} in {iterable}]\n",
        "{{k: v for k, v in {iterable}}}\n",
    ]

    names = ["foo", "bar", "baz", "process_data", "train_model", "evaluate", "predict", "transform", "compute_loss", "forward", "backward", "optimize", "get_batch", "save_checkpoint", "load_weights"]
    vars = ["x", "y", "z", "data", "model", "loss", "output", "inputs", "labels", "params", "grads", "batch", "config", "state"]
    modules = ["torch", "numpy", "pandas", "transformers", "datasets", "os", "sys", "json", "math", "typing"]
    types = ["int", "str", "float", "list", "dict", "tuple", "Optional", "List", "Dict", "Any"]

    tmpl = random.choice(templates)
    return tmpl.format(
        name=random.choice(names),
        Name=random.choice(names).capitalize(),
        args=", ".join(f"{v}: {random.choice(types)}" for v in random.sample(vars, k=random.randint(1, 4))),
        body=random.choice(["pass", "return None", "raise NotImplementedError()", "...", f"return {random.choice(vars)}"]),
        init_body="\n        ".join(f"self.{v} = {v}" for v in random.sample(vars, k=random.randint(1, 3))),
        module=random.choice(modules),
        var=random.choice(vars),
        iterable=random.choice(["range(10)", "data.items()", "enumerate(batch)", "zip(x, y)"]),
        condition=random.choice(["x > 0", "len(data) > 0", "model is not None", "loss < threshold"]),
        else_body=random.choice(["pass", "return None"]),
        exc=random.choice(["ValueError", "RuntimeError", "KeyError", "TypeError"]),
        handler=random.choice(["pass", "logger.error(e)", "return None"]),
        expr=random.choice(["x + y", "model(inputs)", "loss.item()", "[1, 2, 3]"]),
        msg='"error"',
        path='"data.json"',
    )
