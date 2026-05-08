"""Local LLM engine with generation, fine-tuning, and pi integration.

Supports two modes:
  - "pretrained": Load a model from HuggingFace Hub (original behavior)
  - "scratch": Train a custom transformer from scratch (new default for ASSBRAIN)
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import threading

from .config import Config
from .custom_transformer import AssBrainTransformer, ModelConfig
from .custom_trainer import CustomTrainer, seed_corpus_for_tokenizer
from .tokenizer_trainer import CodeTokenizer, collect_code_files

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages local LLM loading, inference, and incremental fine-tuning."""

    def __init__(self, config: Config):
        self.cfg = config.llm
        self.device = self._resolve_device(self.cfg.get("device", "auto"))
        self.mode = self.cfg.get("mode", "pretrained")  # "pretrained" | "scratch"
        self.model_id = self.cfg.get("model_id", "microsoft/DialoGPT-medium")
        self.context_window = self.cfg.get("context_window", 4096)
        self.max_new_tokens = self.cfg.get("max_new_tokens", 2048)
        self.temperature = self.cfg.get("temperature", 0.7)
        self.top_p = self.cfg.get("top_p", 0.9)
        self.top_k = self.cfg.get("top_k", 50)
        self.repetition_penalty = self.cfg.get("repetition_penalty", 1.1)
        self.token = self.cfg.get("huggingface_token") or os.getenv("HF_TOKEN")

        # Model state
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None          # HF tokenizer (pretrained mode)
        self.custom_tokenizer: Optional[CodeTokenizer] = None  # Custom tokenizer (scratch mode)
        self.base_model: Optional[Any] = None
        self._is_loaded = False
        self._lora_active = False
        self._custom_trainer: Optional[CustomTrainer] = None

        # Paths
        self.models_dir = Path(config.get("app.models_dir", "./models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.models_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()

    # ------------------------------------------------------------------
    # Device & Quantization
    # ------------------------------------------------------------------

    def _resolve_device(self, device_str: str) -> str:
        if device_str == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_str

    def _quantization_config(self) -> Optional[BitsAndBytesConfig]:
        load_8bit = self.cfg.get("load_in_8bit", False)
        load_4bit = self.cfg.get("load_in_4bit", False)
        if load_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if load_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self.mode == "scratch":
            self._load_scratch_model()
        else:
            self._load_pretrained_model()

    def _load_scratch_model(self):
        """Initialize or load a custom transformer from scratch."""
        logger.info("Initializing custom ASSBRAIN transformer from scratch...")

        # Load or train custom tokenizer
        tok_path = self.models_dir / "custom_tokenizer" / "tokenizer.json"
        self.custom_tokenizer = CodeTokenizer(vocab_size=self.cfg.get("vocab_size", 32000))

        if tok_path.exists():
            logger.info(f"Loading existing tokenizer from {tok_path}")
            self.custom_tokenizer.load(str(tok_path))
        else:
            logger.info("Training new tokenizer on seed corpus...")
            corpus_files = seed_corpus_for_tokenizer(str(self.models_dir / "corpus"))
            if not corpus_files:
                logger.warning("No corpus files found, using synthetic data")
                corpus_files = seed_corpus_for_tokenizer(str(self.models_dir / "corpus"))
            tok_save = self.models_dir / "custom_tokenizer"
            self.custom_tokenizer.train(corpus_files, str(tok_save))
            self.custom_tokenizer.save_config(str(tok_save / "vocab.json"))

        vocab_size = self.custom_tokenizer.vocab_size
        logger.info(f"Tokenizer vocab size: {vocab_size}")

        # Build model config from YAML
        mcfg = ModelConfig(
            vocab_size=vocab_size,
            block_size=self.cfg.get("block_size", 4096),
            n_layer=self.cfg.get("n_layer", 24),
            n_head=self.cfg.get("n_head", 16),
            n_embd=self.cfg.get("n_embd", 1024),
            dropout=self.cfg.get("dropout", 0.0),
            ffn_mult=self.cfg.get("ffn_mult", 4.0),
            multiple_of=self.cfg.get("multiple_of", 256),
            tie_weights=self.cfg.get("tie_weights", True),
        )

        logger.info(
            f"Model config: {mcfg.n_layer} layers, {mcfg.n_head} heads, "
            f"{mcfg.n_embd} dim, ~{mcfg.estimate_params() / 1e6:.0f}M params"
        )

        self.model = AssBrainTransformer(mcfg).to(self.device)
        param_count = self.model.get_num_params()
        logger.info(f"Model initialized with {param_count / 1e6:.1f}M parameters on {self.device}")

        # Try to load checkpoint
        latest_ckpt = self._find_latest_checkpoint()
        if latest_ckpt:
            logger.info(f"Resuming from checkpoint: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint at step {ckpt.get('step', 'unknown')}")
        else:
            logger.info("No checkpoint found — starting from random initialization")

        self._is_loaded = True

    def _load_pretrained_model(self):
        """Load a pre-trained model from HuggingFace (original behavior)."""
        logger.info(f"Loading model {self.model_id} on {self.device}...")
        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "token": self.token,
        }
        qcfg = self._quantization_config()
        if qcfg:
            kwargs["quantization_config"] = qcfg

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.token, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, **kwargs
        )
        if self.device in ("cpu", "mps"):
            self.model = self.model.to(self.device)

        self.base_model = self.model
        self._is_loaded = True
        logger.info("Pretrained model loaded successfully.")

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint-*.pt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            return None
        # Prefer final, then latest by mtime
        for c in ckpts:
            if "final" in c.name:
                return str(c)
        return str(ckpts[-1])

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if self.mode == "scratch":
            return self._generate_scratch(prompt, system_prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_pretrained(prompt, system_prompt, max_new_tokens, temperature, top_p, stream)

    def _generate_scratch(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        assert self.custom_tokenizer is not None and self.model is not None

        # Build chat prompt
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(parts)

        input_ids = torch.tensor([self.custom_tokenizer.encode(full_prompt)], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.custom_tokenizer.eos_token_id,
            pad_token_id=self.custom_tokenizer.pad_token_id,
        )

        # Decode only the new tokens
        new_ids = output_ids[0, input_ids.size(1):].tolist()
        text = self.custom_tokenizer.decode(new_ids)
        # Strip any remaining special tokens
        for tok in ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
            text = text.replace(tok, "").strip()
        return text

    def _generate_pretrained(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        assert self.tokenizer is not None and self.model is not None

        full_prompt = self._build_prompt(prompt, system_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.context_window)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
            thread.start()
            return streamer

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(full_prompt):].strip()

    def _build_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        return f"User: {user_prompt}\n\nAssistant:"

    def chat(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        """OpenAI-style chat completion."""
        if self.mode == "scratch":
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)
            return self._generate_scratch(prompt, None, **gen_kwargs)
        else:
            prompt_parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("Assistant:")
            return self._generate_pretrained("\n\n".join(prompt_parts), None, **gen_kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def apply_lora(self) -> None:
        """Apply LoRA adapters (pretrained mode only)."""
        if self.mode == "scratch":
            logger.info("LoRA not needed in scratch mode — training full model")
            return
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        lora_cfg = self.cfg.get("lora", {})
        config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        if self.cfg.get("load_in_8bit") or self.cfg.get("load_in_4bit"):
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, config)
        self._lora_active = True
        logger.info("LoRA adapters applied.")

    def train_on_experiences(
        self,
        experiences: List[Dict[str, str]],
        output_dir: Optional[str] = None,
    ) -> str:
        """Train on successful experiences."""
        if self.mode == "scratch":
            return self._train_scratch(experiences, output_dir)
        else:
            return self._train_pretrained(experiences, output_dir)

    def _train_scratch(self, experiences: List[Dict[str, str]], output_dir: Optional[str] = None) -> str:
        assert self.model is not None and self.custom_tokenizer is not None

        tcfg = self.cfg.get("training", {})
        out = output_dir or str(self.checkpoint_dir)

        if self._custom_trainer is None:
            self._custom_trainer = CustomTrainer(
                model=self.model,
                tokenizer=self.custom_tokenizer,
                device=self.device,
                learning_rate=tcfg.get("learning_rate", 1e-4),
                batch_size=tcfg.get("per_device_train_batch_size", 4),
                grad_accum_steps=tcfg.get("gradient_accumulation_steps", 4),
                max_steps=tcfg.get("max_steps", 200),
                warmup_steps=tcfg.get("warmup_steps", 20),
                compile_model=self.cfg.get("compile", True),
            )

        self._custom_trainer.train_on_experiences(experiences, out)
        return out

    def _train_pretrained(self, experiences: List[Dict[str, str]], output_dir: Optional[str] = None) -> str:
        """Original HF Trainer-based fine-tuning."""
        if not self._lora_active:
            self.apply_lora()
        assert self.model is not None and self.tokenizer is not None

        texts = []
        for exp in experiences:
            prompt = exp.get("prompt", "")
            chosen = exp.get("chosen", "")
            texts.append(f"{prompt}\n\nAssistant: {chosen}")

        ds = HFDataset.from_dict({"text": texts})

        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.context_window,
                padding="max_length",
            )

        tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        tcfg = self.cfg.get("training", {})
        out = output_dir or tcfg.get("output_dir", "./models/checkpoints")
        Path(out).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=out,
            num_train_epochs=tcfg.get("num_train_epochs", 1),
            per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
            learning_rate=tcfg.get("learning_rate", 2e-4),
            max_steps=tcfg.get("max_steps", 100),
            warmup_steps=tcfg.get("warmup_steps", 10),
            logging_steps=tcfg.get("logging_steps", 5),
            save_steps=tcfg.get("save_steps", 50),
            fp16=self.device == "cuda",
            optim="paged_adamw_8bit" if self.cfg.get("load_in_8bit") else "adamw_torch",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        logger.info(f"Starting HF training on {len(experiences)} experiences...")
        trainer.train()

        adapter_path = os.path.join(out, f"adapter-{uuid.uuid4().hex[:8]}")
        self.model.save_pretrained(adapter_path)
        logger.info(f"Adapter saved to {adapter_path}")
        return adapter_path

    def save_full_model(self, path: str) -> None:
        """Save complete model + tokenizer."""
        if self.mode == "scratch":
            assert self.model is not None and self.custom_tokenizer is not None
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": {
                        "vocab_size": self.model.config.vocab_size,
                        "block_size": self.model.config.block_size,
                        "n_layer": self.model.config.n_layer,
                        "n_head": self.model.config.n_head,
                        "n_embd": self.model.config.n_embd,
                        "dropout": self.model.config.dropout,
                        "ffn_mult": self.model.config.ffn_mult,
                        "multiple_of": self.model.config.multiple_of,
                        "tie_weights": self.model.config.tie_weights,
                    },
                },
                out / "model.pt",
            )
            self.custom_tokenizer.save_config(str(out / "vocab.json"))
            logger.info(f"Full scratch model saved to {out}")
        else:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded")
            self.merge_and_unload()
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(out)
            self.tokenizer.save_pretrained(out)
            logger.info(f"Full pretrained model saved to {out}")

    def merge_and_unload(self) -> None:
        """Merge LoRA weights back into base model."""
        if self.mode == "scratch":
            return
        if self._lora_active and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            self._lora_active = False
            logger.info("LoRA merged into base model.")

    def load_adapter(self, adapter_path: str) -> None:
        """Load a previously saved LoRA adapter."""
        if self.mode == "scratch":
            logger.warning("Adapters not supported in scratch mode")
            return
        if self.model is None:
            raise RuntimeError("Base model not loaded")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self._lora_active = True
        logger.info(f"Loaded adapter from {adapter_path}")

    # ------------------------------------------------------------------
    # Properties & Stats
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._is_loaded

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "model_id": self.model_id,
            "device": self.device,
            "mode": self.mode,
            "lora_active": self._lora_active,
            "context_window": self.context_window,
        }
        if self.mode == "scratch" and self.model is not None:
            stats["params_M"] = round(self.model.get_num_params() / 1e6, 1)
            stats["dtype"] = str(next(self.model.parameters()).dtype)
        elif self.model is not None:
            stats["dtype"] = str(self.model.dtype)
        return stats
