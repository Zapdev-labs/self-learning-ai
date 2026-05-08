"""Local LLM engine with generation, fine-tuning, and pi integration."""

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
from datasets import Dataset
import threading

from .config import Config

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages local LLM loading, inference, and incremental fine-tuning."""

    def __init__(self, config: Config):
        self.cfg = config.llm
        self.model_id = self.cfg.get("model_id", "microsoft/DialoGPT-medium")
        self.device = self._resolve_device(self.cfg.get("device", "auto"))
        self.context_window = self.cfg.get("context_window", 4096)
        self.max_new_tokens = self.cfg.get("max_new_tokens", 2048)
        self.temperature = self.cfg.get("temperature", 0.7)
        self.top_p = self.cfg.get("top_p", 0.9)
        self.top_k = self.cfg.get("top_k", 50)
        self.repetition_penalty = self.cfg.get("repetition_penalty", 1.1)
        self.token = self.cfg.get("huggingface_token") or os.getenv("HF_TOKEN")

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.base_model: Optional[Any] = None
        self._is_loaded = False
        self._lora_active = False

        self.load_model()

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

    def load_model(self) -> None:
        """Load base model and tokenizer from HuggingFace Hub or local cache."""
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
        if self.device == "cpu" or self.device == "mps":
            self.model = self.model.to(self.device)

        self.base_model = self.model
        self._is_loaded = True
        logger.info("Model loaded successfully.")

    def apply_lora(self) -> None:
        """Apply LoRA adapters for efficient fine-tuning."""
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text completion. Returns string or stream iterator."""
        if not self._is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded")

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
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("Assistant:")
        return self.generate("\n\n".join(prompt_parts), **gen_kwargs)

    def train_on_experiences(
        self,
        experiences: List[Dict[str, str]],
        output_dir: Optional[str] = None,
    ) -> str:
        """Fine-tune LoRA on successful experiences (self-imitation learning)."""
        if not self._lora_active:
            self.apply_lora()
        assert self.model is not None and self.tokenizer is not None

        texts = []
        for exp in experiences:
            # Format: prompt -> chosen (good) response
            prompt = exp.get("prompt", "")
            chosen = exp.get("chosen", "")
            texts.append(f"{prompt}\n\nAssistant: {chosen}")

        ds = Dataset.from_dict({"text": texts})

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

        logger.info(f"Starting training on {len(experiences)} experiences...")
        trainer.train()

        adapter_path = os.path.join(out, f"adapter-{uuid.uuid4().hex[:8]}")
        self.model.save_pretrained(adapter_path)
        logger.info(f"Adapter saved to {adapter_path}")
        return adapter_path

    def merge_and_unload(self) -> None:
        """Merge LoRA weights back into base model."""
        if self._lora_active and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            self._lora_active = False
            logger.info("LoRA merged into base model.")

    def save_full_model(self, path: str) -> None:
        """Save complete model + tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        self.merge_and_unload()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        logger.info(f"Full model saved to {out}")

    def load_adapter(self, adapter_path: str) -> None:
        """Load a previously saved LoRA adapter."""
        if self.model is None:
            raise RuntimeError("Base model not loaded")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self._lora_active = True
        logger.info(f"Loaded adapter from {adapter_path}")

    @property
    def is_ready(self) -> bool:
        return self._is_loaded

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "lora_active": self._lora_active,
            "context_window": self.context_window,
            "dtype": str(self.model.dtype) if self.model else "N/A",
        }
