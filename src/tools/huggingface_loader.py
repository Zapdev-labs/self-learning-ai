"""Dataset and model loading from HuggingFace Hub."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel

from ..core.config import Config

logger = logging.getLogger(__name__)


class HuggingFaceLoader:
    """Loads datasets and models from HuggingFace for training and inference."""

    def __init__(self, config: Config):
        self.token = config.llm.get("huggingface_token") or os.getenv("HF_TOKEN")
        self.cache_dir = Path("./data/huggingface_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(
        self,
        name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        streaming: bool = False,
        **kwargs
    ) -> Union[Dataset, DatasetDict]:
        """Load a HuggingFace dataset."""
        logger.info(f"Loading dataset: {name} (config={config_name}, split={split})")
        ds = load_dataset(
            name,
            config_name,
            split=split,
            streaming=streaming,
            cache_dir=str(self.cache_dir),
            token=self.token,
            **kwargs
        )
        logger.info(f"Dataset loaded: {len(ds) if hasattr(ds, '__len__') else 'streaming'} samples")
        return ds

    def load_code_datasets(self) -> Dict[str, Any]:
        """Load a curated set of code/ML datasets for initial training."""
        datasets = {}
        try:
            datasets["code_alpaca"] = self.load_dataset(
                "sahil2801/CodeAlpaca-20k", split="train[:1000]"
            )
        except Exception as e:
            logger.warning(f"Could not load CodeAlpaca: {e}")

        try:
            datasets["openassistant"] = self.load_dataset(
                "timdettmers/openassistant-guanaco", split="train[:1000]"
            )
        except Exception as e:
            logger.warning(f"Could not load OpenAssistant: {e}")

        try:
            datasets["python_code"] = self.load_dataset(
                "iamtarun/python_code_instructions_18k_alpaca", split="train[:1000]"
            )
        except Exception as e:
            logger.warning(f"Could not load Python code dataset: {e}")

        return datasets

    def format_for_training(self, dataset: Dataset, format_type: str = "alpaca") -> List[Dict[str, str]]:
        """Convert a dataset into prompt-response pairs for fine-tuning."""
        examples = []
        for item in dataset:
            if format_type == "alpaca":
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
                examples.append({"prompt": prompt, "chosen": output})
            elif format_type == "guanaco":
                text = item.get("text", "")
                # Split on response marker
                if "### Assistant:" in text:
                    parts = text.split("### Assistant:")
                    prompt = parts[0] + "### Assistant:\n"
                    chosen = parts[1].strip()
                    examples.append({"prompt": prompt, "chosen": chosen})
            else:
                # Generic fallback
                prompt = item.get("prompt", item.get("question", ""))
                chosen = item.get("completion", item.get("answer", item.get("response", "")))
                examples.append({"prompt": prompt, "chosen": chosen})
        return examples

    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load a sentence transformer model for embeddings."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
