"""Train a custom BPE tokenizer for ASSBRAIN's multimodal model."""

import json
import os
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


class CodeTokenizer:
    """Byte-Pair Encoding tokenizer trained on code with multimodal support."""

    SPECIAL_TOKENS = [
        "<pad>", "<unk>", "<s>", "</s>",
        "<|im_start|>", "<|im_end|>",
        "<|user|>", "<|assistant|>", "<|system|>",
        "<|image_start|>", "<|image_end|>",
        "<|tool_call_begin|>", "<|tool_call_end|>",
        "<|tool_result_begin|>", "<|tool_result_end|>",
    ]

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._tok: Optional[Tokenizer] = None

    def train(self, files: List[str], save_dir: str) -> str:
        """Train BPE tokenizer on provided text files."""
        os.makedirs(save_dir, exist_ok=True)

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train(files, trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )

        save_path = Path(save_dir) / "tokenizer.json"
        tokenizer.save(str(save_path))
        self._tok = tokenizer
        return str(save_path)

    def load(self, path: str):
        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[int]:
        if self._tok is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self._tok is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self._tok is None:
            raise RuntimeError("Tokenizer not loaded")
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    @property
    def vocab_size(self) -> int:
        if self._tok is None:
            return self._vocab_size
        return self._tok.get_vocab_size()

    @vocab_size.setter
    def vocab_size(self, value: int):
        self._vocab_size = value

    def _get_id(self, token: str, default: int = 0) -> int:
        if self._tok is None:
            return default
        try:
            return self._tok.token_to_id(token)
        except Exception:
            return default

    @property
    def pad_token_id(self) -> int:
        return self._get_id("<pad>", 0)

    @property
    def eos_token_id(self) -> int:
        return self._get_id("</s>", 2)

    @property
    def unk_token_id(self) -> int:
        return self._get_id("<unk>", 1)

    @property
    def image_start_token_id(self) -> int:
        return self._get_id("<|image_start|>", 32000)

    @property
    def image_end_token_id(self) -> int:
        return self._get_id("<|image_end|>", 32001)

    @property
    def tool_call_begin_id(self) -> int:
        return self._get_id("<|tool_call_begin|>", 32002)

    @property
    def tool_call_end_id(self) -> int:
        return self._get_id("<|tool_call_end|>", 32003)

    @property
    def tool_result_begin_id(self) -> int:
        return self._get_id("<|tool_result_begin|>", 32004)

    @property
    def tool_result_end_id(self) -> int:
        return self._get_id("<|tool_result_end|>", 32005)

    def save_config(self, path: str):
        if self._tok is None:
            return
        vocab = self._tok.get_vocab()
        with open(path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)


def collect_code_files(root_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".md", ".html", ".css", ".rs", ".go", ".java", ".cpp", ".c", ".h"]
    files = []
    root = Path(root_dir)
    for ext in extensions:
        files.extend(str(p) for p in root.rglob(f"*{ext}"))
    return files
