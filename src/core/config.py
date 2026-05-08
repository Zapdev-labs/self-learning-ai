"""Configuration management for ASSBRAIN."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized config with dot-access and dict-access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access: config.get('llm.temperature')."""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"

    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()

    @property
    def llm(self) -> Dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def learning(self) -> Dict[str, Any]:
        return self._data.get("learning", {})

    @property
    def memory(self) -> Dict[str, Any]:
        return self._data.get("memory", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self._data.get("evaluation", {})

    @property
    def browser(self) -> Dict[str, Any]:
        return self._data.get("browser", {})

    @property
    def lsp(self) -> Dict[str, Any]:
        return self._data.get("lsp", {})

    @property
    def mcp(self) -> Dict[str, Any]:
        return self._data.get("mcp", {})

    @property
    def generation(self) -> Dict[str, Any]:
        return self._data.get("generation", {})

    @property
    def pi(self) -> Dict[str, Any]:
        return self._data.get("pi", {})

    @property
    def data_dir(self) -> Path:
        return Path(self.get("app.data_dir", "./data")).resolve()

    @property
    def memory_dir(self) -> Path:
        return Path(self.get("app.memory_dir", "./memory")).resolve()

    @property
    def models_dir(self) -> Path:
        return Path(self.get("app.models_dir", "./models")).resolve()

    @property
    def projects_dir(self) -> Path:
        return Path(self.get("app.projects_dir", "./projects")).resolve()

    @property
    def logs_dir(self) -> Path:
        return Path(self.get("app.logs_dir", "./logs")).resolve()


def load_config(path: Optional[str] = None) -> Config:
    """Load config from YAML, overlay with env vars."""
    if path is None:
        # Try default locations
        candidates = [
            os.getenv("ASSBRAIN_CONFIG"),
            "./config/local.yaml",
            "./config/default.yaml",
        ]
        for c in candidates:
            if c and Path(c).exists():
                path = c
                break
    if path is None or not Path(path).exists():
        raise FileNotFoundError("No config file found. Set ASSBRAIN_CONFIG or create config/default.yaml")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Overlay HF token if available
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        if "llm" not in data:
            data["llm"] = {}
        data["llm"]["huggingface_token"] = hf_token

    return Config(data)
