"""Language Server Protocol client for code intelligence."""

import json
import logging
import subprocess
import threading
from typing import Any, Dict, List, Optional

from ..core.config import Config

logger = logging.getLogger(__name__)


class LSPClient:
    """
    Lightweight LSP client for diagnostics, hover, and completions.
    Currently supports python-lsp-server for Python and tsserver for TS.
    """

    def __init__(self, config: Config):
        self.cfg = config.lsp
        self.python_server_cmd = self.cfg.get("python_server", "pylsp")
        self.ts_server_cmd = self.cfg.get("typescript_server", "typescript-language-server")
        self.timeout = self.cfg.get("startup_timeout", 10)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self, language: str = "python") -> None:
        """Start the language server."""
        cmd = {
            "python": [self.python_server_cmd],
            "typescript": [self.ts_server_cmd, "--stdio"],
        }.get(language)

        if not cmd:
            logger.warning(f"No LSP server configured for {language}")
            return

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Send initialize request
        self._send_request("initialize", {
            "processId": None,
            "rootUri": None,
            "capabilities": {},
        })
        logger.info(f"LSP server started: {language}")

    def stop(self) -> None:
        if self._proc:
            self._send_notification("exit", {})
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def get_diagnostics(self, uri: str, source: str) -> List[Dict[str, Any]]:
        """Publish diagnostics for a document."""
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": 1,
                "text": source,
            }
        })
        # Some servers push diagnostics; we'd need to read async
        # Simplified: return empty and rely on our linter instead
        return []

    def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return None
            req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            }
            line = json.dumps(req) + "\r\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            # Read response (simplified)
            resp = self._proc.stdout.readline()
            try:
                return json.loads(resp)
            except json.JSONDecodeError:
                return None

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return
            note = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
            line = json.dumps(note) + "\r\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
