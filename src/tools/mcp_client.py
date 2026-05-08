"""Model Context Protocol (MCP) client for tool use."""

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

from ..core.config import Config

logger = logging.getLogger(__name__)


class MCPServer:
    """One connected MCP server process."""

    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []

    def start(self) -> None:
        logger.info(f"Starting MCP server: {self.name}")
        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # MCP uses JSON-RPC over stdio; we'd do the handshake here
        # Simplified: just log that it's started

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool via JSON-RPC over stdio."""
        if not self.process or self.process.poll() is not None:
            return {"error": "Server not running"}
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        line = json.dumps(req) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()
        resp = self.process.stdout.readline()
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return {"error": "Invalid response", "raw": resp}


class MCPClient:
    """Manages multiple MCP servers and exposes them as agent tools."""

    def __init__(self, config: Config):
        self.servers: Dict[str, MCPServer] = {}
        for srv_cfg in config.mcp.get("servers", []):
            name = srv_cfg["name"]
            self.servers[name] = MCPServer(
                name=name,
                command=srv_cfg["command"],
                args=srv_cfg.get("args", []),
            )

    def start_all(self) -> None:
        for srv in self.servers.values():
            srv.start()

    def stop_all(self) -> None:
        for srv in self.servers.values():
            srv.stop()

    def list_tools(self) -> List[Dict[str, str]]:
        tools = []
        for name, srv in self.servers.items():
            for t in srv.tools:
                tools.append({
                    "server": name,
                    "tool": t.get("name", "unknown"),
                    "description": t.get("description", ""),
                })
        return tools

    def call(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        srv = self.servers.get(server_name)
        if not srv:
            return {"error": f"Server {server_name} not found"}
        return srv.call_tool(tool_name, arguments)
