"""Real-time GPU monitoring for ASSBRAIN."""

import threading
import time
from typing import Optional

import torch
from rich.console import Console
from rich.table import Table

console = Console()


class GPUMonitor:
    """Monitor GPU utilization and memory in a background thread."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_mem_mb = 0.0

    def start(self):
        if not torch.cuda.is_available():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        while self._running:
            try:
                self._snapshot()
            except Exception:
                pass
            time.sleep(self.interval)

    def _snapshot(self):
        dev = torch.cuda.current_device()
        mem_alloc = torch.cuda.memory_allocated(dev) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(dev) / 1024**2
        mem_total = torch.cuda.get_device_properties(dev).total_memory / 1024**2
        self._peak_mem_mb = max(self._peak_mem_mb, mem_alloc)

        # utilization via nvidia-ml-py if available, otherwise just memory
        util = "N/A"
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev)
            util_obj = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util = f"{util_obj.gpu}%"
        except Exception:
            pass

        console.print(
            f"[dim]GPU {dev} | Util: {util} | Mem: {mem_alloc:.0f}/{mem_total:.0f} MB "
            f"(Reserved: {mem_reserved:.0f} MB) | Peak: {self._peak_mem_mb:.0f} MB[/dim]",
            end="\r",
        )

    def print_summary(self):
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        mem_total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
        table = Table(title="GPU Session Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Device", torch.cuda.get_device_name(dev))
        table.add_row("Total VRAM", f"{mem_total:.1f} GB")
        table.add_row("Peak Usage", f"{self._peak_mem_mb:.0f} MB")
        console.print(table)
