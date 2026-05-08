#!/usr/bin/env python3
"""GPU-aware Python installer for ASSBRAIN.

Run this instead of `pip install -r requirements.txt`:
    python install.py

This script will:
1. Create a virtualenv (if not already inside one)
2. Detect NVIDIA GPU and install CUDA-enabled PyTorch
3. Install GPU or CPU FAISS accordingly
4. Install all remaining Python dependencies
5. Install Playwright browsers
6. Run GPU validation
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
VENV_DIR = REPO_ROOT / ".venv"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result


def in_virtualenv():
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.getenv("VIRTUAL_ENV")
    )


def create_venv():
    if VENV_DIR.exists():
        print(f"Using existing venv at {VENV_DIR}")
        return
    print(f"Creating virtualenv at {VENV_DIR}...")
    venv.create(VENV_DIR, with_pip=True)


def get_python():
    if in_virtualenv():
        return sys.executable
    return str(VENV_DIR / "bin" / "python")


def get_pip():
    if in_virtualenv():
        return [sys.executable, "-m", "pip"]
    return [str(VENV_DIR / "bin" / "python"), "-m", "pip"]


def has_nvidia_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_torch(pip, gpu):
    if gpu:
        print("Installing PyTorch with CUDA 12.1 support...")
        run(
            pip
            + [
                "install",
                "torch>=2.1.0",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    else:
        print("No NVIDIA GPU detected. Installing CPU-only PyTorch...")
        run(
            pip
            + [
                "install",
                "torch>=2.1.0",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )


def install_faiss(pip, gpu):
    if gpu:
        print("Installing FAISS GPU (CUDA 12)...")
        run(pip + ["install", "faiss-gpu-cu12>=1.14.0"])
    else:
        print("Installing FAISS CPU...")
        run(pip + ["install", "faiss-cpu>=1.7.4"])


def install_requirements(pip):
    req_file = REPO_ROOT / "requirements.txt"
    lines = req_file.read_text().splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip().lower()
        # Skip torch/faiss lines so we don't override our GPU installs
        if any(pkg in stripped for pkg in ["torch", "torchvision", "torchaudio", "faiss-cpu", "faiss-gpu"]):
            if not stripped.startswith("#"):
                continue
        filtered.append(line)
    temp_req = REPO_ROOT / "._req_temp.txt"
    temp_req.write_text("\n".join(filtered))
    try:
        run(pip + ["install", "-r", str(temp_req)])
    finally:
        temp_req.unlink()


def install_playwright(python):
    print("Installing Playwright browsers...")
    run([python, "-m", "playwright", "install", "chromium"])


def validate(python):
    print("\nRunning GPU validation...")
    run([python, str(REPO_ROOT / "validate_gpu.py")])


def main():
    print("=" * 50)
    print("ASSBRAIN Installer")
    print("=" * 50)

    if not in_virtualenv():
        create_venv()
    else:
        print(f"Already in virtualenv: {sys.prefix}")

    pip = get_pip()
    python = get_python()

    run(pip + ["install", "--upgrade", "pip", "wheel", "setuptools"])

    gpu = has_nvidia_gpu()
    print(f"NVIDIA GPU detected: {gpu}\n")

    install_torch(pip, gpu)
    install_faiss(pip, gpu)
    install_requirements(pip)
    install_playwright(python)

    print("\n" + "=" * 50)
    print("Installation complete!")
    print("=" * 50)

    validate(python)

    if not in_virtualenv():
        print(f"\nTo activate your environment, run:")
        print(f"  source {VENV_DIR}/bin/activate")
        print(f"\nThen run ASSBRAIN with:")
        print(f"  python run.py status")
        print(f"  python run.py learn --episodes 5")


if __name__ == "__main__":
    main()
