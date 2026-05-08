#!/usr/bin/env python3
"""Validate that ASSBRAIN can use GPU acceleration.

Run after install.py or any time to verify your setup:
    python validate_gpu.py
"""

import sys


def check():
    ok = True
    print("=" * 50)
    print("ASSBRAIN GPU Validation")
    print("=" * 50)

    print("\n1. PyTorch")
    try:
        import torch
        print(f"   Version: {torch.__version__}")
    except ImportError:
        print("   ERROR: PyTorch not installed")
        return False

    print("\n2. CUDA / GPU")
    if torch.cuda.is_available():
        print(f"   CUDA available: YES")
        print(f"   CUDA version:   {torch.version.cuda}")
        print(f"   GPU device:     {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU memory:     {mem_gb:.1f} GB")

        # Quick matmul smoke test
        x = torch.rand(1000, 1000, device="cuda")
        y = torch.mm(x, x)
        print(f"   GPU tensor op:  PASSED (device={y.device})")
    else:
        print("   CUDA available: NO — will fall back to CPU")
        print("   (If you have an NVIDIA GPU, check drivers and CUDA toolkit)")

    print("\n3. Transformers")
    try:
        import transformers
        print(f"   Version: {transformers.__version__}")
    except ImportError:
        print("   ERROR: Transformers not installed")
        ok = False

    print("\n4. Accelerate")
    try:
        import accelerate
        print(f"   Version: {accelerate.__version__}")
    except ImportError:
        print("   ERROR: Accelerate not installed")
        ok = False

    print("\n5. PEFT (LoRA)")
    try:
        import peft
        print(f"   Version: {peft.__version__}")
    except ImportError:
        print("   ERROR: PEFT not installed")
        ok = False

    print("\n6. BitsAndBytes (4-bit / 8-bit quantization)")
    try:
        import bitsandbytes as bnb
        print(f"   Version: {bnb.__version__}")
    except ImportError:
        print("   WARNING: Not installed (only needed for quantization)")

    print("\n7. Sentence-Transformers (embeddings)")
    try:
        import sentence_transformers
        print(f"   Version: {sentence_transformers.__version__}")
    except ImportError:
        print("   ERROR: Not installed")
        ok = False

    print("\n8. ChromaDB")
    try:
        import chromadb
        print(f"   Version: {chromadb.__version__}")
    except ImportError:
        print("   ERROR: Not installed")
        ok = False

    print("\n9. Playwright")
    try:
        import playwright
        print(f"   Version: {playwright.__version__}")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            print(f"   Chromium: {'OK' if p.chromium else 'MISSING'}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 50)
    if ok:
        print("All critical checks PASSED.")
        if torch.cuda.is_available():
            print("GPU is ready — ASSBRAIN will use CUDA automatically.")
        else:
            print("Running on CPU. Install NVIDIA drivers for GPU acceleration.")
    else:
        print("Some checks FAILED. Review errors above.")
    print("=" * 50)
    return ok


if __name__ == "__main__":
    success = check()
    sys.exit(0 if success else 1)
