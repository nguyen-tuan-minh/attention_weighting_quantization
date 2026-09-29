"""Report CPU and accelerator devices visible to PyTorch."""

from __future__ import annotations

import os
import platform
import sys


def available_system_memory_gib() -> float | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def main() -> int:
    print("System")
    print(f"  OS: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor() or 'not reported'}")
    print(f"  Logical CPU cores: {os.cpu_count() or 'unknown'}")
    memory_gib = available_system_memory_gib()
    if memory_gib is not None:
        print(f"  Physical memory: {memory_gib:.1f} GiB")

    try:
        import torch
    except (ImportError, OSError) as error:
        print("\nPyTorch is unavailable; install project dependencies to inspect accelerators.")
        print(f"  Details: {error}")
        return 0

    print("\nPyTorch")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA runtime: {torch.version.cuda or 'not included'}")
    print(f"  ROCm runtime: {torch.version.hip or 'not included'}")

    if torch.cuda.is_available():
        print(f"  CUDA-compatible devices: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024**3)
            print(f"    [{index}] {properties.name} — {memory_gib:.1f} GiB")
    else:
        print("  CUDA-compatible device: unavailable")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        print(f"  Apple MPS built: {mps.is_built()}")
        print(f"  Apple MPS available: {mps.is_available()}")
    else:
        print("  Apple MPS: not supported by this PyTorch build")

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "is_available"):
        print(f"  Intel XPU available: {xpu.is_available()}")
        if xpu.is_available():
            for index in range(xpu.device_count()):
                print(f"    [{index}] {xpu.get_device_name(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
