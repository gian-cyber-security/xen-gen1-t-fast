#!/usr/bin/env python3
"""Report runtime facts; never installs drivers or claims a GPU is present."""
import os, platform, shutil, subprocess, sys
try:
    import psutil
except Exception: psutil = None
try:
    import torch
except Exception: torch = None
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("cpu:", platform.processor() or platform.machine())
print("ram_gb:", round(psutil.virtual_memory().total / 2**30, 2) if psutil else "unavailable (install psutil for RAM detail)")
print("pytorch:", getattr(torch, "__version__", "unavailable"))
print("cuda_available:", bool(torch and torch.cuda.is_available()))
print("cuda_version:", getattr(getattr(torch, "version", None), "cuda", None) if torch else "unavailable")
if torch and torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu_memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
else:
    print("gpu: unavailable")
print("nvidia_smi:", shutil.which("nvidia-smi") or "unavailable")
