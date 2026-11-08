from __future__ import annotations
import functools
import os
import torch

_ACCEL = {
    (9, 0): "-gencode=arch=compute_90a,code=sm_90a",
    (12, 0): "-gencode=arch=compute_120a,code=sm_120a",
}


@functools.lru_cache(maxsize=8)
def capability(device: int | None = None) -> tuple[int, int]:
    try:
        if not torch.cuda.is_available():
            return (9, 0)
        return torch.cuda.get_device_capability(device)
    except Exception:
        return (9, 0)


@functools.lru_cache(maxsize=8)
def arch_flag(device: int | None = None) -> str:
    ov = os.environ.get("LOCKS_CUDA_ARCH", "auto").strip().lower()
    if ov and ov != "auto":
        ov = ov.removeprefix("sm_")
        return f"-gencode=arch=compute_{ov},code=sm_{ov}"
    cc = capability(device)
    if cc in _ACCEL:
        return _ACCEL[cc]
    n = f"{cc[0]}{cc[1]}"
    return f"-gencode=arch=compute_{n},code=sm_{n}"


def is_hopper(device: int | None = None) -> bool:
    return capability(device) == (9, 0)


def is_blackwell_sm120(device: int | None = None) -> bool:
    return capability(device) == (12, 0)


@functools.lru_cache(maxsize=8)
def smem_cap(device: int | None = None) -> int:
    try:
        if not torch.cuda.is_available():
            return 232448
        props = torch.cuda.get_device_properties(device)
        v = getattr(props, "shared_memory_per_block_optin", None)
        if v:
            return int(v)
        return int(props.shared_memory_per_multiprocessor) - 1024
    except Exception:
        return 232448


@functools.lru_cache(maxsize=8)
def has_pdl(device: int | None = None) -> bool:
    return False


def describe(device: int | None = None) -> str:
    cc = capability(device)
    return f"sm_{cc[0]}{cc[1]} flag={arch_flag(device)} smem_optin={smem_cap(device)}B pdl={has_pdl(device)}"
