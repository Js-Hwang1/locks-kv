"""CUDA architecture dispatch for the LOCKS hand-CUDA kernels.

The kernels were written and tuned for Hopper (sm_90a).  An instruction audit
(scratch_sm120/) established that NONE of them use a Hopper-exclusive
instruction: there is no ``wgmma``, no ``cp.async.bulk``/TMA and no cluster
primitive anywhere in the CUDA sources.  The only architecture-sensitive PTX is

  * ``mma.sync.aligned.m16n8k16`` (bf16/f16) and ``.m16n8k32`` (s8)  -- both are
    sm_80+ instructions, valid on every target we care about;
  * ``cp.async.ca.shared.global``                                    -- sm_80+;
  * ``griddepcontrol.wait`` / ``.launch_dependents`` (PDL)           -- sm_90+.

All four were verified to ASSEMBLE and EXECUTE on sm_120a (RTX PRO 6000
Blackwell), including a genuinely armed PDL launch through ``cudaLaunchKernelEx``
with ``cudaLaunchAttributeProgrammaticStreamSerialization``.  So the port is an
arch-flag dispatch, not an instruction rewrite.

What does NOT transfer is the shared-memory budget.  Hopper offers 227 KB of
opt-in dynamic shared memory per block; consumer/workstation Blackwell (sm_120)
offers 99 KB.  Kernels whose tile was sized against Hopper's budget must shrink
their tile on sm_120 -- see ``smem_cap()`` and the ``FWAVES``/``WARPS_F``
fallback in ``attention/decode_cuda.py``.

Environment overrides
---------------------
``LOCKS_CUDA_ARCH``   ``auto`` (default) | ``90a`` | ``120a`` | ``80`` | ...
                      Forces the -gencode target regardless of the live device.
``LOCKS_PDL``         ``0`` disables programmatic dependent launch (already
                      honoured by the kernels themselves).
"""
from __future__ import annotations

import functools
import os

import torch

# The exact string the kernels shipped with.  Returned unchanged on Hopper so
# the sm_90a build is byte-identical to the pre-port tree.
_SM90A = "-gencode=arch=compute_90a,code=sm_90a"
_SM120A = "-gencode=arch=compute_120a,code=sm_120a"

# Architectures with an "a" (architecture-accelerated) variant that we target.
_ACCEL = {(9, 0): _SM90A, (12, 0): _SM120A}


@functools.lru_cache(maxsize=8)
def capability(device: int | None = None) -> tuple[int, int]:
    """(major, minor) of the current CUDA device; (9, 0) when unavailable."""
    try:
        if not torch.cuda.is_available():
            return (9, 0)
        return torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 -- no driver / no device
        return (9, 0)


@functools.lru_cache(maxsize=8)
def arch_flag(device: int | None = None) -> str:
    """The single ``-gencode`` flag for the live device (or LOCKS_CUDA_ARCH)."""
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
    """Max opt-in dynamic shared memory per block, in bytes.

    Hopper H200: 232448 (227 KB).  sm_120 Blackwell: 101376 (99 KB).
    Falls back to the Hopper value when no device is visible so that a
    host-only import keeps the shipped tile choices.
    """
    try:
        if not torch.cuda.is_available():
            return 232448
        props = torch.cuda.get_device_properties(device)
        v = getattr(props, "shared_memory_per_block_optin", None)
        if v:
            return int(v)
        # Older torch: fall back to the per-SM figure minus the 1 KB reserve.
        return int(props.shared_memory_per_multiprocessor) - 1024
    except Exception:  # noqa: BLE001
        return 232448


@functools.lru_cache(maxsize=8)
def has_pdl(device: int | None = None) -> bool:
    """Programmatic dependent launch (``griddepcontrol``) availability.

    Verified to assemble AND execute (armed launch) on sm_90a and sm_120a.
    """
    if os.environ.get("LOCKS_PDL", "1") == "0":
        return False
    return capability(device) >= (9, 0)


def describe(device: int | None = None) -> str:
    cc = capability(device)
    return (f"sm_{cc[0]}{cc[1]} flag={arch_flag(device)} "
            f"smem_optin={smem_cap(device)}B pdl={has_pdl(device)}")
