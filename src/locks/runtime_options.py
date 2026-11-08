from __future__ import annotations
import json
import os
import sys
from pathlib import Path


def _read_spec():
    spec = os.environ.get("LOCKS_CONFIG", "")
    if not spec:
        return {}
    if not spec.lstrip().startswith("{"):
        path = Path(spec)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(text) or {}
        return json.loads(text)
    return json.loads(spec)


def _resolve(data):
    rank = int(data.get("r8_rank", os.environ.get("LOCKS_RKI4_RANK", "4")))
    if rank not in (2, 4, 8):
        raise ValueError("The optimized selector supports ranks 2, 4 and 8")
    if int(os.environ.get("LOCKS_RKI4_IBITS", "4")) != 4:
        raise ValueError("The optimized selector requires INT4 basis coefficients")
    if "LOCKS_RKI4_RANK" in os.environ and int(os.environ["LOCKS_RKI4_RANK"]) != rank:
        raise ValueError("LOCKS_RKI4_RANK and the configuration rank disagree")
    packed = data.get("variant", "fast") == "mem-kv" and rank == 4
    return {"rank": rank, "packed": packed}


OPTIONS = _resolve(_read_spec())


def configure(data):
    resolved = _resolve(data)
    if "locks.selection.rki4_state" in sys.modules and resolved != OPTIONS:
        raise RuntimeError(
            "Configure rank and KV mode before importing selector state; use a new process for a different engine configuration"
        )
    OPTIONS.update(resolved)
