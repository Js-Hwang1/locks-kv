from __future__ import annotations
import dataclasses
import json
import os
from typing import Literal, Optional

Variant = Literal["fast", "mem-v", "mem-kv"]
Table = Literal["kv-shared"]


@dataclasses.dataclass(frozen=True)
class LocksConfig:
    variant: Variant = "fast"
    coverage: float = 0.95
    budget: Optional[float] = None
    budget_pages: Optional[int] = None
    table: Table = "kv-shared"
    sink_pages: int = 1
    window_pages: int = 1
    r8_screen: bool = False
    r8_rank: int = 4
    quad_combine: Literal["shared_average"] = "shared_average"
    rki4_zsplit: Optional[int] = None
    hot_slots: int = 0
    hot_ratio: float = 6.0
    v_blocks: Optional[int] = None
    max_pool_gb: float = 256.0
    mem_k_only: bool = True
    mem_summary_cache: bool = False
    use_cuda: bool = True
    force_eager: bool = False

    @property
    def is_mem(self) -> bool:
        return self.variant in ("mem-v", "mem-kv")

    @property
    def offload_k(self) -> bool:
        return self.variant == "mem-kv"

    def __post_init__(self) -> None:
        if self.budget is None and self.budget_pages is None:
            raise ValueError(
                "An explicit dynamic-page budget or compression ratio is required"
            )
        if self.variant not in ("fast", "mem-v", "mem-kv"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.offload_k and (not self.r8_screen):
            object.__setattr__(self, "r8_screen", True)
        if not 0.0 < self.coverage <= 1.0:
            raise ValueError(f"coverage must be in (0,1], got {self.coverage}")
        if self.budget is not None and (not 0.0 < self.budget <= 1.0):
            raise ValueError(f"budget must be in (0,1], got {self.budget}")
        if self.rki4_zsplit is not None and (not 1 <= self.rki4_zsplit <= 65535):
            raise ValueError(
                f"rki4_zsplit must be None (auto) or in [1, 65535] (CUDA gridDim.z limit), got {self.rki4_zsplit}"
            )
        if self.quad_combine != "shared_average" or self.table != "kv-shared":
            raise ValueError(
                "LOCKS requires shared_average and kv-shared; legacy peak-normalized or union settings are rejected"
            )
        from .runtime_options import configure

        configure(dataclasses.asdict(self))
        from .selection.shared_average import validate_policy_environment

        validate_policy_environment()
        if not self.use_cuda:
            raise ValueError(
                "Only the optimized CUDA/Triton implementation is available"
            )
        if self.offload_k:
            object.__setattr__(self, "mem_summary_cache", True)
        if self.sink_pages < 0 or self.window_pages < 1:
            raise ValueError(
                "Require nonnegative sink count and at least one current/recent page"
            )
        if self.budget_pages is not None and self.budget_pages < 1:
            raise ValueError(
                "budget_pages is the positive dynamic-page allowance after mandatory pages"
            )
        if not 0.0 < float(self.hot_ratio) <= 64.0:
            raise ValueError(f"hot_ratio must be in (0, 64], got {self.hot_ratio}")
        if self.mem_summary_cache:
            if self.variant != "mem-kv":
                raise ValueError(
                    f"mem_summary_cache requires variant='mem-kv' (K must live in the tier); got {self.variant!r}"
                )
            if not self.mem_k_only:
                raise ValueError(
                    "mem_summary_cache requires mem_k_only=True (the engine page is re-keyed to hold summary records)"
                )

    @classmethod
    def load(cls, overrides: Optional[dict] = None) -> "LocksConfig":
        data: dict = {}
        spec = os.environ.get("LOCKS_CONFIG")
        if spec:
            if os.path.isfile(spec):
                text = open(spec).read()
                if spec.endswith((".yaml", ".yml")):
                    import yaml

                    data = yaml.safe_load(text) or {}
                else:
                    data = json.loads(text)
            else:
                data = json.loads(spec)
        if overrides:
            data.update({k: v for (k, v) in overrides.items() if v is not None})
        data.setdefault("r8_rank", int(os.environ.get("LOCKS_RKI4_RANK", "4")))
        _sc = os.environ.get("LOCKS_MEM_SUMMARY_CACHE")
        if _sc is not None and "mem_summary_cache" not in data:
            data["mem_summary_cache"] = _sc not in ("", "0", "false", "False")
        _z = os.environ.get("LOCKS_RKI4_Z")
        if _z is not None and "rki4_zsplit" not in data:
            data["rki4_zsplit"] = int(_z)
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown LocksConfig keys: {sorted(unknown)}")
        return cls(**data)
