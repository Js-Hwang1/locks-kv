"""Architecture-generic discovery + patching of the fused-QKV attention path.

WHY THIS FILE EXISTS (2026-07-22, user ruling "ALL of the OPTIMIZATIONS should
be MODEL AGNOSTIC, it should WORK for ANY MODEL as a vLLM plugin")
------------------------------------------------------------------------------
The Q-separate-GEMM lanes (``LOCKS_QFIRST*``, ``LOCKS_QFIRST_CO``,
``LOCKS_QFIRST_FIXC``, ``LOCKS_QKVGEMV``) and the NOROPE cos/sin publish used
to be wired by CLASS-patching exactly one model class::

    from vllm.model_executor.models.chatglm import GLMAttention
    GLMAttention.forward = <lane forward>
    layers = model.transformer.encoder.layers        # GLM-only walk

On any non-GLM model that patch targets a class the model never instantiates,
so the lane was SILENTLY INERT: no banner, no error, no effect.  Every
efficiency number was GLM-4-9B-1M with the lanes ON; every accuracy number was
Llama-3.1-8B with the lanes structurally OFF.  The two lanes had never measured
the same configuration.

WHAT THE LANE ACTUALLY NEEDS (universal in vLLM decoder models)
--------------------------------------------------------------
A FUSED ``QKVParallelLinear`` whose output splits into q/k/v by
``(q_size, kv_size, kv_size)``, a ``rotary_emb``, an ``Attention`` module, an
output projection, and optionally per-head ``q_norm``/``k_norm`` (QK-norm
families).  Nothing in that list is GLM-specific; only the ATTRIBUTE NAMES and
the forward ARGUMENT ORDER differ between architectures (``qkv_proj`` /
``query_key_value``, ``o_proj`` / ``dense``, ``(positions, hidden_states)`` /
``(hidden_states, position_ids)``).  This module discovers all of that from the
loaded model and patches the discovered modules only.

PER-INSTANCE, NOT PER-CLASS
---------------------------
Discovered modules get a freshly minted SUBCLASS of their own class
(``mod.__class__ = type("Locks<Cls>", (Cls,), {"forward": lane_forward})``).
Two properties this buys that a global class patch does not:
  * arch-neutral by construction -- no import of any model module;
  * scoped -- another model loaded in the same process keeps the stock class.
A plain ``mod.forward = fn`` instance attribute was REJECTED: dynamo resolves
``type(mod).forward`` when it inlines an nn.Module call, so an instance-level
override can be silently skipped under ``VLLM_COMPILE`` -- exactly the failure
class this file exists to remove.  The subclass IS visible to dynamo.

Everything the patched forward needs is stamped on the INSTANCE as plain
``_locks_*`` attributes (module aliases register as duplicate children, which
``named_modules``/``named_parameters`` de-duplicate by identity), so the traced
body reads ``self.<const attr>`` exactly like the stock forward it replaces --
no per-call dict lookups, no user-defined-object indirection for dynamo.

NO SILENT FALLBACK (house rule)
-------------------------------
If a lane flag is set and the model does not expose the surface (no fused QKV,
quantized QKV, per-layer-heterogeneous rotary geometry, QK-norm on a lane whose
rope is absorbed into a kernel), this module RAISES and names the architecture.
It never degrades to a no-op.  Engagement is proved by a printed banner naming
the lane, the arch, and the number of patched modules.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from . import _runtime

# attribute-name candidates, most specific first.  Discovery does NOT depend on
# these alone: a candidate only counts if the WHOLE surface resolves.
_QKV_NAMES = ("qkv_proj", "query_key_value", "Wqkv", "c_attn", "qkv")
_OUT_NAMES = ("o_proj", "dense", "out_proj", "wo", "proj")
_POS_NAMES = ("positions", "position_ids", "pos")
_HID_NAMES = ("hidden_states", "hidden", "x")


@dataclass
class AttnSite:
    """One decoder layer's attention surface (all sizes are PER TP RANK)."""

    idx: int                 # decoder-layer index (position in the ModuleList)
    path: str                # dotted module name, for banners/errors
    mod: Any                 # the attention nn.Module itself
    qkv_name: str
    qkv: Any                 # fused QKVParallelLinear
    out_name: str
    out: Any                 # output projection (RowParallelLinear)
    rotary: Any              # rotary embedding module
    q_norm: Any              # per-head q RMSNorm or None
    k_norm: Any              # per-head k RMSNorm or None
    q_size: int
    kv_size: int
    head_dim: int
    hid_i: int               # positional index of hidden_states (after self)
    hid_name: str
    pos_i: int               # positional index of positions (after self)
    pos_name: str


# --------------------------------------------------------------------------- #
# discovery                                                                     #
# --------------------------------------------------------------------------- #
def _first_attr(mod, names):
    for n in names:
        v = getattr(mod, n, None)
        if v is not None:
            return n, v
    return None, None


def _arg_index(cls, names) -> tuple[int, str]:
    """(0-based index after ``self``, name) of the first parameter of
    ``cls.forward`` whose name is in ``names``."""
    params = list(inspect.signature(cls.forward).parameters)
    if params and params[0] == "self":
        params = params[1:]
    for i, p in enumerate(params):
        if p in names:
            return i, p
    raise RuntimeError(
        f"locks attnwire: {cls.__name__}.forward{tuple(params)} exposes no "
        f"parameter named one of {names}")


def _surface(attn_mod):
    """(qkv_name, qkv, out_name, out) if ``attn_mod`` looks like a fused-QKV
    attention block, else None."""
    qn, qkv = _first_attr(attn_mod, _QKV_NAMES)
    if qkv is None or not isinstance(qkv, torch.nn.Module):
        return None
    if getattr(attn_mod, "rotary_emb", None) is None:
        return None
    if getattr(attn_mod, "attn", None) is None:
        return None
    if not (hasattr(attn_mod, "q_size") and hasattr(attn_mod, "kv_size")):
        return None
    on, out = _first_attr(attn_mod, _OUT_NAMES)
    if out is None or out is qkv or not isinstance(out, torch.nn.Module):
        return None
    return qn, qkv, on, out


def _attn_child(block):
    """The attention sub-module of one decoder block (ANY child name:
    ``self_attn`` on Llama/Qwen, ``self_attention`` on GLM, ...)."""
    for _, child in block.named_children():
        if _surface(child) is not None:
            return child
    return None


def discover(model) -> list[AttnSite]:
    """Every decoder layer's attention surface, in LAYER ORDER.

    Finds the largest ``nn.ModuleList`` all of whose entries carry a fused-QKV
    attention child (the deleted lookahead engine's ``_find_decoder_layers``
    discovery, generalized off its hardcoded ``.self_attn``); the list position
    IS the layer index the selection state is keyed by.  Raises -- it never
    returns a short or empty list quietly."""
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        kids = [_attn_child(blk) for blk in mod]
        if any(k is None for k in kids):
            continue
        if best is None or len(mod) > len(best[1]):
            best = (name, mod, kids)
    if best is None:
        raise RuntimeError(
            "locks attnwire: no decoder-layer ModuleList with a fused-QKV "
            "attention child found. The Q-first lanes need a fused "
            "QKVParallelLinear (+ rotary_emb + attn + q_size/kv_size); this "
            "architecture does not expose one.")
    lname, layers, kids = best
    cls = type(kids[0])
    hid_i, hid_name = _arg_index(cls, _HID_NAMES)
    pos_i, pos_name = _arg_index(cls, _POS_NAMES)
    sites: list[AttnSite] = []
    for i, a in enumerate(kids):
        if type(a) is not cls:
            raise RuntimeError(
                f"locks attnwire: heterogeneous attention classes in {lname} "
                f"({cls.__name__} vs {type(a).__name__} at layer {i})")
        qn, qkv, on, out = _surface(a)
        rot = a.rotary_emb
        sites.append(AttnSite(
            idx=i, path=f"{lname}.{i}", mod=a, qkv_name=qn, qkv=qkv,
            out_name=on, out=out, rotary=rot,
            q_norm=getattr(a, "q_norm", None),
            k_norm=getattr(a, "k_norm", None),
            q_size=int(a.q_size), kv_size=int(a.kv_size),
            head_dim=int(getattr(rot, "head_size", 0)
                         or getattr(a, "head_dim", 0)),
            hid_i=hid_i, hid_name=hid_name, pos_i=pos_i, pos_name=pos_name))
    return sites


# --------------------------------------------------------------------------- #
# validation + rope-geometry publish                                            #
# --------------------------------------------------------------------------- #
def _check_unquantized(sites, lane: str) -> None:
    for s in sites:
        qm = getattr(s.qkv, "quant_method", None)
        if qm is not None and type(qm).__name__ != "UnquantizedLinearMethod":
            raise RuntimeError(
                f"locks {lane}: {s.path} QKV is {type(qm).__name__}; the "
                "Q-separate GEMM slices ROWS of the fused weight, which is "
                "wrong on a packed quant layout. Unquantized QKV only.")
        w = getattr(s.qkv, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
            raise RuntimeError(
                f"locks {lane}: {s.path} QKV weight is not a 2-D bf16 tensor "
                f"({None if w is None else (tuple(w.shape), w.dtype)}) -- the "
                "lane kernels are bf16-only.")
        if w.shape[0] != s.q_size + 2 * s.kv_size:
            raise RuntimeError(
                f"locks {lane}: {s.path} fused QKV rows {w.shape[0]} != "
                f"q_size+2*kv_size ({s.q_size}+2*{s.kv_size}) -- the row-slice "
                "split would be wrong.")


def publish_rope(sites, lane: str) -> dict:
    """Publish the arch's rotary geometry to ``_runtime`` (the kernels that
    absorb the rope read it there, and their build flags are derived from it)
    and return it for the banner.  Raises on per-layer heterogeneous geometry
    (e.g. Gemma-3 local/global rope), which one published cache cannot
    represent."""
    rot = sites[0].rotary
    csc = getattr(rot, "cos_sin_cache", None)
    if csc is None:
        raise RuntimeError(
            f"locks {lane}: {type(rot).__name__} exposes no cos_sin_cache; "
            "the rope-absorbing kernels cannot be wired for this arch.")
    rot_dim = int(getattr(rot, "rotary_dim", csc.shape[-1]))
    neox = bool(getattr(rot, "is_neox_style", False))
    head = int(getattr(rot, "head_size", sites[0].head_dim))
    if not (csc.dim() == 2 and csc.shape[1] == rot_dim
            and csc.dtype == torch.bfloat16 and csc.stride(1) == 1):
        raise RuntimeError(
            f"locks {lane}: unexpected cos_sin_cache {tuple(csc.shape)} "
            f"{csc.dtype} strides {tuple(csc.stride())} for rotary_dim "
            f"{rot_dim} -- the kernels index it as a row-contiguous "
            "(P, rotary_dim) bf16 table.")
    if rot_dim % 2 or head % 2 or rot_dim > head:
        raise RuntimeError(
            f"locks {lane}: unsupported rotary geometry rot_dim={rot_dim} "
            f"head_size={head}")
    for s in sites[1:]:
        r = s.rotary
        if (getattr(r, "cos_sin_cache", None) is None
                or r.cos_sin_cache.data_ptr() != csc.data_ptr()
                or int(getattr(r, "rotary_dim", -1)) != rot_dim
                or bool(getattr(r, "is_neox_style", False)) != neox):
            raise RuntimeError(
                f"locks {lane}: layer {s.idx} has a DIFFERENT rotary "
                "(per-layer local/global rope). One published cos/sin cache "
                "cannot serve this arch.")
    _runtime._NOROPE_CSC = csc
    _runtime._NOROPE_HEAD = head
    _runtime._NOROPE_ROT = rot_dim
    _runtime._NOROPE_NEOX = neox
    return {"csc": tuple(csc.shape), "head": head, "rot": rot_dim,
            "neox": neox}


# --------------------------------------------------------------------------- #
# lane forwards.  Architecture-neutral bodies: every arch-specific fact is a
# ``self._locks_*`` constant stamped by _patch().                               #
# --------------------------------------------------------------------------- #
def _unpack(self, args, kwargs):
    """(hidden_states, positions) from an arbitrary call form (vLLM decoder
    layers call attention positionally or by keyword depending on the arch)."""
    h = (kwargs[self._locks_hid_name] if self._locks_hid_name in kwargs
         else args[self._locks_hid_i])
    p = (kwargs[self._locks_pos_name] if self._locks_pos_name in kwargs
         else args[self._locks_pos_i])
    return h, p


def _qk_norm(self, q, k):
    """Per-head q_norm/k_norm (QK-norm families), replicating vLLM's own
    view-normalize-view sequence.  Dead code on archs with neither."""
    hd = self._locks_hd
    if self._locks_qn is not None:
        q = self._locks_qn(
            q.view(*q.shape[:-1], q.shape[-1] // hd, hd)).view(q.shape)
    if self._locks_kn is not None:
        k = self._locks_kn(
            k.view(*k.shape[:-1], k.shape[-1] // hd, hd)).view(k.shape)
    return q, k


def _fwd_qfirst(self, *args, **kwargs):
    """QFIRST milestone 1: Q projected BEFORE KV, as two F.linear calls over
    ROW SLICES of the same fused weight/bias.  Identical math per output
    element; the reduction ORDER differs by cuBLAS shape choice, which is why
    this lane is selection-identity-gated rather than bitwise-gated."""
    import torch.nn.functional as F
    h, pos = _unpack(self, args, kwargs)
    lin = self._locks_qkv
    W, B = lin.weight, lin.bias
    qs, ks = self._locks_qs, self._locks_kvs
    q = F.linear(h, W[:qs], B[:qs] if B is not None else None)
    kv = F.linear(h, W[qs:], B[qs:] if B is not None else None)
    k, v = kv.split([ks, ks], dim=-1)
    q, k = _qk_norm(self, q, k)
    q, k = self._locks_rot(pos, q, k)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


def _fwd_qfirst2(self, *args, **kwargs):
    """QF2: q first + roped + PRESCORED (side stream) before the KV projection
    starts.  Rope is called per tensor with a dummy partner (rope is
    elementwise per tensor -> bitwise-identical to the joint call; the dummies
    are scratch, never read)."""
    import torch.nn.functional as F
    h, pos = _unpack(self, args, kwargs)
    lin = self._locks_qkv
    W, B = lin.weight, lin.bias
    qs, ks = self._locks_qs, self._locks_kvs
    q = F.linear(h, W[:qs], B[:qs] if B is not None else None)
    dk = q.new_empty(q.shape[0], ks)
    q, _ = self._locks_rot(pos, q, dk)
    torch.ops.locks_qf.prescore(q, self._locks_lidx)
    kv = F.linear(h, W[qs:], B[qs:] if B is not None else None)
    k, v = kv.split([ks, ks], dim=-1)
    dq = k.new_empty(k.shape[0], qs)
    _, k = self._locks_rot(pos, dq, k)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


def _fwd_qfirst_lite(self, *args, **kwargs):
    """QF2-LITE: stock fused QKV + rope kept VERBATIM (no sliced numerics); the
    prescore forks the score onto the side stream between rope and the
    attention call.  Zero selection-numerics surface."""
    h, pos = _unpack(self, args, kwargs)
    qkv, _ = self._locks_qkv(h)
    q, k, v = qkv.split([self._locks_qs, self._locks_kvs, self._locks_kvs],
                        dim=-1)
    q, k = _qk_norm(self, q, k)
    q, k = self._locks_rot(pos, q, k)
    torch.ops.locks_qf.prescore(q, self._locks_lidx)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


def _fwd_qkvgemv(self, *args, **kwargs):
    """P1 numerics wire: the QKV projection through the opaque pure op
    ``locks_gv.qkv`` (deterministic sliced GEMV at nt==1, stock F.linear inside
    the op at nt>1).  Everything after the projection is VERBATIM stock."""
    h, pos = _unpack(self, args, kwargs)
    lin = self._locks_qkv
    qkv = torch.ops.locks_gv.qkv(h, lin.weight, lin.bias, self._locks_qs)
    q, k, v = qkv.split([self._locks_qs, self._locks_kvs, self._locks_kvs],
                        dim=-1)
    q, k = _qk_norm(self, q, k)
    q, k = self._locks_rot(pos, q, k)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


def _fwd_qfirst_gv(self, *args, **kwargs):
    """P2: the full Q-first forward on the deterministic GEMV pair; the score
    forks to the side stream the moment roped q exists, and the kv-GEMV +
    rope-k + attention preamble are its cover."""
    h, pos = _unpack(self, args, kwargs)
    lin = self._locks_qkv
    W, B = lin.weight, lin.bias
    qs, ks = self._locks_qs, self._locks_kvs
    q = torch.ops.locks_gv.q_first(h, W, B, qs)
    dk = q.new_empty(q.shape[0], ks)
    q, _ = self._locks_rot(pos, q, dk)
    torch.ops.locks_qf.prescore(q, self._locks_lidx)
    kv = torch.ops.locks_gv.kv_second(h, W, B, qs)
    k, v = kv.split([ks, ks], dim=-1)
    dq = k.new_empty(k.shape[0], qs)
    _, k = self._locks_rot(pos, dq, k)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


def _fwd_qfirst_co(self, *args, **kwargs):
    """P2b / FIX-C: the ``locks_co.qkv`` op replaces [QKV GEMM + rotary]
    wholesale (decode ropes q inside the score kernel and k in the KV-write
    kernel; prefill ropes in the op body); everything downstream is stock."""
    h, pos = _unpack(self, args, kwargs)
    lin = self._locks_qkv
    qkv = torch.ops.locks_co.qkv(h, lin.weight, lin.bias, pos,
                                 self._locks_qs, self._locks_lidx)
    q, k, v = qkv.split([self._locks_qs, self._locks_kvs, self._locks_kvs],
                        dim=-1)
    ctx = self.attn(q, k, v)
    o = self._locks_out(ctx)
    return o[0] if isinstance(o, tuple) else o


# lane -> (forward, rope_absorbed_into_kernels)
_LANES: dict[str, tuple[Callable, bool]] = {
    "QFIRST": (_fwd_qfirst, False),
    "QFIRST2": (_fwd_qfirst2, False),
    "QFIRST_LITE": (_fwd_qfirst_lite, False),
    "QKVGEMV": (_fwd_qkvgemv, False),
    "QFIRST_GV": (_fwd_qfirst_gv, False),
    "QFIRST_CO": (_fwd_qfirst_co, True),
}


# --------------------------------------------------------------------------- #
# patching                                                                      #
# --------------------------------------------------------------------------- #
def stamp(site: AttnSite) -> None:
    """Stamp the per-instance constants the lane forwards read.  Module
    aliases are duplicate registrations of modules the parent already owns;
    ``named_modules``/``named_parameters`` de-duplicate by identity, so nothing
    downstream sees them twice."""
    m = site.mod
    m._locks_lidx = site.idx
    m._locks_qkv = site.qkv
    m._locks_out = site.out
    m._locks_rot = site.rotary
    m._locks_qn = site.q_norm
    m._locks_kn = site.k_norm
    m._locks_qs = site.q_size
    m._locks_kvs = site.kv_size
    m._locks_hd = site.head_dim
    m._locks_hid_i = site.hid_i
    m._locks_hid_name = site.hid_name
    m._locks_pos_i = site.pos_i
    m._locks_pos_name = site.pos_name


def _patch(sites, lane: str, fwd: Callable) -> type:
    cls = type(sites[0].mod)
    patched = type(f"Locks{cls.__name__}", (cls,),
                   {"forward": fwd, "_locks_lane": lane})
    for s in sites:
        stamp(s)
        s.mod.__class__ = patched
    return cls


_CALLBACKS: list[Callable] = []
_HOOKED = False


def on_model_loaded(fn: Callable) -> None:
    """Register ``fn(model, vllm_config)`` to run right after load_model.

    ONE wrapper for every lane: the old code wrapped
    ``BaseModelLoader.load_model`` once per flag, so each wrap re-walked the
    model and the wrappers nested in flag order."""
    global _HOOKED
    _CALLBACKS.append(fn)
    if _HOOKED:
        return
    _HOOKED = True
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    orig = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        model = orig(self, vllm_config, model_config, prefix=prefix)
        for cb in _CALLBACKS:
            cb(model, vllm_config)
        return model

    BaseModelLoader.load_model = load_model


def wire(lane: str, extra_check: Callable | None = None) -> None:
    """Arm ``lane`` on whatever architecture the engine loads.

    Runs at load_model: discover -> validate -> publish the rope geometry (for
    the lanes whose kernels absorb the rope) -> patch -> banner.  Any failure
    RAISES with the architecture named; there is no inert path."""
    fwd, rope_absorbed = _LANES[lane]

    def _cb(model, vllm_config):
        arch = getattr(vllm_config.model_config.hf_config, "model_type", "?")
        sites = discover(model)
        _check_unquantized(sites, lane)
        geom = None
        if rope_absorbed:
            if sites[0].q_norm is not None or sites[0].k_norm is not None:
                raise RuntimeError(
                    f"locks {lane}: arch {arch!r} uses QK-norm (per-head "
                    "q_norm/k_norm). This lane absorbs the rope into the "
                    "score / KV-write kernels, which apply no head norm, so "
                    "it cannot serve this arch. Use LOCKS_QFIRST or "
                    "LOCKS_QFIRST_LITE (stock rope), or extend the kernels.")
            geom = publish_rope(sites, lane)
            if sites[0].head_dim != 128:
                raise RuntimeError(
                    f"locks {lane}: head_size {sites[0].head_dim} -- the "
                    "rope-absorbing kernels are compiled for head_size 128.")
        if extra_check is not None:
            extra_check(sites, arch)
        base = _patch(sites, lane, fwd)
        tail = ("" if geom is None else
                f" rope(dim={geom['rot']} neox={geom['neox']} "
                f"csc{geom['csc']})")
        s0 = sites[0]
        qk = (f"q_norm={type(s0.q_norm).__name__ if s0.q_norm is not None else None}"
              f" k_norm={type(s0.k_norm).__name__ if s0.k_norm is not None else None}")
        print(f"[locks] {lane} ACTIVE: arch={arch} cls={base.__name__} "
              f"modules={len(sites)} qkv={s0.qkv_name} out={s0.out_name} "
              f"call({s0.hid_name}@{s0.hid_i},{s0.pos_name}@{s0.pos_i}) "
              f"q_size={s0.q_size} kv_size={s0.kv_size} head_dim={s0.head_dim} "
              f"G={s0.q_size // max(1, s0.kv_size)} "
              f"bias={'yes' if s0.qkv.bias is not None else 'NO(zero-row)'} "
              f"{qk}{tail}", flush=True)

    on_model_loaded(_cb)


def wire_rope_only(lane: str) -> None:
    """Publish the rotary geometry without patching any forward (NOROPE v3:
    the compiled trace stays 100% stock and the rope nodes are pruned from the
    captured graph instead)."""

    def _cb(model, vllm_config):
        arch = getattr(vllm_config.model_config.hf_config, "model_type", "?")
        sites = discover(model)
        geom = publish_rope(sites, lane)
        for s in sites:
            stamp(s)
        print(f"[locks] {lane} ACTIVE: arch={arch} layers={len(sites)} "
              f"rope(dim={geom['rot']} neox={geom['neox']} "
              f"csc{geom['csc']}) absorbed", flush=True)

    on_model_loaded(_cb)
