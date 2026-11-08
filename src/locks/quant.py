from __future__ import annotations
import dataclasses
import os
import re
import torch


@dataclasses.dataclass(frozen=True)
class KVQConfig:
    k_bits: int
    v_bits: int
    stage: str
    k_group: int
    v_group: int

    def describe(self) -> str:
        return f"KIVI fake-quant k{self.k_bits}v{self.v_bits} stage={self.stage} kgroup={self.k_group} vgroup={self.v_group}"


def _parse() -> "KVQConfig | None":
    spec = os.environ.get("LOCKS_KVQ", "off").strip().lower()
    if spec in ("", "off", "0", "none"):
        return None
    m = re.fullmatch("k([24])v([24])", spec)
    if m is None:
        raise ValueError(
            f"LOCKS_KVQ={spec!r}: expected 'off' or 'k{{2|4}}v{{2|4}}' (e.g. k2v2, k4v4)"
        )
    stage = os.environ.get("LOCKS_KVQ_STAGE", "post").strip().lower()
    if stage not in ("pre", "post", "selq"):
        raise ValueError(f"LOCKS_KVQ_STAGE={stage!r}: expected pre|post|selq")
    return KVQConfig(
        k_bits=int(m.group(1)),
        v_bits=int(m.group(2)),
        stage=stage,
        k_group=int(os.environ.get("LOCKS_KVQ_KGROUP", "32")),
        v_group=int(os.environ.get("LOCKS_KVQ_VGROUP", "32")),
    )


KVQ = _parse()
_ANNOUNCED = False
_DEBUG = os.environ.get("LOCKS_KVQ_DEBUG", "0") == "1"
_DEBUG_SAID = False


def _qdq(x32: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    levels = (1 << bits) - 1
    mn = x32.amin(dim=dim, keepdim=True)
    mx = x32.amax(dim=dim, keepdim=True)
    scale = ((mx - mn) / levels).clamp_min(1e-08)
    q = torch.round((x32 - mn) / scale).clamp_(0, levels)
    return q * scale + mn


def quantize_dequant_k(K: torch.Tensor, cfg: KVQConfig = None) -> torch.Tensor:
    cfg = cfg or KVQ
    (N, page, n_kv, d) = K.shape
    g = min(cfg.k_group, page)
    assert page % g == 0, f"page {page} not divisible by K group {g}"
    x = K.float().reshape(N, page // g, g, n_kv, d)
    y = _qdq(x, cfg.k_bits, dim=2)
    return y.reshape(N, page, n_kv, d).to(K.dtype)


def quantize_dequant_v(V: torch.Tensor, cfg: KVQConfig = None) -> torch.Tensor:
    cfg = cfg or KVQ
    (N, page, n_kv, d) = V.shape
    g = min(cfg.v_group, d)
    assert d % g == 0, f"d {d} not divisible by V group {g}"
    x = V.float().reshape(N, page, n_kv, d // g, g)
    y = _qdq(x, cfg.v_bits, dim=4)
    return y.reshape(N, page, n_kv, d).to(V.dtype)


def check_compat(cfg) -> None:
    if cfg is not None and getattr(cfg, "is_mem", False):
        raise RuntimeError(
            "LOCKS_KVQ: mem variants are unsupported (V/K pages are served from the DRAM tier, which this fake-quant hook does not touch). Run variant=fast."
        )
    raise RuntimeError(
        "LOCKS_KVQ requires LOCKS_TAIL_GRAPH=0: the tail graph replays _write_pre/_write_post inside a captured CUDA graph, bypassing the quant hook (arm contamination). Tail-graph off is bitwise-inert on the summaries ."
    )


def _done_slab(st) -> torch.Tensor:
    d = getattr(st, "_kvq_done", None)
    if d is None:
        d = st._kvq_done = torch.zeros(
            st.L, st.NB, dtype=torch.bool, device=st.pp_tag.device
        )
    return d


def _writeback(st, Kq: torch.Tensor, lidx: torch.Tensor, bidx: torch.Tensor) -> None:
    layers = getattr(st, "_kvq_layers", None)
    if layers is None:
        raise RuntimeError(
            "LOCKS_KVQ: engine K/V layer views not stashed on the state -- this build path (MLA? bridge?) is outside the KVQ study's coverage; refusing a silent partial arm."
        )
    global _DEBUG_SAID
    for l in torch.unique(lidx).tolist():
        m = lidx == l
        b = bidx[m]
        (K_half, V_half) = layers[int(l)]
        if _DEBUG and (not _DEBUG_SAID):
            _DEBUG_SAID = True
            k0 = K_half[b].float()
            v0 = V_half[b].float()
            dk = (k0 - Kq[m].float()).abs()
            K_half[b] = Kq[m].to(K_half.dtype)
            Vq = quantize_dequant_v(V_half[b]).to(V_half.dtype)
            dv = (v0 - Vq.float()).abs()
            V_half[b] = Vq
            rb_k = torch.equal(K_half[b], Kq[m].to(K_half.dtype))
            rb_v = torch.equal(V_half[b], Vq)
            print(
                f"[locks] KVQ WRITEBACK EVIDENCE layer={int(l)} pages={int(b.numel())}: |dK| mean={dk.mean():.4e} max={dk.max():.4e}; |dV| mean={dv.mean():.4e} max={dv.max():.4e}; readback K=={rb_k} V=={rb_v}",
                flush=True,
            )
            continue
        K_half[b] = Kq[m].to(K_half.dtype)
        V_half[b] = quantize_dequant_v(V_half[b]).to(V_half.dtype)


def build_hook(st, Kp_flat: torch.Tensor, lidx: torch.Tensor, bidx: torch.Tensor):
    global _ANNOUNCED
    cfg = KVQ
    (page, tagw) = (st.page, st.tagw)
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"[locks] KVQ hook LIVE: {cfg.describe()}", flush=True)
    if cfg.stage == "selq":
        Kq = quantize_dequant_k(Kp_flat, cfg)
        return (Kq, Kp_flat[:, page - 1, :, :tagw], lidx, bidx)
    done = _done_slab(st)
    tag_cur = Kp_flat[:, page - 1, :, :tagw].to(st.pp_tag.dtype)
    tag_old = st.pp_tag[lidx, bidx]
    unchanged = (tag_cur == tag_old).all(dim=(1, 2))
    keep = ~(done[lidx, bidx] & unchanged)
    if not bool(keep.any()):
        return None
    if not bool(keep.all()):
        (Kp_flat, lidx, bidx) = (Kp_flat[keep], lidx[keep], bidx[keep])
    Kq = quantize_dequant_k(Kp_flat, cfg)
    _writeback(st, Kq, lidx, bidx)
    done[lidx, bidx] = True
    if cfg.stage == "pre":
        return (Kq, None, lidx, bidx)
    return (Kp_flat, Kq[:, page - 1, :, :tagw], lidx, bidx)
