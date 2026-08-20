"""KIVI-style KV-cache FAKE quantization -- the LOCKS KV-quant extended study.

METHOD CHOICE (user ruling 2026-07-26: exactly ONE method).  KIVI (ICML'24,
Liu et al.): asymmetric uniform quantization, K per-CHANNEL (statistics over a
group of tokens for each channel -- K outliers live in channels), V per-TOKEN
(statistics over a group of channels for each token).  Chosen over TurboQuant
(2025, rotation-based) because: (a) KIVI is the community-standard 2-bit KV
scheme every KV paper compares against (already slotted as the quant axis in
ours_doc/BASELINES_VLLM.md #16); (b) the algorithm is a closed-form 20-line
torch transform -- reproducible here EXACTLY, with no calibration data, no
rotation matrices, no reference-implementation drift to gate against; (c) it
maps 1:1 onto the vLLM-0.24 paged layout (page = 16 tokens = the K channel
group), whereas TurboQuant's online rotations have no seam in the 0.24 paged
cache short of new decode kernels.  This is an ACCURACY study: fake-quant
(quantize -> dequantize, cache stays bf16) is the correct instrument; no
packed-int cache, no speed claim.

WHAT IT DOES.  Pages are fake-quantized exactly ONCE, when they FINALIZE --
the same lifecycle point where the rki4 summary is built (`_rki4_write` is the
single funnel for bulk / delta / tail / refresh / overlap builds).  The
partial tail page stays fp16 inside the always-attended window: this is the
page-aligned analog of KIVI's fp16 residual window (ours is <= page tokens =
16, HARSHER than KIVI's default 128 -- a conservative stress test).  Stages:

  post  (default) summary built from fp16 K; K/V pages in the ENGINE CACHE
        overwritten with quantize(dequantize(.)) -> decode attends quantized
        KV at extreme sparsity, selection signal is fp16-built.
  pre   summary built FROM the quantized K (double-quantization of the
        selection signal: the rki4 summary is itself int4/int8 inside), AND
        the cache is overwritten -> both selection and attention see
        quantized KV.
  selq  summary built from quantized K, cache left fp16 -> isolates the
        SELECTION sensitivity (select-on-quant, attend-fp).

The stored content tag always matches the bytes the cache holds after this
step (post overrides the tag with the quantized page-final token; selq keeps
the fp16 tag), so the builder's delta tag scan stays a no-op in the steady
state and never silently rebuilds an arm with the wrong K source.  A
per-(layer, block) `done` slab + tag-equality filter makes the in-place
overwrite idempotent (a re-issued tail/bulk over an unchanged, already-
quantized page is SKIPPED: re-quantizing dequantized bf16 values would drift
by the bf16 rounding of the min/max grid points, and in `post` it would
silently rebuild the summary from quantized bytes = arm contamination).
Block reuse by a new request is caught by the tag mismatch (new fp16 content
!= stored tag) exactly like the existing delta scan.

ENV (read ONCE at import; LOCKS_KVQ unset/off is BYTE-INERT -- the only added
work on any build is one `is None` check, no tensor is touched):

  LOCKS_KVQ        off (default) | k{2|4}v{2|4}   e.g. k2v2, k4v4, k2v4
  LOCKS_KVQ_STAGE  post (default) | pre | selq
  LOCKS_KVQ_KGROUP K token-group per channel (default 32, clamped to page=16:
                   KIVI's G=32 spans two pages; page-aligned grouping keeps
                   quantization local to the finalize lifecycle. Finer groups
                   = slightly gentler than KIVI-32; deviation documented.)
  LOCKS_KVQ_VGROUP V channel-group per token (default 32; d=128 -> 4 groups)

COMPATIBILITY (checked loudly at state alloc, `check_compat`): fast variant
only (mem tiers serve V from DRAM pools this hook does not touch);
LOCKS_TAIL_GRAPH=0 (the tail graph replays _write_pre/_write_post inside a
captured CUDA graph, bypassing the hook -- and the hook's data-dependent
filter is not capturable; tail-graph off is documented bitwise-inert on the
summaries); LOCKS_PREFILL_OVERLAP=0 (the side-stream build would overwrite
cache pages concurrently with main-stream prefill attention reads).  The MLA
path does not route through `_rki4_write` and is NOT covered (the GQA rki4
mainline is the study's scope); its builds never stash `_kvq_layers`, so any
accidental use raises in `_writeback` rather than silently no-opping.

Not graph-relevant: the hook runs only inside the page-finalize build, which
is already host-prologue / off-graph by design.
"""
from __future__ import annotations

import dataclasses
import os
import re

import torch


@dataclasses.dataclass(frozen=True)
class KVQConfig:
    k_bits: int
    v_bits: int
    stage: str          # "pre" | "post" | "selq"
    k_group: int        # tokens per K per-channel group (clamped to page)
    v_group: int        # channels per V per-token group

    def describe(self) -> str:
        return (f"KIVI fake-quant k{self.k_bits}v{self.v_bits} "
                f"stage={self.stage} kgroup={self.k_group} "
                f"vgroup={self.v_group}")


def _parse() -> "KVQConfig | None":
    spec = os.environ.get("LOCKS_KVQ", "off").strip().lower()
    if spec in ("", "off", "0", "none"):
        return None
    m = re.fullmatch(r"k([24])v([24])", spec)
    if m is None:
        raise ValueError(
            f"LOCKS_KVQ={spec!r}: expected 'off' or 'k{{2|4}}v{{2|4}}' "
            "(e.g. k2v2, k4v4)")
    stage = os.environ.get("LOCKS_KVQ_STAGE", "post").strip().lower()
    if stage not in ("pre", "post", "selq"):
        raise ValueError(
            f"LOCKS_KVQ_STAGE={stage!r}: expected pre|post|selq")
    return KVQConfig(
        k_bits=int(m.group(1)), v_bits=int(m.group(2)), stage=stage,
        k_group=int(os.environ.get("LOCKS_KVQ_KGROUP", "32")),
        v_group=int(os.environ.get("LOCKS_KVQ_VGROUP", "32")))


KVQ = _parse()          # None <=> disabled (the byte-inert default)

_ANNOUNCED = False
# LOCKS_KVQ_DEBUG=1: one-shot writeback evidence (mean/max |K - qdq(K)| on the
# first overwritten pages + a readback equality check against the cache view).
# Liveness instrument for the gates -- proves the engine cache bytes actually
# changed; off by default, and only reachable when KVQ is already on.
_DEBUG = os.environ.get("LOCKS_KVQ_DEBUG", "0") == "1"
_DEBUG_SAID = False


# --------------------------------------------------------------------------- #
# Core fake-quant transforms (pure torch, fp32 math, output dtype == input).   #
# --------------------------------------------------------------------------- #
def _qdq(x32: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric uniform quantize->dequantize of fp32 ``x32`` with min/max
    statistics over dimension ``dim`` (the KIVI group dimension)."""
    levels = (1 << bits) - 1
    mn = x32.amin(dim=dim, keepdim=True)
    mx = x32.amax(dim=dim, keepdim=True)
    scale = ((mx - mn) / levels).clamp_min(1e-8)
    q = torch.round((x32 - mn) / scale).clamp_(0, levels)
    return q * scale + mn


def quantize_dequant_k(K: torch.Tensor, cfg: KVQConfig = None) -> torch.Tensor:
    """KIVI K fake-quant: PER-CHANNEL, statistics over token groups.

    K: (N, page, n_kv, d).  Group = ``min(k_group, page)`` consecutive tokens
    per (channel, kv-head); each channel of each group gets its own
    (min, scale).  Returns the same shape/dtype."""
    cfg = cfg or KVQ
    N, page, n_kv, d = K.shape
    g = min(cfg.k_group, page)
    assert page % g == 0, f"page {page} not divisible by K group {g}"
    x = K.float().reshape(N, page // g, g, n_kv, d)
    y = _qdq(x, cfg.k_bits, dim=2)
    return y.reshape(N, page, n_kv, d).to(K.dtype)


def quantize_dequant_v(V: torch.Tensor, cfg: KVQConfig = None) -> torch.Tensor:
    """KIVI V fake-quant: PER-TOKEN, statistics over channel groups.

    V: (N, page, n_kv, d).  Group = ``v_group`` consecutive channels per
    (token, kv-head).  Returns the same shape/dtype."""
    cfg = cfg or KVQ
    N, page, n_kv, d = V.shape
    g = min(cfg.v_group, d)
    assert d % g == 0, f"d {d} not divisible by V group {g}"
    x = V.float().reshape(N, page, n_kv, d // g, g)
    y = _qdq(x, cfg.v_bits, dim=4)
    return y.reshape(N, page, n_kv, d).to(V.dtype)


# --------------------------------------------------------------------------- #
# Compatibility gate (called by the builder at state alloc when KVQ is on).    #
# --------------------------------------------------------------------------- #
def check_compat(cfg) -> None:
    if cfg is not None and getattr(cfg, "is_mem", False):
        raise RuntimeError(
            "LOCKS_KVQ: mem variants are unsupported (V/K pages are served "
            "from the DRAM tier, which this fake-quant hook does not touch). "
            "Run variant=fast.")
    if os.environ.get("LOCKS_TAIL_GRAPH", "0") == "1":
        raise RuntimeError(
            "LOCKS_KVQ requires LOCKS_TAIL_GRAPH=0: the tail graph replays "
            "_write_pre/_write_post inside a captured CUDA graph, bypassing "
            "the quant hook (arm contamination). Tail-graph off is bitwise-"
            "inert on the summaries (ours_doc/TTFT_16K_KERNEL.md).")
    if os.environ.get("LOCKS_PREFILL_OVERLAP", "0") == "1":
        raise RuntimeError(
            "LOCKS_KVQ requires LOCKS_PREFILL_OVERLAP=0: the side-stream "
            "build would overwrite cache pages while main-stream prefill "
            "attention reads them (unfenced RAW race).")


# --------------------------------------------------------------------------- #
# The build hook (called from rki4_build._rki4_write when KVQ is not None).    #
# --------------------------------------------------------------------------- #
def _done_slab(st) -> torch.Tensor:
    d = getattr(st, "_kvq_done", None)
    if d is None:
        d = st._kvq_done = torch.zeros(
            st.L, st.NB, dtype=torch.bool, device=st.pp_tag.device)
    return d


def _writeback(st, Kq: torch.Tensor, lidx: torch.Tensor,
               bidx: torch.Tensor) -> None:
    """Overwrite the engine cache pages (lidx, bidx) in place: K <- ``Kq``,
    V <- quantize_dequant_v(V).  Host-syncs on unique(lidx) -- acceptable:
    page-finalize only, KVQ accuracy arms only (never the shipped path)."""
    layers = getattr(st, "_kvq_layers", None)
    if layers is None:
        raise RuntimeError(
            "LOCKS_KVQ: engine K/V layer views not stashed on the state -- "
            "this build path (MLA? bridge?) is outside the KVQ study's "
            "coverage; refusing a silent partial arm.")
    global _DEBUG_SAID
    for l in torch.unique(lidx).tolist():
        m = lidx == l
        b = bidx[m]
        K_half, V_half = layers[int(l)]
        if _DEBUG and not _DEBUG_SAID:
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
            print(f"[locks] KVQ WRITEBACK EVIDENCE layer={int(l)} "
                  f"pages={int(b.numel())}: |dK| mean={dk.mean():.4e} "
                  f"max={dk.max():.4e}; |dV| mean={dv.mean():.4e} "
                  f"max={dv.max():.4e}; readback K=={rb_k} V=={rb_v}",
                  flush=True)
            continue
        K_half[b] = Kq[m].to(K_half.dtype)
        V_half[b] = quantize_dequant_v(V_half[b]).to(V_half.dtype)


def build_hook(st, Kp_flat: torch.Tensor, lidx: torch.Tensor,
               bidx: torch.Tensor):
    """The single seam in ``_rki4_write``.  Returns ``None`` when every page
    in the call is already processed and unchanged (skip the write entirely),
    else ``(K_for_summary, tag_override, lidx, bidx)`` where ``tag_override``
    (or the tag `_write_pre` extracts from ``K_for_summary`` when None) always
    equals the page-final-token bytes the CACHE holds after this step."""
    global _ANNOUNCED
    cfg = KVQ
    page, tagw = st.page, st.tagw
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"[locks] KVQ hook LIVE: {cfg.describe()}", flush=True)
    if cfg.stage == "selq":
        # select-on-quant, attend-fp: cache untouched, tag stays fp16.
        Kq = quantize_dequant_k(Kp_flat, cfg)
        return Kq, Kp_flat[:, page - 1, :, :tagw], lidx, bidx
    # pre | post: in-place cache overwrite -> idempotency filter first.
    done = _done_slab(st)
    tag_cur = Kp_flat[:, page - 1, :, :tagw].to(st.pp_tag.dtype)
    tag_old = st.pp_tag[lidx, bidx]
    unchanged = (tag_cur == tag_old).all(dim=(1, 2))   # NaN-init -> False
    keep = ~(done[lidx, bidx] & unchanged)
    if not bool(keep.any()):
        return None
    if not bool(keep.all()):
        Kp_flat, lidx, bidx = Kp_flat[keep], lidx[keep], bidx[keep]
    Kq = quantize_dequant_k(Kp_flat, cfg)
    _writeback(st, Kq, lidx, bidx)
    done[lidx, bidx] = True
    if cfg.stage == "pre":
        return Kq, None, lidx, bidx        # tag from Kq == cache bytes
    return Kp_flat, Kq[:, page - 1, :, :tagw], lidx, bidx   # post
