import os
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

POLICY = "all_visible_page_share_average"
UNSUPPORTED_FLAGS = (
    "LOCKS_FORCE_BRIDGE",
    "LOCKS_BRIDGE_FAMILY",
    "LOCKS_NOROPE",
    "LOCKS_QFIRST",
    "LOCKS_QFIRST2",
    "LOCKS_QFIRST_LITE",
    "LOCKS_QFIRST_CO",
    "LOCKS_QFIRST_FIXC",
    "LOCKS_QFIRST_GV",
    "LOCKS_QKVGEMV",
    "LOCKS_FLAT",
    "LOCKS_PMAX_FOLD",
    "LOCKS_PMAX_FOLDR",
    "RKI4_MMA_FLAT",
    "LOCKS_MEM_K8D",
    "LOCKS_HMAX",
    "LOCKS_SEL_V2",
    "LOCKS_PROBE_SKIP_MU",
    "LOCKS_PROBE_SKIP_C8",
    "LOCKS_PROBE_SKIP_V4",
    "RKI4_MMA_S1PROBE",
    "RKI4_CP_NOEXP",
    "RKI4_CP_NODT",
    "LOCKS_SCREEN",
)


def validate_policy_environment():
    if os.environ.get("RKI4_MMA_PFR"):
        assert os.environ.get("RKI4_MMA_AOS") == "1"
        assert os.environ.get("LOCKS_COMPACT_AOS_TAGS") == "1"
        assert os.environ.get("LOCKS_RKI4_RANK") == "4"
    presence_flags = (
        "RKI4_MMA_",
        "RKI4_CP_",
        "LOCKS_PROBE_",
        "LOCKS_SCREEN",
        "LOCKS_PMAX_FOLDR",
        "LOCKS_BRIDGE_FAMILY",
    )
    enabled = [
        key
        for key in UNSUPPORTED_FLAGS
        if os.environ.get(key, "")
        not in (
            ("",) if key.startswith(presence_flags) else ("", "0", "false", "False")
        )
    ]
    if enabled:
        raise ValueError(
            "These legacy paths are not validated for all-visible-page share-average: "
            + ", ".join(enabled)
        )


@triton.jit
def _current_page_kernel(
    Q,
    K,
    BT,
    SL,
    VBO,
    SH,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    BTS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    STAGED: tl.constexpr,
    SCALE: tl.constexpr,
):
    request = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_head = tl.program_id(2)
    length = tl.load(SL + request)
    if length > 0:
        page = (length - 1) // PAGE
        block = tl.load(BT + request * BTS + page)
        if STAGED:
            block = tl.load(VBO + block)
            tl.device_assert(
                block >= 0, "Current page must be present in the staging cache"
            )
        d = tl.arange(0, D)
        t = tl.arange(0, PAGE)
        query = tl.load(Q + request * QS0 + (kv_head * G + group_head) * QS1 + d).to(
            tl.float32
        )
        query_scale = tl.maximum(tl.max(tl.abs(query), 0), 1e-08) / 127.0
        query = libdevice.nearbyint(query / query_scale) * query_scale
        valid = (page * PAGE + t < length) & (block >= 0)
        key = tl.load(
            K + block * KS0 + t[:, None] * KS1 + kv_head * KS2 + d[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)
        logits = tl.sum(key * query[None, :], 1) * SCALE
        logits = tl.where(valid, logits, float("-inf"))
        peak = tl.max(logits, 0)
        score = peak + tl.log(tl.sum(tl.exp(logits - peak), 0))
        tl.store(SH + request * S0 + kv_head * S1 + group_head * S2 + page, score)


@triton.jit
def _normalizer_chunks(
    SH,
    NP,
    CHUNKS,
    MP: tl.constexpr,
    G: tl.constexpr,
    KV: tl.constexpr,
    Z: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    chunk = tl.program_id(1)
    request = head // (KV * G)
    pages = chunk * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(NP + request)
    valid = (pages < n) & (pages < MP)
    values = tl.load(SH + head * MP + pages, mask=valid, other=float("-inf"))
    peak = tl.max(values, 0)
    safe_peak = tl.where(n > chunk * BLOCK, peak, 0.0)
    total = tl.sum(tl.exp(values - safe_peak), 0)
    tl.store(CHUNKS + head * Z + chunk, safe_peak + tl.log(total))


@triton.jit
def _normalizer_finish(CHUNKS, LOGZ, Z: tl.constexpr, PAD: tl.constexpr):
    head = tl.program_id(0)
    index = tl.arange(0, PAD)
    values = tl.load(CHUNKS + head * Z + index, mask=index < Z, other=float("-inf"))
    peak = tl.max(values, 0)
    safe_peak = tl.where(peak == float("-inf"), 0.0, peak)
    total = tl.sum(tl.exp(values - safe_peak), 0)
    tl.store(LOGZ + head, tl.where(total > 0, safe_peak + tl.log(total), 0.0))


@triton.jit
def _mean_shares(
    SH,
    NP,
    LOGZ,
    OUT,
    MP: tl.constexpr,
    KV: tl.constexpr,
    G: tl.constexpr,
    GPAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    pages = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    heads = tl.arange(0, GPAD)
    n = tl.load(NP + unit // KV)
    valid = (heads[:, None] < G) & (pages[None, :] < n) & (pages[None, :] < MP)
    score = tl.load(
        SH + (unit * G + heads[:, None]) * MP + pages[None, :],
        mask=valid,
        other=float("-inf"),
    )
    normalizer = tl.load(LOGZ + unit * G + heads, mask=heads < G, other=0)
    mean = tl.sum(tl.exp(score - normalizer[:, None]), 0) / G
    tl.store(OUT + unit * MP + pages, mean, mask=pages < MP)


def combine(st, n_req):
    if st.combine != "shared_average":
        raise ValueError("LOCKS requires shared_average")
    if n_req == 0:
        return
    if not st.score_h.is_contiguous():
        raise ValueError("Page logits must be contiguous")
    heads = n_req * st.n_kv * st.G
    chunks = triton.cdiv(st.max_pages, 512)
    _normalizer_chunks[heads, chunks](
        st.score_h,
        st.n_pages,
        st.share_chunks,
        MP=st.max_pages,
        G=st.G,
        KV=st.n_kv,
        Z=chunks,
        BLOCK=512,
    )
    _normalizer_finish[heads,](
        st.share_chunks, st.share_logz, Z=chunks, PAD=triton.next_power_of_2(chunks)
    )
    _mean_shares[n_req * st.n_kv, triton.cdiv(st.max_pages, 256)](
        st.score_h,
        st.n_pages,
        st.share_logz,
        st.score,
        MP=st.max_pages,
        KV=st.n_kv,
        G=st.G,
        GPAD=triton.next_power_of_2(st.G),
        BLOCK=256,
    )
    st._topb_done = False


def complete_current_and_combine(
    st, n_req, *, q, kv_cache, block_table, seq_lens, scale, layer
):
    from ..attention.decode import _split_kv
    from ..backend import _runtime

    validate_policy_environment()
    tier = _runtime.get_tier()
    staged = tier is not None and tier.offload_k
    if staged:
        key = tier.staging.k_pool[layer]
        slots = tier.staging.vbo
    else:
        (key, _) = _split_kv(kv_cache)
        slots = seq_lens
    if key.stride(-1) != 1 or q.stride(-1) != 1:
        raise ValueError("Head components must be contiguous")
    if n_req:
        _current_page_kernel[n_req, st.n_kv, st.G](
            q,
            key,
            block_table,
            seq_lens,
            slots,
            st.score_h,
            QS0=q.stride(0),
            QS1=q.stride(1),
            KS0=key.stride(0),
            KS1=key.stride(1),
            KS2=key.stride(2),
            BTS=block_table.stride(0),
            S0=st.score_h.stride(0),
            S1=st.score_h.stride(1),
            S2=st.score_h.stride(2),
            G=st.G,
            D=st.d,
            PAGE=st.page,
            STAGED=staged,
            SCALE=float(scale),
            debug=True,
        )
    combine(st, n_req)
