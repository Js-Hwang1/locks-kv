import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _current_and_chunks(
    Q, K, BT, SL, SH, NP, CHUNKS,
    QS0: tl.constexpr, QS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    BTS: tl.constexpr, MP: tl.constexpr, KV: tl.constexpr,
    G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    Z: tl.constexpr, SCALE: tl.constexpr,
):
    head = tl.program_id(0)
    chunk = tl.program_id(1)
    request = head // (KV * G)
    group_query = head % (KV * G)
    kv_head = group_query // G
    length = tl.load(SL + request)
    current = (length - 1) // PAGE
    current_score = float('-inf')
    if (length > 0) & (current // 512 == chunk):
        block = tl.load(BT + request * BTS + current)
        d = tl.arange(0, D)
        t = tl.arange(0, PAGE)
        query = tl.load(Q + request * QS0 + group_query * QS1 + d).to(tl.float32)
        query_scale = tl.maximum(tl.max(tl.abs(query), 0), 1e-08) / 127.0
        query = libdevice.nearbyint(query / query_scale) * query_scale
        valid = (current * PAGE + t < length) & (block >= 0)
        key = tl.load(
            K + block * KS0 + t[:, None] * KS1 + kv_head * KS2 + d[None, :],
            valid[:, None], other=0,
        ).to(tl.float32)
        logits = tl.sum(key * query[None, :], 1) * SCALE
        logits = tl.where(valid, logits, float('-inf'))
        peak = tl.max(logits, 0)
        current_score = peak + tl.log(tl.sum(tl.exp(logits - peak), 0))
        tl.store(SH + head * MP + current, current_score)
    pages = chunk * 512 + tl.arange(0, 512)
    n = tl.load(NP + request)
    values = tl.load(
        SH + head * MP + pages, (pages < n) & (pages < MP), other=float('-inf')
    )
    values = tl.where((pages == current) & (length > 0), current_score, values)
    peak = tl.max(values, 0)
    safe_peak = tl.where(n > chunk * 512, peak, 0.0)
    total = tl.sum(tl.exp(values - safe_peak), 0)
    tl.store(CHUNKS + head * Z + chunk, safe_peak + tl.log(total))


@triton.jit
def _finish_and_mean(
    SH, NP, CHUNKS, LOGZ, OUT,
    MP: tl.constexpr, KV: tl.constexpr, G: tl.constexpr,
    GPAD: tl.constexpr, Z: tl.constexpr, ZPAD: tl.constexpr, BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    chunk = tl.program_id(1)
    heads = tl.arange(0, GPAD)
    z = tl.arange(0, ZPAD)
    values = tl.load(
        CHUNKS + (unit * G + heads[:, None]) * Z + z[None, :],
        (heads[:, None] < G) & (z[None, :] < Z), other=float('-inf'),
    )
    peak = tl.max(values, 1)
    safe_peak = tl.where(peak == float('-inf'), 0.0, peak)
    total = tl.sum(tl.exp(values - safe_peak[:, None]), 1)
    normalizer = tl.where(total > 0, safe_peak + tl.log(total), 0.0)
    if chunk == 0:
        tl.store(LOGZ + unit * G + heads, normalizer, heads < G)
    pages = chunk * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(NP + unit // KV)
    valid = (heads[:, None] < G) & (pages[None, :] < n) & (pages[None, :] < MP)
    score = tl.load(
        SH + (unit * G + heads[:, None]) * MP + pages[None, :],
        valid, other=float('-inf'),
    )
    mean = tl.sum(tl.exp(score - normalizer[:, None]), 0) / G
    tl.store(OUT + unit * MP + pages, mean, pages < MP)


def complete(st, n_req, *, q, kv_cache, block_table, seq_lens, scale, layer):
    from ..attention.decode import _split_kv

    if st.combine != 'shared_average':
        raise ValueError('LOCKS requires shared_average')
    if not st.score_h.is_contiguous():
        raise ValueError('Page logits must be contiguous')
    if not n_req:
        return
    (key, _) = _split_kv(kv_cache)
    if key.stride(-1) != 1 or q.stride(-1) != 1:
        raise ValueError('Head components must be contiguous')
    chunks = triton.cdiv(st.max_pages, 512)
    _current_and_chunks[n_req * st.n_kv * st.G, chunks](
        q, key, block_table, seq_lens, st.score_h, st.n_pages, st.share_chunks,
        QS0=q.stride(0), QS1=q.stride(1),
        KS0=key.stride(0), KS1=key.stride(1), KS2=key.stride(2),
        BTS=block_table.stride(0), MP=st.max_pages, KV=st.n_kv,
        G=st.G, D=st.d, PAGE=st.page, Z=chunks, SCALE=scale, num_warps=4,
    )
    _finish_and_mean[n_req * st.n_kv, triton.cdiv(st.max_pages, 256)](
        st.score_h, st.n_pages, st.share_chunks, st.share_logz, st.score,
        MP=st.max_pages, KV=st.n_kv, G=st.G, GPAD=triton.next_power_of_2(st.G),
        Z=chunks, ZPAD=triton.next_power_of_2(chunks), BLOCK=256,
    )
    st._topb_done = False
