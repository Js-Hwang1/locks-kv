from pathlib import Path
import hashlib
import os
import torch
from .. import arch

_MOD = None


def _get():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_suffix(".cu")
        tag = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _MOD = load(
            name="locks_topk_scalable_" + tag,
            sources=[str(source)],
            extra_cuda_cflags=["-O3", arch.arch_flag()],
            verbose=False,
        )
    return _MOD


def select(st, n_req, tile=None):
    if not n_req:
        return
    preferred = 2048 if st.max_pages <= 2048 else 1024
    if 8192 < st.max_pages <= 32768:
        preferred = 2048
    tile = int(tile or os.environ.get("LOCKS_TOPK_TILE", str(preferred)))
    cap = st.budget_abs or max(1, int(st.max_pages * st.budget + 1))
    while tile < 2 * cap and tile < 4096:
        tile *= 2
    if tile < 2 * cap:
        _get().streaming(
            st.score,
            st.n_pages,
            st.n_sel_hi,
            st.b_fix,
            st.page_table,
            st.page_cnt,
            n_req,
            st.sink_pages,
        )
        st._topb_done = False
        return
    signature = (tile, cap, st.max_pages, st.score.shape[:2], st.score.device)
    if getattr(st, "_scalable_topk_signature", None) != signature:
        (stages, n) = ([], st.max_pages)
        dummy = torch.empty(0, device=st.score.device, dtype=torch.int64)
        previous = dummy
        first = True
        while True:
            last = n <= tile
            size = (n + tile - 1) // tile * cap
            output = (
                dummy
                if last
                else torch.empty(
                    *st.score.shape[:2], size, device=st.score.device, dtype=torch.int64
                )
            )
            stages.append((previous, output, n, first, last))
            if last:
                break
            (first, previous, n) = (False, output, size)
        st._scalable_topk_signature = signature
        st._scalable_topk_stages = stages
    mod = _get()
    for previous, output, n, first, last in st._scalable_topk_stages:
        mod.reduce(
            st.score,
            previous,
            output,
            st.n_pages,
            st.n_sel_hi,
            st.b_fix,
            st.page_table,
            st.page_cnt,
            n_req,
            st.sink_pages,
            cap,
            n,
            tile,
            first,
            last,
        )
    st._topb_done = False
