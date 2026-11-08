from __future__ import annotations
import triton
import triton.language as tl
from .topk_scalable import select as topb_select

__all__ = ['derive_page_params', 'topb_select']


@triton.jit
def _derive_params_kernel(
    sl_ptr,
    budget_ptr,
    npg_ptr,
    nsh_ptr,
    nscore_ptr,
    b_ptr,
    n_reqs,
    PAGE: tl.constexpr,
    WIN: tl.constexpr,
    SINK: tl.constexpr,
    BUDGET_ABS: tl.constexpr,
    R_PAD: tl.constexpr,
):
    offs = tl.arange(0, R_PAD)
    mask = offs < n_reqs
    sl = tl.load(sl_ptr + offs, mask=mask, other=0)
    npg = (sl + (PAGE - 1)) // PAGE
    nsh = tl.maximum(npg - WIN, 0)
    s = nsh - SINK
    if BUDGET_ABS > 0:
        b = tl.minimum(BUDGET_ABS, tl.maximum(s, 1))
        b = tl.maximum(b, 1)
    else:
        budget = tl.load(budget_ptr)
        bf = tl.math.ceil(s.to(tl.float64) * budget)
        b = tl.minimum(tl.maximum(bf.to(tl.int32), 1), tl.maximum(s, 1))
    tl.store(nscore_ptr + offs, tl.maximum(npg - 1, 0), mask=mask)
    tl.store(npg_ptr + offs, npg, mask=mask)
    tl.store(nsh_ptr + offs, nsh, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)


def derive_page_params(st, seq_lens, n_req: int) -> None:
    assert n_req <= st.max_reqs
    _derive_params_kernel[1,](
        seq_lens,
        st.budget64,
        st.n_pages,
        st.n_sel_hi,
        st.n_score_hi,
        st.b_fix,
        n_req,
        PAGE=st.page,
        WIN=st.window_pages,
        SINK=st.sink_pages,
        BUDGET_ABS=int(getattr(st, "budget_abs", 0)),
        R_PAD=triton.next_power_of_2(max(st.max_reqs, 2)),
        num_warps=1,
    )
