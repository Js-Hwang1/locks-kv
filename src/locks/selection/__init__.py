"""Stage A -- decode-time page selection (SHARED by LOCKS-fast and LOCKS-mem).

ONE score, ONE path: **r8i4** (rank-8 int-4 basis).  A query-INDEPENDENT
low-rank page summary (mu / V / C from the page-gram eigh, built on
page-finalize with content-tag invalidation) scores every page DIRECTLY, and a
static top-b keeps the highest-scoring pages per kv-head (kv-union = group max),
always attending the sink pages + recent window + partial tail.  No second exact
LSE pass, no adaptive per-head coverage, no score variants.

Pipeline (per decode step, per layer):

  r8i4_build_refresh(st, layer, K, block_table, seq_lens, n_req)  # page-finalize
  derive_page_params(st, seq_lens, n_req)                         # once/step
  r8i4_score_only(...) -> st.score_h -> fused nrm+topb -> st.page_table/page_cnt

All persistent state lives in :class:`R8i4State` (allocated once from engine
maxima).  Scoring and selection are fixed-shape, host-sync-free and
allocation-free (full-CUDA-graph safe); the build runs off the hot path.

The quad / clse / r8 score variants were DELETED 2026-07-22 (user directive:
one maintained mainline, no dead code behind flags).
"""
from __future__ import annotations

from .r8i4_build import (r8i4_build_bulk, r8i4_build_delta,
                         r8i4_build_refresh, r8i4_build_tail)
from .r8i4_state import R8i4State, TAGW
from .select import derive_page_params, topb_select
# MLA (DeepSeek V2/V3/V3.2) selection surface -- the latent-cache analog of the
# GQA path. Scorer + top-b keep + the single mla_select_pages entry the LOCKS
# MLA backend forwards to.
from .mla_score_torch import mla_page_score_ref, topb_middle
from .mla_select import build_page_summary, mla_page_score, mla_select_pages

__all__ = [
    "R8i4State",
    "TAGW",
    "r8i4_build_refresh",
    "r8i4_build_tail",
    "r8i4_build_bulk",
    "r8i4_build_delta",
    "derive_page_params",
    "topb_select",
    # MLA
    "mla_page_score_ref",
    "topb_middle",
    "build_page_summary",
    "mla_page_score",
    "mla_select_pages",
]
