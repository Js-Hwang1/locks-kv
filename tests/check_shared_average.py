import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from locks.selection.shared_average import combine
from reference_selection import reference_scores, reference_selection
from locks.selection.topk_scalable import select

p = argparse.ArgumentParser()
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
torch.manual_seed(20261108)
rows = []
for group in (1, 2, 4, 8):
    count, heads, pages = 3, 2, 1033
    logits = torch.randn(count, heads, group, pages, device='cuda') * 3
    visible = torch.tensor([pages, 17, 0], dtype=torch.int32, device='cuda')
    logits[1, :, :, 17:] = float('nan')
    logits[2] = float('nan')
    state = SimpleNamespace(
        combine='shared_average', score_h=logits, score=torch.empty(count,heads,pages,device='cuda'),
        max_pages=pages, n_kv=heads, G=group, n_pages=visible,
        share_chunks=torch.empty(count,heads,group,(pages+511)//512,device='cuda'),
        share_logz=torch.empty(count,heads,group,device='cuda'),
        sink_pages=1, budget_abs=6, budget=0.0,
        n_sel_hi=(visible-1).clamp_min(0), b_fix=torch.tensor([6,6,0],device='cuda',dtype=torch.int32),
        page_table=torch.empty(count,heads,pages,device='cuda',dtype=torch.int32),
        page_cnt=torch.empty(count,heads,device='cuda',dtype=torch.int32),
    )
    combine(state,count)
    expected = reference_scores(logits, visible)
    error = float((state.score-expected).abs().max())
    assert error < 2e-6 and torch.isfinite(state.score).all()
    select(state,count)
    _, ids, counts = reference_selection(logits, visible, total_pages=8,sink_pages=1,window_pages=1)
    assert torch.equal(state.page_table,ids) and torch.equal(state.page_cnt,counts)
    before = state.score.clone()
    logits.add_(torch.randn(count,heads,group,1,device='cuda')*20)
    combine(state,count)
    shift_error = float((state.score-before).abs().max())
    assert shift_error < 4e-6
    rows.append(dict(query_heads_per_kv=group,max_absolute_share_error=error,head_shift_error=shift_error,passed=True))
a.report.write_text(json.dumps(dict(passed=True,seed=20261108,cases=rows),indent=2)+'\n')
print('All-visible-page shared-average and exact shared budget checks passed')
