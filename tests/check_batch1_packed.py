import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
assert not a.report.exists()
os.environ['LOCKS_CONFIG'] = json.dumps(dict(variant='fast', r8_rank=4))
os.environ['LOCKS_RKI4_RANK'] = '4'
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from locks.selection.rki4_state import Rki4State
from locks.selection.rki4_build import _rki4_write
from locks.selection.rki4_score_cuda import rki4_score_state
from locks.selection.select import derive_page_params
from locks.selection.select_cuda import topb_select_cuda

torch.set_num_threads(1)
torch.manual_seed(20261109)
rows = []
for max_pages in (1033, 2057, 4105, 8193, 16384, 32777):
    geometry = dict(device='cuda', num_layers=1, num_blocks=151, n_kv=4,
        G=8, head_dim=128, page=16, max_pages=max_pages, rank=4, budget_pages=126)
    reference = Rki4State(max_reqs=2, **geometry)
    candidate = Rki4State(max_reqs=1, resident_low_context=True, **geometry)
    assert reference.pp_rec is None
    packed = max_pages <= 8192 or max_pages >= 16384
    assert (candidate.pp_rec is not None) == packed
    keys = torch.randn(151, 16, 4, 128, device='cuda', dtype=torch.bfloat16)
    original = keys.clone()
    cache = torch.stack((keys, torch.randn_like(keys)))
    blocks = torch.arange(151, device='cuda')
    layer = torch.zeros_like(blocks)
    for st in (reference, candidate):
        _rki4_write(st, keys, layer, blocks)
    for name in ('pp_v4', 'pp_vs', 'pp_c8', 'pp_cs', 'pp_mus', 'pp_tag'):
        assert torch.equal(getattr(reference, name), getattr(candidate, name)), name
    mu = candidate.pp_mu8
    if packed:
        restored = torch.empty_like(mu)
        restored[..., 0::2] = mu[..., :64]
        restored[..., 1::2] = mu[..., 64:]
        mu = restored
    assert torch.equal(reference.pp_mu8, mu)
    q = torch.randn(1, 32, 128, device='cuda', dtype=torch.bfloat16)
    table = torch.zeros(1, max_pages, device='cuda', dtype=torch.int32)
    table[0, :151] = torch.randperm(151, device='cuda').int()
    for visible in (137 * 16 + 1, 138 * 16, 138 * 16 + 15):
        lengths = torch.tensor([visible], device='cuda', dtype=torch.int32)

        def run(st):
            derive_page_params(st, lengths, 1)
            rki4_score_state(st, 0, q, table, lengths, 1, 128**-.5, kv_cache=cache)
            topb_select_cuda(st, 1)

        run(reference)
        run(candidate)
        n = (visible + 15) // 16
        assert torch.equal(reference.score_h[0, ..., :n], candidate.score_h[0, ..., :n])
        assert torch.equal(reference.score[0], candidate.score[0])
        assert torch.equal(reference.page_table[0], candidate.page_table[0])
        assert torch.equal(reference.page_cnt[0], candidate.page_cnt[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(candidate)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(reference.page_table[0], candidate.page_table[0])
        assert torch.equal(keys, original)
        rows.append(dict(max_pages=max_pages, visible_tokens=visible, packed=packed, low_context=candidate.low_context,
            scores_bitwise_equal=True, shares_bitwise_equal=True,
            selected_ids_bitwise_equal=True, original_keys_unchanged=True))
a.report.write_text(json.dumps(dict(passed=True, seed=20261109, rows=rows), indent=2) + '\n')
print('Batch-1 packed summaries preserve scores, shared selection and graph replay')
