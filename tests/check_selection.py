import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
import numpy as np
import torch
from locks.selection.topk_scalable import select

torch.set_num_threads(1)
torch.manual_seed(20261107)

def state(n, rows=1, heads=4, cap=126, sink=1):
    score = torch.randn(rows,heads,n,device='cuda')
    return SimpleNamespace(score=score, max_pages=n, n_kv=heads, sink_pages=sink,
        budget_abs=cap, budget=.1,
        n_pages=torch.full((rows,),n,dtype=torch.int32,device='cuda'),
        n_sel_hi=torch.full((rows,),n-1,dtype=torch.int32,device='cuda'),
        b_fix=torch.full((rows,),cap,dtype=torch.int32,device='cuda'),
        page_table=torch.full((rows,heads,n),-7,dtype=torch.int32,device='cuda'),
        page_cnt=torch.full((rows,heads),-7,dtype=torch.int32,device='cuda'))

def reference(s):
    score=s.score.cpu().numpy()
    npg=s.n_pages.cpu().tolist(); hi=s.n_sel_hi.cpu().tolist(); b=s.b_fix.cpu().tolist()
    table=np.full(score.shape,-1,dtype=np.int32)
    cnt=np.zeros(score.shape[:2],dtype=np.int32)
    for r in range(score.shape[0]):
        for h in range(score.shape[1]):
            candidates=list(range(min(npg[r],s.sink_pages),min(npg[r],hi[r])))
            candidates.sort(key=lambda i:(-float(score[r,h,i]),i))
            selected=set(candidates[:b[r]])
            selected.update(range(min(npg[r],s.sink_pages)))
            selected.update(range(max(s.sink_pages,hi[r]),npg[r]))
            selected=sorted(selected)
            cnt[r,h]=len(selected)
            table[r,h,:len(selected)]=selected
    return torch.from_numpy(table).cuda(),torch.from_numpy(cnt).cuda()

start=time.time(); checks=[]
for n in (1,2,3,127,128,129,1024,1033,2048,2057,4096,4105,8192,8201,16384,16393,32768,32777,65545,131081):
    for pattern in ('random','all_equal','ties','ascending','descending','signed_zero','infinity'):
        s=state(n,rows=3,heads=4,cap=126,sink=1)
        with torch.no_grad():
            s.n_pages.copy_(torch.tensor([n,max(0,n-19),min(n,3)],device='cuda',dtype=torch.int32))
            s.n_sel_hi.copy_(torch.maximum(s.n_pages-1,torch.zeros_like(s.n_pages)))
            s.b_fix.copy_(torch.tensor([126,13,0],device='cuda',dtype=torch.int32))
            if pattern=='all_equal':s.score.fill_(.25)
            elif pattern=='ties':s.score.copy_(torch.randint(-3,4,s.score.shape,device='cuda').float())
            elif pattern=='ascending':s.score.copy_(torch.arange(n,device='cuda').float())
            elif pattern=='descending':s.score.copy_(-torch.arange(n,device='cuda').float())
            elif pattern=='signed_zero':
                s.score.zero_();s.score[...,::2]=-0.0
            elif pattern=='infinity':
                s.score[...,::3]=float('inf');s.score[...,1::3]=float('-inf')
        expected,counts=reference(s)
        for tile in (1024,2048,4096):
            original=s.score.clone()
            select(s,3,tile)
            torch.cuda.synchronize()
            assert torch.equal(s.page_table,expected),(n,pattern,tile,'ids')
            assert torch.equal(s.page_cnt,counts),(n,pattern,tile,'counts')
            assert torch.equal(s.score,original),(n,pattern,tile,'score modified')
            checks.append(dict(n=n,pattern=pattern,tile=tile,passed=True))
    print('correctness',n,flush=True)

for cap in (1,4,63,128,256,512,1000,1024,2048,2049,4096,8192,40000):
    s=state(32777,rows=2,heads=3,cap=cap,sink=3)
    s.n_sel_hi.sub_(4)
    expected,counts=reference(s)
    select(s,2,2048)
    assert torch.equal(s.page_table,expected) and torch.equal(s.page_cnt,counts)
    checks.append(dict(n=32777,cap=cap,passed=True))

for pattern in ('equal','ties','infinity'):
    s=state(16393,cap=4096)
    if pattern=='equal':s.score.zero_()
    elif pattern=='ties':s.score.copy_(torch.randint(-1,2,s.score.shape,device='cuda').float())
    else:s.score[...,::2]=float('-inf')
    expected,counts=reference(s)
    select(s,1)
    assert torch.equal(s.page_table,expected) and torch.equal(s.page_cnt,counts)
    checks.append(dict(streaming=pattern,passed=True))

s=state(32777)
select(s,1,2048)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):select(s,1,2048)
for step in range(6):
    s.score.uniform_(0,1)
    s.n_pages.fill_(32760+step)
    s.n_sel_hi.copy_(s.n_pages-1)
    graph.replay()
    expected,counts=reference(s)
    assert torch.equal(s.page_table,expected) and torch.equal(s.page_cnt,counts)
checks.append(dict(graph_replays=6,changing_scores_and_visible_pages=True,passed=True))
a.report.write_text(json.dumps(dict(passed=True,checks=checks,
    seconds=time.time()-start,seed=20261107),indent=2))
print('All correctness checks passed',len(checks),flush=True)
