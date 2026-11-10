import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='4')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
import torch
from locks.selection.topk_scalable import select

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(20261110)
rows=[]
for cap in (14,30,62,126):
    score=torch.randn(1,8,2086,device='cuda')
    st=SimpleNamespace(score=score,max_pages=2086,n_kv=8,G=4,sink_pages=1,window_pages=1,
        low_context=True,budget_abs=cap,budget=.1,n_pages=torch.zeros(1,device='cuda',dtype=torch.int32),
        n_sel_hi=torch.zeros(1,device='cuda',dtype=torch.int32),b_fix=torch.tensor([cap],device='cuda',dtype=torch.int32),
        page_table=torch.empty(1,8,2086,device='cuda',dtype=torch.int32),page_cnt=torch.empty(1,8,device='cuda',dtype=torch.int32))
    select(st,1)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):select(st,1)
    for pattern in ('random','ties','zero','infinity'):
        for n in (2086,0,1,2,3,16,37,513,1023,1024,1025,2047,2048,2049):
            st.n_pages.fill_(n);st.n_sel_hi.fill_(max(0,n-1))
            score.normal_()
            if pattern=='ties':score.round_()
            elif pattern=='zero':score.zero_()
            elif pattern=='infinity':score[...,::3]=float('inf');score[...,1::3]=float('-inf')
            before=score.clone();graph.replay()
            expected=torch.full_like(st.page_table,-1)
            for h in range(8):
                data=score[0,h].cpu().tolist()
                keep=sorted(set(range(min(n,1)))|set(range(max(1,n-1),n))|
                            set(sorted(range(1,max(1,n-1)),key=lambda i:(-data[i],i))[:cap]))
                expected[0,h,:len(keep)]=torch.tensor(keep,device='cuda',dtype=torch.int32)
            assert torch.equal(st.page_table,expected),(cap,pattern,n,'ids')
            assert torch.all(st.page_cnt==min(n,cap+2))
            assert torch.equal(score,before)
            rows.append(dict(cap=cap,pattern=pattern,visible_pages=n,graph=True,passed=True))
    print(json.dumps(dict(cap=cap,cases=len(rows))),flush=True)
with (a.output/'report.json').open('x') as f:json.dump(dict(passed=True,rows=rows),f,indent=2)
