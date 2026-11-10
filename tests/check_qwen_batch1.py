import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2',VLLM_PLUGINS='',
    LOCKS_CONFIG=json.dumps(dict(variant='fast',r8_rank=4)),LOCKS_RKI4_RANK='4')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from locks.selection.rki4_state import Rki4State
    from locks.selection.rki4_build import _rki4_write
    from locks.selection.rki4_score_cuda import rki4_score_state
    from locks.selection.select import derive_page_params
    from locks.selection.select_cuda import topb_select_cuda
    from locks.attention.decode import ensure_stage_b_buffers,ResidentVSource,sparse_paged_decode_batched
    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    max_pages=2086
    keys=torch.randn(max_pages,16,8,128,device='cuda',dtype=torch.bfloat16)*.2
    values=torch.randn_like(keys)
    cache=torch.stack((keys,values))
    original=cache.clone()
    block_ids=torch.arange(max_pages,device='cuda')
    layers=torch.zeros_like(block_ids)
    table=torch.randperm(max_pages,device='cuda').int()[None]
    query=torch.randn(1,32,128,device='cuda',dtype=torch.bfloat16)
    out=torch.empty_like(query)
    rows=[]
    for budget in (64,128,256,512,1024,2048):
        geometry=dict(device='cuda',num_layers=1,num_blocks=max_pages,n_kv=8,G=4,
            head_dim=128,page=16,max_pages=max_pages,rank=4,budget_pages=budget//16-2,
            sink_pages=1,window_pages=1,combine='shared_average')
        reference=Rki4State(max_reqs=2,**geometry)
        candidate=Rki4State(max_reqs=1,resident_low_context=True,**geometry)
        assert reference.pp_rec is None and candidate.pp_rec is not None and candidate.low_context
        for state in (reference,candidate):
            state.D=state.d
            _rki4_write(state,keys,layers,block_ids)
        ensure_stage_b_buffers(candidate,torch.device('cuda'),128)
        vsource=ResidentVSource(cache)
        for zero,visible in [(z,n) for z in (False,True) for n in (1,15,16,17,31,32,33,63,64,65,127,128,129,149,160,598,608,1023,1024,1025,2048,2049,16384,33364)]:
            if zero:query.zero_()
            else:query.normal_()
            sl=torch.tensor([visible],device='cuda',dtype=torch.int32)
            def select(state):
                derive_page_params(state,sl,1)
                rki4_score_state(state,0,query,table,sl,1,128**-.5,kv_cache=cache)
                topb_select_cuda(state,1)
            select(reference);select(candidate)
            n=(visible+15)//16
            torch.testing.assert_close(reference.score_h[0,...,:n],candidate.score_h[0,...,:n],atol=1.e-6,rtol=1.e-6)
            logit_difference=float((reference.score_h[0,...,:n]-candidate.score_h[0,...,:n]).abs().max())
            torch.testing.assert_close(reference.score[0],candidate.score[0],atol=1.e-7,rtol=1.e-6)
            assert torch.equal(reference.page_cnt[0],candidate.page_cnt[0])
            count=min(budget//16,n)
            assert torch.equal(reference.page_table[0,:,:count],candidate.page_table[0,:,:count])
            ids=candidate.page_table[0,:,:count]
            assert torch.all(candidate.page_cnt[0]==count)
            expected=[]
            for h in range(8):
                keep=ids[h].long()
                assert keep.unique().numel()==count and int(keep.min())==0 and int(keep.max())==n-1
                positions=(keep[:,None]*16+torch.arange(16,device='cuda')[None,:]).flatten()
                positions=positions[positions<visible]
                physical=table[0,positions//16].long()
                key=keys[physical,positions%16,h].float()
                val=values[physical,positions%16,h].float()
                q=query[0,h*4:(h+1)*4].float()
                expected.append(torch.softmax(q@key.T*128**-.5,-1)@val)
            sparse_paged_decode_batched(query,cache,table,sl,candidate,out,vsource,scale=128**-.5)
            target=torch.cat(expected)[None]
            error=float((target-out.float()).norm()/target.norm().clamp_min(1.e-12))
            assert torch.isfinite(out).all() and error<.02,error
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                select(candidate)
                sparse_paged_decode_batched(query,cache,table,sl,candidate,out,vsource,scale=128**-.5)
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(reference.page_table[0,:,:count],candidate.page_table[0,:,:count])
            row=dict(budget=budget,visible_tokens=visible,zero_query_ties=zero,pages_per_kv_head=count,
                group_size=4,shared_selected_set=True,protected_pages_inside_budget=2,
                packed_logits_bitwise_equal=logit_difference==0,max_logit_difference=logit_difference,shares_close=True,selected_ids_identical=True,
                relative_attention_error=error,cuda_graph=True)
            rows.append(row);print(json.dumps(row),flush=True)
            del graph
        del reference,candidate
        torch.cuda.empty_cache()
    assert torch.equal(cache,original)
    (args.output/'report.json').write_text(json.dumps(dict(passed=True,rows=rows,original_kv_unchanged=True),indent=2)+'\n')


if __name__=='__main__':main()
