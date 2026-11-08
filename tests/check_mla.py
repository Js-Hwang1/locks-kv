import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['LOCKS_CONFIG'] = json.dumps(dict(variant='fast',r8_rank=4,budget_pages=8))
os.environ['VLLM_PLUGINS'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from locks.config import LocksConfig
from locks.backend import _runtime
from locks.backend import mla_attn
from locks.selection.mla_select import build_page_factors, topb_middle
from locks.selection.mla_score_triton import mla_page_score_factored_batched

p = argparse.ArgumentParser()
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(20261108)
_runtime.set_config(LocksConfig.load())
rows = []
for page, latent, heads, rank in ((16,512,16,4),(16,512,32,8),(64,512,16,4)):
    _runtime.set_config(SimpleNamespace(budget_pages=8,budget=None,sink_pages=1,window_pages=1,r8_rank=rank))
    batch, pages, rope = 2, 17, 64
    cache = torch.randn(batch*pages,page,latent+rope,device='cuda',dtype=torch.bfloat16)
    qc = torch.randn(batch,heads,latent,device='cuda',dtype=torch.float32)
    qr = torch.randn(batch,heads,rope,device='cuda',dtype=torch.float32)
    lf,rf = build_page_factors(cache[:pages,:,:latent].reshape(-1,latent).float(),pages,page,rank)
    kr = cache[:pages,:,latent:].float().contiguous()
    valid = torch.full((pages,),page,device='cuda',dtype=torch.int32)
    scores = mla_page_score_factored_batched(qc,qr,lf,rf,kr,valid,torch.zeros(pages,device='cuda',dtype=torch.int32),(latent+rope)**-.5)
    full = lf @ rf
    expected = ((torch.einsum('hd,ptd->hpt',qc[0],full)+torch.einsum('hd,ptd->hpt',qr[0],kr))*(latent+rope)**-.5).amax(dim=(0,2))
    score_error = float((scores-expected).abs().max())
    assert score_error < .02, score_error
    keep = topb_middle(scores,8,1,1)
    candidates = sorted(range(1,pages-1),key=lambda i:(-float(scores[i]),i))[:6]
    assert keep.nonzero().flatten().tolist() == sorted([0,pages-1]+candidates)
    impl = mla_attn.LocksMLAImpl.__new__(mla_attn.LocksMLAImpl)
    impl.kv_lora_rank = latent
    impl.scale = (latent+rope)**-.5
    lengths = torch.tensor([pages*page-3,pages*page-7],device='cuda',dtype=torch.int32)
    bt = torch.arange(batch*pages,device='cuda',dtype=torch.int32).reshape(batch,pages)
    metadata = SimpleNamespace(decode=SimpleNamespace(block_table=bt,seq_lens=lengths))
    q = torch.cat((qc,qr),-1).to(torch.bfloat16)
    selected=[]
    original=mla_attn.topb_middle
    def recorded(*args,**kwargs):
        result=original(*args,**kwargs)
        selected.append(result.nonzero().flatten())
        return result
    mla_attn.topb_middle=recorded
    mla_attn._C8CACHE.clear()
    output,lse=impl.forward_mqa(q,cache,metadata,object())
    mla_attn.topb_middle=original
    max_error=0.0
    for request,ids in enumerate(selected):
        assert len(ids)==8 and int(ids[0])==0 and int(ids[-1])==pages-1
        keys=cache[bt[request,ids].long()].reshape(-1,latent+rope).float()
        pos=(ids[:,None]*page+torch.arange(page,device='cuda')).flatten()
        logits=(q[request].float()@keys.T)*impl.scale
        logits[:,pos>=lengths[request]]=float('-inf')
        reference=torch.softmax(logits,-1)@keys[:,:latent]
        error=float(torch.linalg.vector_norm(output[request].float()-reference)/torch.linalg.vector_norm(reference))
        max_error=max(max_error,error)
        assert error < .004
        assert torch.allclose(lse[request].float(),torch.logsumexp(logits,-1),rtol=.005,atol=.04)
    rows.append(dict(page_size=page,latent_dim=latent,query_heads=heads,rank=rank,score_max_abs_error=score_error,attention_relative_l2=max_error,passed=True))
assert mla_attn.LocksMLABackend.get_impl_cls() is mla_attn.LocksMLAImpl
a.report.write_text(json.dumps(dict(passed=True,cases=rows,full_model_checkpoint_test=False),indent=2)+'\n')
print('MLA fused scoring, shared selected budget and original-latent-KV attention passed')
