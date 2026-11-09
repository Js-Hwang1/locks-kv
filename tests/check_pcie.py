"""Original-KV attention, eviction, shared prefixes and graph replay on PCIe."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['LOCKS_CONFIG'] = json.dumps(dict(variant='mem-kv', r8_rank=4, budget_pages=14))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from locks.config import LocksConfig
from locks.tier.tier import Tier
from locks.tier.decode_mem import mem_decode_ptrsel

p = argparse.ArgumentParser()
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
torch.cuda.set_device(0)
torch.empty(1, device='cuda').zero_()
torch.manual_seed(29)
torch.set_num_threads(2)
L, R, NK, G, P, D, MP, C, SPLIT = 2, 2, 4, 8, 16, 128, 257, 16, 128
rows = []

for hot_slots, fused in ((64, True), (7, True), (64, False), (7, False)):
    cfg = LocksConfig(variant='mem-kv', budget_pages=C-2, hot_slots=hot_slots,
                      mem_transport='pcie')
    tier = Tier.from_config(cfg, num_layers=L, num_blocks=R*MP, n_kv=NK,
        page=P, d=D, max_reqs=R, max_pages=MP, max_tokens=R+8,
        dtype=torch.bfloat16, device=torch.device('cuda'))
    tier.pcie_cache.fused = fused  # Also exercise the large-geometry fallback.
    k = torch.randn((L,R*MP,NK,P,D), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tier.pool_k.host.copy_(k)
    tier.pool.host.copy_(v)
    tier.residency.pool_valid.fill_(1)
    bt = torch.arange(R*MP, device='cuda', dtype=torch.int32).view(R,MP)
    bt[1,:32] = bt[0,:32]  # Exercise duplicate claims for a shared prefix.
    sl = torch.full((R,),4096+7,device='cuda',dtype=torch.int32)
    current = bt[:,-1].long()
    tier.staging.vbo[current] = torch.arange(R,device='cuda',dtype=torch.int32)
    tier.staging.k_pool[:,:R].copy_(k[:,current].permute(0,1,3,2,4))
    tier.staging.v_pool[:,:R].copy_(v[:,current].permute(0,1,3,2,4))
    q = torch.randn((R,NK*G,D),device='cuda',dtype=torch.bfloat16)
    tab = torch.zeros((R,NK,MP),device='cuda',dtype=torch.int32)
    cnt = torch.full((R,NK),C,device='cuda',dtype=torch.int32)
    st = SimpleNamespace(G=G,split=SPLIT,page_table=tab,page_cnt=cnt,
        budget_abs=C-2,sink_pages=1,window_pages=1,
        m_part=torch.zeros((R,NK,SPLIT,G),device='cuda'),
        l_part=torch.zeros((R,NK,SPLIT,G),device='cuda'),
        acc_part=torch.zeros((R,NK,SPLIT,G,D),device='cuda'))
    output = torch.empty((L,R,NK*G,D),device='cuda',dtype=torch.bfloat16)

    def decode(layer, fused_fill=False):
        mem_decode_ptrsel(q,None,bt,sl,st,output[layer],tier,layer,scale=1/math.sqrt(D), fused_fill=fused_fill)

    def prepare_and_decode():
        for layer in range(L):
            tier.pcie_cache.prepare(layer,tab,cnt,bt,sl)
            decode(layer, fused_fill=tier.pcie_cache.fused)
            tier.pcie_cache.finish(layer)

    # Warm all kernels before capture; empty rows must remain safe.
    cnt.zero_()
    prepare_and_decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_and_decode()

    maximum_error = 0.0
    for iteration in range(8):
        ids = torch.cat([torch.tensor([0]),torch.arange(1,15)+iteration*17,torch.tensor([256])])
        tab[:,:,:C] = ids.to(device='cuda',dtype=torch.int32)
        cnt.fill_(C)
        if iteration % 2:
            cnt[0,0],cnt[0,1],cnt[1,2] = 0,1,7
        if iteration == 4:
            # Simulate a completed write/flush of the shared first page.
            tier.residency.invalidate_written(bt,torch.ones_like(sl),
                                              torch.arange(R+1,device='cuda',dtype=torch.int32))
            blocks = bt[:,0].long().unique()
            k[:,blocks] += 0.5
            v[:,blocks] -= 0.25
            tier.pool_k.host[:,blocks.cpu()] = k[:,blocks].cpu()
            tier.pool.host[:,blocks.cpu()] = v[:,blocks].cpu()
            tier.residency.pool_valid[:,:,blocks] = 1
        for layer in range(L):
            decode(layer)
        before = output.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output,before), ('PCIe fill changed attention',hot_slots,iteration)
        # Replaying the same selection exercises the no-miss kernel as well
        # as the fill kernel. Staged partial pages must remain readable.
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output,before), ('Warm-cache attention changed',hot_slots,iteration)
        if hot_slots == 64:
            assert not bool(tier.pcie_cache.counts.any()), 'Expected a fully cached replay'
        physical = bt[:,None,:].expand(R,NK,MP).gather(2,tab[:,:,:C].long()).long()
        heads = torch.arange(NK,device='cuda')[None,:,None]
        tokens = (tab[:,:,:C].long()[:,:,:,None]*P+torch.arange(P,device='cuda')).flatten(-2)
        mask = (torch.arange(C*P,device='cuda')[None,None,:] < cnt[:,:,None]*P) & (tokens < sl[:,None,None])
        for layer in range(L):
            sk = k[layer][physical,heads].reshape(R,NK,C*P,D).float()
            sv = v[layer][physical,heads].reshape(R,NK,C*P,D).float()
            logits = torch.einsum('rhgd,rhtd->rhgt',q.view(R,NK,G,D).float(),sk)/math.sqrt(D)
            logits.masked_fill_(~mask[:,:,None,:],-float('inf'))
            weights = logits.softmax(-1).nan_to_num()
            expected = torch.einsum('rhgt,rhtd->rhgd',weights,sv).reshape(R,NK*G,D)
            error = float((output[layer].float()-expected).abs().max())
            maximum_error = max(maximum_error,error)
            assert torch.isfinite(output[layer]).all() and error < 0.012, (hot_slots,iteration,layer,error)
        assert not bool((tier.residency.page2slot < -1).any())
        tier.residency.assert_maps_inverse()
        for layer in range(L):
            for head in range(NK):
                blocks = torch.nonzero(tier.residency.page2slot[layer,head]>=0).flatten()
                slots = tier.residency.page2slot[layer,head,blocks].long()
                assert torch.equal(tier.residency.hot_k[layer,head,slots],k[layer,blocks,head])
                assert torch.equal(tier.residency.hot[layer,head,slots],v[layer,blocks,head])
    rows.append(dict(hot_slots=hot_slots,fused_attention_fill=fused,graph_replays=16,layers=L,
                     includes_warm_cache_replay=True,
                     bitwise_equal_to_direct_original_kv=True,
                     max_absolute_float32_reference_error=maximum_error,passed=True))
    del graph,tier,k,v
    torch.cuda.empty_cache()

a.report.write_text(json.dumps(dict(passed=True,cases=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
