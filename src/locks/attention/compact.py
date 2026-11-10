import triton
import triton.language as tl


@triton.jit
def compact_selected_attention(Q,K,V,BT,IDS,COUNTS,SL,OUT,
    SKB:tl.constexpr,SKT:tl.constexpr,SKH:tl.constexpr,
    SVB:tl.constexpr,SVT:tl.constexpr,SVH:tl.constexpr,
    ID_STRIDE:tl.constexpr,SHARED:tl.constexpr,USE_COUNTS:tl.constexpr,
    TOKENS:tl.constexpr,SCALE:tl.constexpr):
    h=tl.program_id(0)
    ih=h//4 if SHARED else h
    t=tl.arange(0,TOKENS)
    d=tl.arange(0,128)
    count=tl.load(COUNTS+ih) if USE_COUNTS else TOKENS//16
    page=tl.load(IDS+ih*ID_STRIDE+t//16,t//16<count,other=-1)
    visible=tl.load(SL)
    valid=(t//16<count)&(page>=0)&(page<tl.cdiv(visible,16))&(page*16+t%16<visible)
    block=tl.load(BT+page,valid,other=0).to(tl.int64)
    q=tl.load(Q+h*128+d).to(tl.float32)
    k=tl.load(K+block[:,None]*SKB+(t%16)[:,None]*SKT+(h//4)*SKH+d[None,:],
        valid[:,None],other=0).to(tl.float32)
    score=tl.sum(k*q[None,:],1)*SCALE
    score=tl.where(valid,score,float('-inf'))
    peak=tl.max(score,0)
    prob=tl.exp(score-tl.where(peak==float('-inf'),0,peak))
    denom=tl.sum(prob,0)
    prob=prob/tl.where(denom>0,denom,1)
    v=tl.load(V+block[:,None]*SVB+(t%16)[:,None]*SVT+(h//4)*SVH+d[None,:],
        valid[:,None],other=0).to(tl.float32)
    value=tl.sum(prob[:,None]*v,0)
    tl.store(OUT+h*128+d,value)
