import triton
import triton.language as tl


@triton.jit
def select_small(SCORES,IDS,COUNTS,SL,NP:tl.constexpr,K:tl.constexpr,
                 STRIDE:tl.constexpr,SHARED:tl.constexpr,BLOCK:tl.constexpr,IS_PAGES:tl.constexpr=False):
    head=tl.program_id(0)
    p=tl.arange(0,BLOCK)
    visible=tl.load(SL) if IS_PAGES else tl.cdiv(tl.load(SL),16)
    low=tl.minimum(visible,1) if SHARED else 0
    high=tl.maximum(low,visible-1) if SHARED else visible
    count=tl.minimum(K-2 if SHARED else K,high-low)
    x=tl.load(SCORES+head*NP+p,(p<NP)&(p>=low)&(p<high),other=float('-inf'))
    bits=x.to(tl.uint32,bitcast=True)
    bits=tl.where(x==0,0,bits).to(tl.uint32)
    ordered=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000).to(tl.uint32)
    packed=(ordered.to(tl.uint64)<<32)|(0xffffffff-p.to(tl.uint64))
    packed=tl.where((p<NP)&(p>=low)&(p<high),packed,0)
    j=tl.arange(0,K)
    chosen=tl.full((K,),2147483647,tl.int32)
    if SHARED:
        chosen=tl.where(j<low,0,chosen)
        chosen=tl.where((j==low+count)&(visible>low),visible-1,chosen)
    for i in range(count):
        best=tl.max(packed,0)
        index=(0xffffffff-(best&0xffffffff)).to(tl.int32)
        chosen=tl.where(j==i+low,index,chosen)
        packed=tl.where(p==index,0,packed)
    chosen=tl.sort(chosen,descending=False)
    valid_count=low+count+tl.where(SHARED & (visible>low),1,0)
    if SHARED:
        tl.store(IDS+head*STRIDE+p,-1,p<NP)
        tl.store(COUNTS+head,valid_count)
    tl.store(IDS+head*STRIDE+j,tl.where(j<valid_count,chosen,-1) if SHARED else chosen)
