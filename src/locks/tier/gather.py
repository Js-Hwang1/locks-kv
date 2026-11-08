import torch
import triton
import triton.language as tl


@triton.jit
def _gather_original_keys(
    BLOCKS,
    VBO,
    STAGING,
    HOST,
    OUT,
    N,
    NB,
    LAYER,
    SL: tl.constexpr,
    SB: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    PAGE: tl.constexpr,
    KV: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    TOKEN: tl.constexpr,
    FULL: tl.constexpr,
    TILE: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    PER: tl.constexpr = PAGE * KV * D if FULL else KV * WIDTH
    row = index // PER
    active = row < N
    block = tl.load(BLOCKS + row, mask=active, other=0).to(tl.int64)
    valid = (block >= 0) & (block < NB)
    tl.device_assert(~active | valid, "Invalid original-K block ID")
    slot = tl.load(VBO + block, mask=active & valid, other=-1).to(tl.int64)
    if FULL:
        token = index % PER // (KV * D)
        head = index // D % KV
        dim = index % D
    else:
        token = TOKEN
        head = index // WIDTH % KV
        dim = index % WIDTH
    layer = (tl.program_id(1) + LAYER).to(tl.int64)
    stage_offset = layer * SL + tl.maximum(slot, 0) * SB + token * ST + head * SH + dim
    host_offset = ((layer * NB + block) * KV + head) * (PAGE * D) + token * D + dim
    pointer = tl.where(slot >= 0, STAGING + stage_offset, HOST + host_offset)
    value = tl.load(pointer, mask=active & valid, other=0)
    tl.store(OUT + tl.program_id(1) * N * PER + index, value, mask=active)


def gather(tier, layer, blocks, *, token=None, width=None):
    return gather_layers(tier, layer, 1, blocks, token=token, width=width)[0]


def gather_layers(tier, first_layer, num_layers, blocks, *, token=None, width=None):
    assert tier.offload_k and tier.pool_k is not None
    assert 0 <= first_layer < first_layer + num_layers <= tier.L
    blocks = blocks.to(device=tier.device, dtype=torch.long).reshape(-1)
    full = token is None
    if full:
        width = tier.d
        token = 0
        shape = (num_layers, blocks.numel(), tier.page, tier.n_kv, tier.d)
    else:
        assert 0 <= token < tier.page and 0 <= width <= tier.d
        shape = (num_layers, blocks.numel(), tier.n_kv, width)
    out = torch.empty(shape, device=tier.device, dtype=tier.dtype)
    if not out.numel():
        return out
    stage = tier.staging.k_pool.view(torch.int16)
    _gather_original_keys[triton.cdiv(out.numel() // num_layers, 256), num_layers](
        blocks,
        tier.staging.vbo,
        stage,
        tier.pool_k.dev_view,
        out.view(torch.int16),
        blocks.numel(),
        tier.NB,
        first_layer,
        SL=stage.stride(0),
        SB=stage.stride(1),
        ST=stage.stride(2),
        SH=stage.stride(3),
        PAGE=tier.page,
        KV=tier.n_kv,
        D=tier.d,
        WIDTH=width,
        TOKEN=token,
        FULL=full,
        TILE=256,
        num_warps=4,
        debug=True,
    )
    return out
