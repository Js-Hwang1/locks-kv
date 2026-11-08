from __future__ import annotations
import traceback
import torch
from .rki4_build import _eigh_chunked, _write_post, _write_pre

_WRAP_EXT = None
_INPROF = [None]


def _in_profiling() -> bool:
    if _INPROF[0] is None:
        from ..backend._runtime import in_profiling

        _INPROF[0] = in_profiling
    return _INPROF[0]()


def _wrap_dev_ptr(ptr: int, shape, device: torch.device) -> torch.Tensor:
    global _WRAP_EXT
    if _WRAP_EXT is None:
        from torch.utils.cpp_extension import load_inline

        _WRAP_EXT = load_inline(
            name="locks_devptr_wrap",
            cpp_sources=r"""
#include <torch/extension.h>
torch::Tensor wrap_i16(int64_t ptr, std::vector<int64_t> shape,
                       int64_t dev) {
  auto opts = torch::TensorOptions().dtype(torch::kInt16)
      .device(torch::kCUDA, (int)dev);
  return torch::from_blob(reinterpret_cast<void*>(ptr), shape,
                          [](void*) {}, opts);
}
""",
            functions=["wrap_i16"],
            with_cuda=True,
            verbose=False,
        )
    return _WRAP_EXT.wrap_i16(
        int(ptr), list(shape), device.index if device.index is not None else 0
    )


class _TailGraph:
    def __init__(self):
        self.ready = False
        self.dead = False

    def _derive_blocks(self):
        fin1 = (self.sl_st.to(torch.int64) // self.page - 1).clamp_(min=0)
        blk = self.bt_st.to(torch.int64).gather(0, fin1)
        self.blocks.copy_(blk.clamp_(0, self.nb - 1))

    def _gather_fast(self):
        Kp = torch.stack([K[self.blocks] for K in self._k_layers])
        return Kp.reshape(self.L, self.page, self.n_kv, self.d)

    def _gather_mem(self):
        b = self.blocks
        vb = self.vbo[b].to(torch.int64)
        stg = (vb >= 0).view(1, 1, 1, 1, 1)
        ks = self.kpool[:, vb.clamp(min=0)]
        kp = self.kpin[:, b].view(torch.bfloat16)
        kp = kp.permute(0, 1, 3, 2, 4)
        sel = torch.where(stg, ks, kp)
        return sel.reshape(self.L, self.page, self.n_kv, self.d)

    def _stage_a(self, st):
        self._derive_blocks()
        return _write_pre(st, self._gather())

    def _capture(self, st):
        mem = self._mem
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                (tag, mu, dc, Gm) = self._stage_a(st)
                (S2, U) = _eigh_chunked(Gm.reshape(-1, self.page, self.page))
                _write_post(
                    st, mu, dc, S2, U, tag, self.lidx, self.blocks.repeat(self.L)
                )
        torch.cuda.current_stream().wait_stream(side)
        self.gA = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gA):
            (self.tag, self.mu, self.dc, self.Gm) = self._stage_a(st)
        self.gA.replay()
        self.Gm_flat = self.Gm.reshape(-1, self.page, self.page)
        (S2, U) = _eigh_chunked(self.Gm_flat)
        self.S2 = S2.clone()
        self.U = U.clone()
        self.gB = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gB):
            _write_post(
                st,
                self.mu,
                self.dc,
                self.S2,
                self.U,
                self.tag,
                self.lidx,
                self.blocks.repeat(self.L),
            )
        self.gB.replay()
        self.ready = True
        print(
            f"[locks] TAIL_GRAPH captured (v2): {('mem' if mem else 'fast')} L={self.L} page={self.page} n_kv={self.n_kv} nb={self.nb} W={self.W}",
            flush=True,
        )

    def _init_static(self, st, K_layers, block_table, seq_lens):
        dev = seq_lens.device
        (self.L, self.page) = (st.L, st.page)
        (self.n_kv, self.d) = (st.n_kv, st.d)
        self.nb = (
            st.pp_rec.shape[1]
            if getattr(st, "pp_rec", None) is not None
            else st.pp_v4.shape[1]
        )
        self.W = int(block_table.shape[1])
        self.bt_st = torch.zeros(self.W, dtype=torch.int32, device=dev)
        self.sl_st = torch.zeros(1, dtype=torch.int32, device=dev)
        self.blocks = torch.zeros(1, dtype=torch.int64, device=dev)
        self.lidx = torch.arange(st.L, device=dev, dtype=torch.int64)
        self._mem = not torch.is_tensor(K_layers[0])
        if self._mem:
            tier = K_layers[0]._tier
            self.vbo = tier.staging.vbo
            self.kpool = tier.staging.k_pool
            self.kpin = _wrap_dev_ptr(
                tier.pool_k.dev_view.data_ptr(), tier.pool_k.host.shape, dev
            )
            self._src_key = id(tier)
            self._gather = self._gather_mem
        else:
            self._src_key = id(K_layers[0])
            self._k_layers = list(K_layers)
            self._gather = self._gather_fast

    def run(self, st, K_layers, block_table, seq_lens, n_req, rows):
        if self.dead or n_req != 1 or len(K_layers) == 0:
            return None
        if rows is not None and rows.numel() != 1:
            return None
        try:
            if torch.cuda.is_current_stream_capturing() or _in_profiling():
                return None
            if not self.ready:
                self._init_static(st, K_layers, block_table, seq_lens)
                self.bt_st.copy_(block_table[0, : self.W])
                self.sl_st.copy_(seq_lens[0:1])
                self._capture(st)
                return st.L
            src = id(K_layers[0]) if not self._mem else id(K_layers[0]._tier)
            if src != self._src_key or int(block_table.shape[1]) != self.W:
                self.dead = True
                print(
                    f"[locks] FALLBACK TAIL_GRAPH: K source/block-table width changed (W={self.W} -> {int(block_table.shape[1])}) -- the captured graph is invalid; EAGER tail refresh for the rest of this run",
                    flush=True,
                )
                return None
            self.bt_st.copy_(block_table[0, : self.W])
            self.sl_st.copy_(seq_lens[0:1])
            self.gA.replay()
            (S2, U) = _eigh_chunked(self.Gm_flat)
            self.S2.copy_(S2)
            self.U.copy_(U)
            self.gB.replay()
            return st.L
        except Exception as e:
            self.dead = True
            print(
                "[locks] TAIL_GRAPH failed -- refusing the silent eager degrade (kill switch: LOCKS_TAIL_GRAPH=0):",
                flush=True,
            )
            traceback.print_exc()
            raise RuntimeError(
                f"locks TAIL_GRAPH: tail-refresh graph capture/replay failed ({type(e).__name__}: {e}); no silent eager fallback -- set LOCKS_TAIL_GRAPH=0 to select the eager tail explicitly."
            ) from e


def tail_graph_run(st, K_layers, block_table, seq_lens, n_req, rows):
    tg = getattr(st, "_tailgraph", None)
    if tg is None:
        tg = st._tailgraph = _TailGraph()
    return tg.run(st, K_layers, block_table, seq_lens, n_req, rows)
