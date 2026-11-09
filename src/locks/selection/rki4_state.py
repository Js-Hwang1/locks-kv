from __future__ import annotations
import torch
from ..runtime_options import OPTIONS

TAGW = 8
RNK = OPTIONS["rank"]
assert RNK >= 1, f"LOCKS_RKI4_RANK must be a positive integer, got {RNK}"
IBITS = 4
assert IBITS in (2, 4, 8), f"LOCKS_RKI4_IBITS must be 2|4|8, got {IBITS}"
QMAX = (1 << IBITS - 1) - 1
QMIN = -(1 << IBITS - 1)
MUC = OPTIONS["packed"]
assert not (MUC and RNK >= 8), (
    "MUC stage 1 covers RNK<8 only (mu rides the mma's dead basis columns; none exist at RNK >= 8)"
)
_AOS = OPTIONS["packed"]
_COMPACT_TAGS = _AOS
assert not _COMPACT_TAGS or (_AOS and RNK == 4 and (IBITS == 4))


def _rec_bytes_aos(d: int, page: int) -> int:
    raw = RNK * d // 2 + d + page * RNK + 2 * page + 2 * RNK + 2
    return (raw + 63) // 64 * 64


_REC_BYTES = _rec_bytes_aos(128, 16)


def _v_bytes(d: int) -> int:
    return RNK * ((d * IBITS + 7) // 8)


def record_bytes(d: int, page: int, tagw: int = TAGW, *, packed=None) -> int:
    if packed is None:
        packed = _AOS
    if packed:
        assert d == 128 and page == 16, "AOS records cover the p16 d128 flagship only"
        assert IBITS == 4, "AOS records cover the i4 flagship only"
        assert tagw == TAGW == 8
        return _REC_BYTES
    return _v_bytes(d) + RNK * 2 + page * RNK + page * 2 + d + 2 + tagw * 2


def record_bytes_padded(d: int, page: int, elem: int = 2, tagw: int = TAGW) -> int:
    rb = record_bytes(d, page, tagw)
    q = page * elem
    return (rb + q - 1) // q * q


class Rki4State:
    def __init__(
        self,
        device,
        *,
        num_layers: int,
        num_blocks: int,
        n_kv: int,
        G: int,
        head_dim: int,
        page: int,
        max_reqs: int,
        max_pages: int,
        rank: int = RNK,
        budget: float = 0.1,
        budget_pages=None,
        sink_pages: int = 1,
        window_pages: int = 1,
        tagw: int = TAGW,
        combine: str = "shared_average",
        zsplit=None,
        resident_low_context=False,
    ):
        assert window_pages >= 1, (
            "window_pages>=1: the partial tail must live inside the always-attended window so the selectable region is exact"
        )
        self.validate_geometry(head_dim=head_dim, G=G, page=page, rank=rank)
        assert combine == "shared_average"
        from .shared_average import validate_policy_environment

        validate_policy_environment()
        assert zsplit is None or zsplit >= 1, (
            f"zsplit must be None (auto) or >= 1, got {zsplit}"
        )
        (L, NB, D, MP, R) = (num_layers, num_blocks, head_dim, max_pages, max_reqs)
        (self.L, self.NB, self.n_kv, self.G, self.d) = (L, NB, n_kv, G, D)
        (self.page, self.r, self.tagw) = (page, RNK, tagw)
        self.v_bits = IBITS
        (self.max_reqs, self.max_pages) = (R, MP)
        self.budget = float(budget)
        self.budget_abs = int(budget_pages or 0)
        (self.sink_pages, self.window_pages) = (int(sink_pages), int(window_pages))
        self.combine = combine
        self.zsplit = int(zsplit) if zsplit else None
        self.device = device
        (u8, i8, bf16) = (torch.uint8, torch.int8, torch.bfloat16)
        (i32, f32, f64) = (torch.int32, torch.float32, torch.float64)
        self.low_context = bool(resident_low_context and R == 1 and RNK == 4
            and D == 128 and page == 16 and G == 8 and 1024 <= MP <= 8192)
        packed = _AOS or self.low_context or (
            R == 1 and RNK == 4 and D == 128 and page == 16 and G == 8
            and MP >= 16384
        )
        self.packed_summaries = packed
        if packed:
            assert D == 128 and page == 16, (
                "AOS records cover the p16 d128 flagship only"
            )
            assert IBITS == 4, "AOS records cover the i4 flagship only"
            o_mu = RNK * D // 2
            o_c8 = o_mu + D
            o_cs = o_c8 + page * RNK
            o_vs = o_cs + 2 * page
            o_mus = o_vs + 2 * RNK
            self.pp_rec = torch.zeros(L, NB, n_kv, _REC_BYTES, device=device, dtype=u8)
            r_ = self.pp_rec
            self.pp_v4 = r_[..., 0:o_mu].unflatten(-1, (RNK, D // 2))
            self.pp_mu8 = r_[..., o_mu:o_c8].view(i8)
            self.pp_c8 = r_[..., o_c8:o_cs].view(i8).unflatten(-1, (page, RNK))
            self.pp_cs = r_[..., o_cs:o_vs].view(bf16)
            self.pp_vs = r_[..., o_vs:o_mus].view(bf16)
            self.pp_mus = r_[..., o_mus : o_mus + 2].view(bf16).squeeze(-1)
            self.pp_nrmC = r_[..., o_mus + 2 : o_mus + 4].view(bf16).squeeze(-1)
        else:
            self.pp_rec = None
            assert D * IBITS % 8 == 0, (
                f"V column d={D} x {IBITS} bits must be byte-aligned"
            )
            self.pp_v4 = torch.zeros(
                L, NB, n_kv, RNK, D * IBITS // 8, device=device, dtype=u8
            )
            self.pp_vs = torch.zeros(L, NB, n_kv, RNK, device=device, dtype=bf16)
            self.pp_c8 = torch.zeros(L, NB, n_kv, page, RNK, device=device, dtype=i8)
            self.pp_cs = torch.zeros(L, NB, n_kv, page, device=device, dtype=bf16)
            self.pp_mu8 = torch.zeros(L, NB, n_kv, D, device=device, dtype=i8)
            self.pp_mus = torch.zeros(L, NB, n_kv, device=device, dtype=bf16)
        if packed:
            assert tagw == TAGW == 8
            tag_start = _REC_BYTES - 2 * tagw
            assert tag_start % 16 == 0 and o_mus + 4 <= tag_start
            self.pp_tag = self.pp_rec[..., tag_start:].view(bf16)
            self.pp_tag.fill_(float("nan"))
        else:
            self.pp_tag = torch.full(
                (L, NB, n_kv, tagw), float("nan"), device=device, dtype=bf16
            )
        self.score = torch.empty(R, n_kv, MP, device=device, dtype=f32)
        self.page_table = torch.empty(R, n_kv, MP, device=device, dtype=i32)
        self.page_cnt = torch.zeros(R, n_kv, device=device, dtype=i32)
        self.score_h = torch.empty(R, n_kv, G, MP, device=device, dtype=f32)
        self.n_pages = torch.zeros(R, device=device, dtype=i32)
        self.n_sel_hi = torch.zeros(R, device=device, dtype=i32)
        self.n_score_hi = torch.zeros(R, device=device, dtype=i32)
        self.share_chunks = torch.empty(
            R, n_kv, G, (MP + 511) // 512, device=device, dtype=f32
        )
        self.share_logz = torch.empty(R, n_kv, G, device=device, dtype=f32)
        self.b_fix = torch.ones(R, device=device, dtype=i32)
        self.budget64 = torch.tensor([self.budget], device=device, dtype=f64)
        planned = record_bytes(D, page, tagw, packed=packed) * n_kv * NB
        actual = self.bytes_per_layer()
        assert actual == planned, (
            f"rki4 slab bytes/layer {actual} != record_bytes() plan {planned} (d={D} page={page} tagw={tagw} n_kv={n_kv} NB={NB}) -- update rki4_state.record_bytes to match the allocation"
        )

    @staticmethod
    def validate_geometry(*, head_dim: int, G: int, page: int, rank: int = RNK) -> None:
        assert head_dim in (64, 128, 256), (
            f"rki4 supports d in {{64, 128, 256}}, model has d={head_dim}"
        )
        assert 1 <= G <= 8, (
            f"rki4 supports 1 <= G <= 8 query heads per kv head, got G={G}. The quantized scorer supports at most eight query heads in a KV group."
        )
        assert page in (16, 32), (
            f"The optimized scorer supports pages of 16 or 32 tokens; got {page}"
        )
        assert RNK < page, (
            f"rki4 rank must satisfy 1 <= RNK < page (centered-gram rank ceiling is page-1), got LOCKS_RKI4_RANK={RNK} at page={page}"
        )
        assert rank == RNK, (
            f"rki4 rank is fixed at import by LOCKS_RKI4_RANK={RNK} (the packed layout + kernel -DRNK), got rank={rank}"
        )

    @classmethod
    def record_bytes(cls, d: int, page: int, tagw: int = TAGW) -> int:
        return record_bytes(d, page, tagw)

    def bytes_per_layer(self) -> int:
        if self.pp_rec is not None:
            return self.pp_rec[0].numel()
        return (
            self.pp_v4[0].numel()
            + self.pp_vs[0].numel() * 2
            + self.pp_c8[0].numel()
            + self.pp_cs[0].numel() * 2
            + self.pp_mu8[0].numel()
            + self.pp_mus[0].numel() * 2
            + self.pp_tag[0].numel() * 2
        )

    def layer_state(self, layer: int):
        return (
            self.pp_v4[layer],
            self.pp_vs[layer],
            self.pp_c8[layer],
            self.pp_cs[layer],
            self.pp_mu8[layer],
            self.pp_mus[layer],
            self.pp_tag[layer],
        )
