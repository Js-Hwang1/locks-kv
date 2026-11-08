from __future__ import annotations
import torch
from .. import arch as _arch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#ifndef NW
#define NW 4
#endif
#define TB (NW * 32)
#ifndef RKI4_PFR_STAGES
#define RKI4_PFR_STAGES 2
#endif
#ifndef RNK
#define RNK 8
#endif
#if RNK != 8 && RNK != 4 && RNK != 2
#error "RNK must be 8, 4, or 2"
#endif


#define MAXG 16


#ifndef LOCKS_ROT_DIM
#define LOCKS_ROT_DIM 64
#endif
#ifndef LOCKS_ROT_NEOX
#define LOCKS_ROT_NEOX 0
#endif
#define LOCKS_ROT_HALF (LOCKS_ROT_DIM / 2)


__device__ __forceinline__ float locks_rope_elem(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ crow, const int dd) {
#if LOCKS_ROT_NEOX

    const int lo = (dd < LOCKS_ROT_HALF) ? dd : dd - LOCKS_ROT_HALF;
    const float c = __bfloat162float(__ldg(crow + lo));
    const float s = __bfloat162float(__ldg(crow + LOCKS_ROT_HALF + lo));
    const float e = __bfloat162float(x[lo]);
    const float o = __bfloat162float(x[lo + LOCKS_ROT_HALF]);
    return (dd < LOCKS_ROT_HALF)
        ? __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s))
        : __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s));
#else

    const int pr = dd >> 1;
    const float c = __bfloat162float(__ldg(crow + pr));
    const float s = __bfloat162float(__ldg(crow + LOCKS_ROT_HALF + pr));
    const float e = __bfloat162float(x[2 * pr]);
    const float o = __bfloat162float(x[2 * pr + 1]);
    return (dd & 1) ? __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s))
                    : __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s));
#endif
}


#define RKI4_RING 3
__device__ __forceinline__ void cpasync16(void* dst_smem, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(s), "l"(src));
}
__device__ __forceinline__ void cpasync_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cpasync_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


__device__ __forceinline__ void r8_mma_s8(int d[4], const unsigned a[4],
                                          const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}


__device__ __forceinline__ unsigned r8_lo(unsigned w) {
    return __vsub4((w & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
}
__device__ __forceinline__ unsigned r8_hi(unsigned w) {
    return __vsub4(((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
}


#define RKI4_RECB_HOST (((RNK * 64 + 128 + 16 * RNK + 32 + 2 * RNK + 2) + 63) / 64 * 64)


#ifdef RKI4_MMA_AOS

#if defined(RKI4_MMA_PFR) && !defined(RKI4_MMA_AOS)
#error "RKI4_MMA_PFR requires RKI4_MMA_AOS"
#endif


#if defined(RKI4_MMA_MUC) && (RNK == 8 || !defined(RKI4_MMA_AOS) || !defined(RKI4_MMA_BIAS) || defined(RKI4_MMA_S1PROBE))
#error "RKI4_MMA_MUC stage 1 covers RNK<8 AOS+BIAS builds only"
#endif


#endif


#define RKI4_G8_MINBLK 0


template <int PGT, int DHEAD, bool BT, bool G4, bool G8 = false, bool PF = false>
__global__ __launch_bounds__(TB, (G4 && DHEAD == 128 && PGT == 16) ? 6
                                 : RKI4_G8_MINBLK)
void rki4_score_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ v4,
    const __nv_bfloat16* __restrict__ vs,
    const int8_t* __restrict__ c8,
    const __nv_bfloat16* __restrict__ cs,
    const int8_t* __restrict__ mu8,
    const __nv_bfloat16* __restrict__ mus,
    const int* __restrict__ bt,
    const int* __restrict__ nsh,
    float* __restrict__ S,
    const int P_dense, const int MP, const int G, const float sm_scale,
    const long q_sr, const long q_sh, const long bt_sr,
    float* __restrict__ gpmax_out,
    int*   __restrict__ ghist_out,

    const float* __restrict__ Kflat,

    const int* __restrict__ sl_rope,
    const __nv_bfloat16* __restrict__ csc_rope,
    __nv_bfloat16* __restrict__ qout_rope,


    unsigned* __restrict__ hmax_pub)


{
    constexpr int VW4 = DHEAD / 32;
    constexpr int MUW = DHEAD / 32;


    constexpr bool USE_MMA = G8 && (DHEAD == 128);
    constexpr bool PIPE = (DHEAD < 256) && G4 && !USE_MMA;


#ifndef RKI4_MMA_AOS
    constexpr bool HM_FOLD_MMA = true;
#else
    constexpr bool HM_FOLD_MMA = false;
#endif
    constexpr bool HM_FOLD = (USE_MMA && HM_FOLD_MMA) || (!USE_MMA && G8);
    float hm_a = -CUDART_INF_F, hm_b = -CUDART_INF_F;
    const int r    = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nworker = gridDim.z * NW;
    const int wid = zsp * NW + warp;
    const int gl  = lane >> 3;
    const int rr  = lane & 7;
    const int n_kv = gridDim.y;
    const int P = BT ? __ldg(nsh + r) : P_dense;

    __shared__ float qsh[MAXG * DHEAD];


    __shared__ uint32_t q8e[MAXG][DHEAD / 8], q8o[MAXG][DHEAD / 8];
    __shared__ uint32_t q8d[MAXG][DHEAD / 4];
    __shared__ float qsc[MAXG];


    __shared__ unsigned s_hm[8];
    if (hmax_pub != nullptr && tid < 8) s_hm[tid] = 0x007FFFFFu;
    if (qout_rope) {


        const long pos = (long)__ldg(sl_rope + r) - 1;
        const __nv_bfloat16* crow = csc_rope + (long)LOCKS_ROT_DIM * pos;
        for (int i = tid; i < G * DHEAD; i += TB) {
            const int h = i / DHEAD, d = i % DHEAD;
            const long qb = (long)r * q_sr + (long)(kh * G + h) * q_sh;
            float v;
            if (d < LOCKS_ROT_DIM) {
                v = locks_rope_elem(q + qb, crow, d);
            } else {
                v = __bfloat162float(q[qb + d]);
            }
            const __nv_bfloat16 vb = __float2bfloat16(v);
            qout_rope[((long)r * n_kv * G + (long)kh * G + h) * DHEAD + d]
                = vb;
            qsh[i] = __bfloat162float(vb);
        }
    } else {
        for (int i = tid; i < G * DHEAD; i += TB)
            qsh[i] = __bfloat162float(
                q[(long)r * q_sr + (long)(kh * G + i / DHEAD) * q_sh
                  + (i % DHEAD)]);
    }
    __syncthreads();


    for (int gg = warp; gg < G; gg += NW) {
        const float* qw = qsh + gg * DHEAD;
        float am = 0.f;
        for (int j = lane; j < DHEAD; j += 32) am = fmaxf(am, fabsf(qw[j]));
        #pragma unroll
        for (int st = 16; st >= 1; st >>= 1)
            am = fmaxf(am, __shfl_xor_sync(~0u, am, st));
        const float qs = fmaxf(am, 1e-8f) / 127.f;
        if (lane == 0) qsc[gg] = qs;
        if (lane < DHEAD / 8) {
            uint32_t we = 0, wo = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int pe = (int)rintf(qw[8 * lane + 2 * b] / qs);
                const int po = (int)rintf(qw[8 * lane + 2 * b + 1] / qs);
                we |= (uint32_t)(uint8_t)(int8_t)pe << (8 * b);
                wo |= (uint32_t)(uint8_t)(int8_t)po << (8 * b);
            }
            q8e[gg][lane] = we; q8o[gg][lane] = wo;
        }

        for (int j = lane; j < DHEAD / 4; j += 32) {
            uint32_t wd = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int pd = (int)rintf(qw[4 * j + b] / qs);
                wd |= (uint32_t)(uint8_t)(int8_t)pd << (8 * b);
            }
            q8d[gg][j] = wd;
        }
    }
    __syncthreads();


    unsigned Af[4][4];
    int qsum_mma = 0;
    __shared__ float qt_sh[NW][8][8];
    if constexpr (USE_MMA) {
        const int gidm = lane >> 2, t4m = lane & 3;
        Af[0][0] = q8e[gidm][t4m];      Af[0][1] = 0u;
        Af[0][2] = q8e[gidm][4 + t4m];  Af[0][3] = 0u;
        Af[1][0] = q8e[gidm][8 + t4m];  Af[1][1] = 0u;
        Af[1][2] = q8e[gidm][12 + t4m]; Af[1][3] = 0u;
        Af[2][0] = q8o[gidm][t4m];      Af[2][1] = 0u;
        Af[2][2] = q8o[gidm][4 + t4m];  Af[2][3] = 0u;
        Af[3][0] = q8o[gidm][8 + t4m];  Af[3][1] = 0u;
        Af[3][2] = q8o[gidm][12 + t4m]; Af[3][3] = 0u;


        #pragma unroll
        for (int j = 0; j < DHEAD / 4 / 4; ++j)
            qsum_mma = __dp4a((int)q8d[gidm][8 * t4m + j], 0x01010101, qsum_mma);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 1);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 2);
        qsum_mma *= 8;
    }

    const long pb_v = (long)kh * (long)P_dense;
    const int* btr = BT ? (bt + (long)r * bt_sr) : nullptr;
    auto row_of = [&](int pp) -> long {
        if constexpr (BT) return (long)__ldg(btr + pp) * n_kv + kh;
        else              return pb_v + pp;
    };
    auto load_page = [&](int pp, uint4* vv, uint32_t* mm, uint2* cc,
                         __nv_bfloat16& sv, __nv_bfloat16& smu,
                         __nv_bfloat16* sc) {
        const long row = row_of(pp);
#ifdef RKI4_MMA_AOS


        constexpr int  RO_MU_L  = RNK * (DHEAD / 2);
        constexpr int  RO_C8_L  = RO_MU_L + DHEAD;
        constexpr int  RO_CS_L  = RO_C8_L + RNK * PGT;
        constexpr int  RO_VS_L  = RO_CS_L + 2 * PGT;
        constexpr int  RO_MUS_L = RO_VS_L + 2 * RNK;
        constexpr long RECB_L   = ((RO_MUS_L + 2) + 63) / 64 * 64;
        const uint8_t* p_rec = v4 + row * RECB_L;
        const int vc = (rr < RNK) ? rr : (RNK - 1);
        const uint4* colv = reinterpret_cast<const uint4*>(
            p_rec + (long)vc * (DHEAD / 2));
        #pragma unroll
        for (int c = 0; c < VW4; ++c) vv[c] = __ldg(colv + c);


        if constexpr (MUW % 4 == 0) {
            const uint4* mup = reinterpret_cast<const uint4*>(
                p_rec + RO_MU_L) + rr * (MUW / 4);
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(mup + j);
                mm[4 * j + 0] = m4.x; mm[4 * j + 1] = m4.y;
                mm[4 * j + 2] = m4.z; mm[4 * j + 3] = m4.w;
            }
        } else {
            const uint32_t* mup = reinterpret_cast<const uint32_t*>(
                p_rec + RO_MU_L) + rr * MUW;
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mm[j] = __ldg(mup + j);
        }


        #pragma unroll
        for (int tt = 0; tt < PGT / 8; ++tt) {
            const uint8_t* pc = p_rec + RO_C8_L + (long)(rr + 8 * tt) * RNK;
#if RNK == 8
            cc[tt] = __ldg(reinterpret_cast<const uint2*>(pc));
#elif RNK == 4
            cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(pc));
            cc[tt].y = 0u;
#else
            cc[tt].x = (unsigned)__ldg(
                reinterpret_cast<const unsigned short*>(pc));
            cc[tt].y = 0u;
#endif
            sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                p_rec + RO_CS_L + (long)(rr + 8 * tt) * 2);
        }
        sv  = *reinterpret_cast<const __nv_bfloat16*>(
            p_rec + RO_VS_L + (long)vc * 2);
        smu = *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_MUS_L);
#else


        const int vcn = (rr < RNK) ? rr : (RNK - 1);
        const uint4* colv = reinterpret_cast<const uint4*>(
            v4 + (row * RNK + vcn) * (DHEAD / 2));
        #pragma unroll
        for (int c = 0; c < VW4; ++c) vv[c] = __ldg(colv + c);
        if constexpr (MUW % 4 == 0) {
            const uint4* mup = reinterpret_cast<const uint4*>(
                mu8 + row * DHEAD) + rr * (MUW / 4);
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(mup + j);
                mm[4 * j + 0] = m4.x; mm[4 * j + 1] = m4.y;
                mm[4 * j + 2] = m4.z; mm[4 * j + 3] = m4.w;
            }
        } else {
            const uint32_t* mup = reinterpret_cast<const uint32_t*>(
                mu8 + row * DHEAD) + rr * MUW;
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mm[j] = __ldg(mup + j);
        }
        #pragma unroll
        for (int tt = 0; tt < PGT / 8; ++tt) {
            const int8_t* pc = c8 + (long)(row * PGT + rr + 8 * tt) * RNK;
#if RNK == 8
            cc[tt] = __ldg(reinterpret_cast<const uint2*>(pc));
#elif RNK == 4
            cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(pc));
            cc[tt].y = 0u;
#else
            cc[tt].x = (unsigned)__ldg(
                reinterpret_cast<const unsigned short*>(pc));
            cc[tt].y = 0u;
#endif
            sc[tt] = cs[row * PGT + rr + 8 * tt];
        }
        sv = vs[row * RNK + vcn];
        smu = mus[row];
#endif
    };


    auto compute_page = [&](int p, const uint4* vv, const uint32_t* mm,
                            const uint2* cc, __nv_bfloat16 sv,
                            __nv_bfloat16 smu, const __nv_bfloat16* sc) {
        const int ob = lane & 24;
        auto score_tile = [&](int g, bool active) {


            int acci = 0;
            #pragma unroll
            for (int c = 0; c < VW4; ++c) {
                const uint32_t ws[4] = {vv[c].x, vv[c].y, vv[c].z, vv[c].w};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {

                    const uint32_t lo = __vsub4(
                        (ws[k] & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                    const uint32_t hi = __vsub4(
                        ((ws[k] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u,
                        0x08080808u);
                    acci = __dp4a((int)lo, (int)q8e[g][c * 4 + k], acci);
                    acci = __dp4a((int)hi, (int)q8o[g][c * 4 + k], acci);
                }
            }


            float qt = (rr < RNK) ? ((float)acci * qsc[g] * __bfloat162float(sv))
                                  : 0.f;

            int mui = 0;
            #pragma unroll
            for (int j = 0; j < MUW; ++j)
                mui = __dp4a((int)mm[j], (int)q8d[g][rr * MUW + j], mui);
            float md = (float)mui;
            md += __shfl_xor_sync(~0u, md, 1);
            md += __shfl_xor_sync(~0u, md, 2);
            md += __shfl_xor_sync(~0u, md, 4);
            const float mud = md * qsc[g] * __bfloat162float(smu);

            float mx = -CUDART_INF_F, es = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * __shfl_sync(~0u, qt, ob + u);
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * __shfl_sync(~0u, qt, ob + 4 + u);
                const float tok = sm_scale * (dt * __bfloat162float(sc[tt])
                    + mud);
                const float nm = fmaxf(mx, tok);
                es = es * __expf(mx - nm) + __expf(tok - nm);
                mx = nm;
            }
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1) {
                const float om = __shfl_xor_sync(~0u, mx, st);
                const float oe = __shfl_xor_sync(~0u, es, st);
                const float nm = fmaxf(mx, om);
                es = es * __expf(mx - nm) + oe * __expf(om - nm);
                mx = nm;
            }
            {
                const float osv = mx + __logf(es);
                if (rr == 0 && active)
                    S[(((long)r * n_kv + kh) * G + g) * MP + p] = osv;
                return osv;
            }
        };
        if constexpr (G4) {
            score_tile(gl, gl < G);
        } else if constexpr (G8) {


            {
                const float hsa = score_tile(gl, true);
                const float hsb = score_tile(gl + 4, true);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa);
                    hm_b = fmaxf(hm_b, hsb);
                }
            }
        } else {
            for (int g0 = 0; g0 < G; g0 += 4)
                score_tile(g0 + gl, g0 + gl < G);
        }
    };
    int p = wid;
    if constexpr (USE_MMA) {


        const int gidm = lane >> 2, t4m = lane & 3;
        uint32_t mm[MUW]; uint2 cc[PGT / 8];
        __nv_bfloat16 smu, sc[PGT / 8];


        float* qtw = &qt_sh[warp][gidm][2 * t4m];


        constexpr int BOFF = G8 ? 4 : 0;
        const float* qt_a = &qt_sh[warp][gl][0];
        const float* qt_b = &qt_sh[warp][gl + BOFF][0];
        const uint32_t* q8d_a = &q8d[gl][rr * MUW];
        const uint32_t* q8d_b = &q8d[gl + BOFF][rr * MUW];
        const float qsc_m = qsc[gidm];
        const float qsc_a = qsc[gl], qsc_b = qsc[gl + BOFF];
        constexpr float K_a = 0.f, K_b = 0.f;
        float* Sp_a = S + ((((long)r * n_kv + kh) * G + gl) * MP) + wid;
        float* Sp_b = Sp_a + (long)BOFF * MP;
        const long row0 = BT ? 0L : (pb_v + wid);
        const uint4* p_mu = reinterpret_cast<const uint4*>(
            mu8 + row0 * DHEAD) + rr * (MUW / 4);
        const int8_t* p_c8 = c8 + (row0 * PGT + rr) * RNK;
        const __nv_bfloat16* p_cs = cs + row0 * PGT + rr;
        const __nv_bfloat16* p_mus = mus + row0;


        const int gvc = (gidm < RNK) ? gidm : (RNK - 1);
        const int svc = (2 * t4m < RNK) ? 2 * t4m : (RNK - 2 < 0 ? 0 : RNK - 2);
        const unsigned* p_v4 = reinterpret_cast<const unsigned*>(
            v4 + (row0 * RNK + gvc) * (DHEAD / 2)) + t4m;
        const __nv_bfloat16* p_vs = vs + row0 * RNK + svc;


        auto tile = [&](const float* qtg, const uint32_t* q8dg,
                        const float qscg, float* Sp, const float Kg) {
#ifdef RKI4_MMA_MUC


            const float mud = qtg[RNK] * qscg * __bfloat162float(smu);
#else
            int mui = 0;
            #pragma unroll
            for (int j = 0; j < MUW; ++j)
                mui = __dp4a((int)mm[j], (int)q8dg[j], mui);
            float md = (float)mui;
            md += __shfl_xor_sync(~0u, md, 1);
            md += __shfl_xor_sync(~0u, md, 2);
            md += __shfl_xor_sync(~0u, md, 4);
            const float mud = md * qscg * __bfloat162float(smu);
#endif


            float tokv[PGT / 8];
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < (RNK < 4 ? RNK : 4); ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * qtg[u];
#if RNK == 8
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * qtg[4 + u];
#endif
                tokv[tt] = sm_scale * (dt * __bfloat162float(sc[tt]) + mud);
            }
            float mx = tokv[0];
            #pragma unroll
            for (int tt = 1; tt < PGT / 8; ++tt) mx = fmaxf(mx, tokv[tt]);
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, st));
            float es = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt)
                es += __expf(tokv[tt] - mx);
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1)
                es += __shfl_xor_sync(~0u, es, st);
            {

                const float osv = mx + __logf(es);
                if (rr == 0) *Sp = osv;
                return osv;
            }
        };
#ifdef RKI4_MMA_AOS


        {
        constexpr int  RO_MU  = RNK * (DHEAD / 2);


        constexpr int  RO_C8  = RO_MU + DHEAD;
        constexpr int  RO_CS  = RO_C8 + RNK * PGT;
        constexpr int  RO_VS  = RO_CS + 2 * PGT;
        constexpr int  RO_MUS = RO_VS + 2 * RNK;
        constexpr long RECB   = ((RO_MUS + 2) + 63) / 64 * 64;
        const uint8_t* p_rec = v4 + (BT ? 0L : (pb_v + wid)) * RECB;


        const long rmax_rec = BT ? ((long)P_dense * n_kv - 1) : 0L;
#ifdef RKI4_MMA_PFR


        {
        constexpr int RC16 = (int)(RECB / 16);
        const int chunk = (P + nworker - 1) / nworker;
        const int first_page = wid * chunk;
        const int last_page = min(P, first_page + chunk);
        const int page_stride = 1;
        Sp_a += first_page - wid;
        Sp_b += first_page - wid;
        __shared__ uint4 recring[NW][RKI4_PFR_STAGES][RC16];
        auto issue_rec = [&](int pv, int slot) {
            if (pv < last_page) {
                long rowb;
                if constexpr (BT) {
                    rowb = row_of(pv);
                    rowb = rowb < 0 ? 0 : (rowb > rmax_rec ? rmax_rec : rowb);
                } else {
                    rowb = pb_v + pv;
                }
                const uint4* r16 = reinterpret_cast<const uint4*>(
                    v4 + rowb * RECB);
                for (int c = lane; c < RC16; c += 32)
                    cpasync16(&recring[warp][slot][c], r16 + c);
            }
            cpasync_commit();
        };
        #pragma unroll
        for (int stage = 0; stage < RKI4_PFR_STAGES; ++stage)
            issue_rec(first_page + stage * page_stride, stage);
        int pslot = 0;
        for (int pp = first_page; pp < last_page; pp += page_stride) {
            cpasync_wait<RKI4_PFR_STAGES - 1>();
            __syncwarp();
            const uint8_t* p_smem = reinterpret_cast<const uint8_t*>(
                &recring[warp][pslot][0]);


#ifndef RKI4_MMA_MUC
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = *(reinterpret_cast<const uint4*>(
                    p_smem + RO_MU) + rr * (MUW / 4) + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
#endif
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                cc[tt] = *(reinterpret_cast<const uint2*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = *(reinterpret_cast<const unsigned*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)*(reinterpret_cast<const unsigned short*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                    p_smem + RO_CS + (rr + 8 * tt) * 2);
            }
            smu = *reinterpret_cast<const __nv_bfloat16*>(p_smem + RO_MUS);
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_MUC
            const bool ismu = (gidm == RNK);
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_smem + (ismu ? RO_MU : gidm * (DHEAD / 2)));
#else
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_smem + gidm * (DHEAD / 2));
#endif
            const unsigned cw0 = cwp[t4m],      cw1 = cwp[4 + t4m],
                           cw2 = cwp[8 + t4m],  cw3 = cwp[12 + t4m];
#ifdef RKI4_MMA_MUC
            const unsigned cm4 = ismu ? cwp[16 + t4m] : 0u,
                           cm5 = ismu ? cwp[20 + t4m] : 0u,
                           cm6 = ismu ? cwp[24 + t4m] : 0u,
                           cm7 = ismu ? cwp[28 + t4m] : 0u;
#endif
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                p_smem + RO_VS + 4 * t4m);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();
            issue_rec(pp + RKI4_PFR_STAGES * page_stride, pslot);
            tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
            if constexpr (G8)
                tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
            pslot = (pslot + 1) % RKI4_PFR_STAGES;
            Sp_a += page_stride; Sp_b += page_stride;
            __syncwarp();
        }
        cpasync_wait<0>();
        }
#else
        for (int pp = wid; pp < P; pp += nworker) {
            if constexpr (BT) {
                long rowb = row_of(pp);
                rowb = rowb < 0 ? 0 : (rowb > rmax_rec ? rmax_rec : rowb);
                p_rec = v4 + rowb * RECB;
            }

#ifndef RKI4_MMA_MUC
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(reinterpret_cast<const uint4*>(
                    p_rec + RO_MU) + rr * (MUW / 4) + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
#endif
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)__ldg(reinterpret_cast<const unsigned short*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                    p_rec + RO_CS + (rr + 8 * tt) * 2);
            }
            smu = *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_MUS);

            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_MUC
            const bool ismu = (gidm == RNK);
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_rec + (ismu ? RO_MU : gidm * (DHEAD / 2)));
#else
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_rec + gidm * (DHEAD / 2));
#endif
            const unsigned cw0 = __ldg(cwp + t4m),      cw1 = __ldg(cwp + 4 + t4m),
                           cw2 = __ldg(cwp + 8 + t4m),  cw3 = __ldg(cwp + 12 + t4m);
#ifdef RKI4_MMA_MUC
            const unsigned cm4 = ismu ? __ldg(cwp + 16 + t4m) : 0u,
                           cm5 = ismu ? __ldg(cwp + 20 + t4m) : 0u,
                           cm6 = ismu ? __ldg(cwp + 24 + t4m) : 0u,
                           cm7 = ismu ? __ldg(cwp + 28 + t4m) : 0u;
#endif
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                p_rec + RO_VS + 4 * t4m);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();

            tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
            if constexpr (G8)
                tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
            if constexpr (!BT) p_rec += (long)nworker * RECB;
            Sp_a += nworker; Sp_b += nworker;
            __syncwarp();
        }
#endif
        }
#else


        if constexpr (BT && PGT == 16 && PF) {
            unsigned nv0 = 0u, nv1 = 0u, nv2 = 0u, nv3 = 0u;
            if (wid < P) {
                const long rowp = row_of(wid);
                const unsigned* pf = reinterpret_cast<const unsigned*>(
                    v4 + (rowp * RNK + gidm) * (DHEAD / 2)) + t4m;
                nv0 = __ldg(pf);      nv1 = __ldg(pf + 4);
                nv2 = __ldg(pf + 8);  nv3 = __ldg(pf + 12);
            }
            for (int pp = wid; pp < P; pp += nworker) {
                const long row = row_of(pp);
                p_mu = reinterpret_cast<const uint4*>(
                    mu8 + row * DHEAD) + rr * (MUW / 4);
                p_c8 = c8 + (row * PGT + rr) * RNK;
                p_cs = cs + row * PGT + rr;
                p_mus = mus + row;
                p_vs = vs + row * RNK + svc;

                #pragma unroll
                for (int j = 0; j < MUW / 4; ++j) {
                    const uint4 m4 = __ldg(p_mu + j);
                    mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
                }
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                    cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                        p_c8 + (8 * tt) * RNK));
#elif RNK == 4
                    cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                        p_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
#else
                    cc[tt].x = (unsigned)__ldg(reinterpret_cast<
                        const unsigned short*>(p_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
#endif
                    sc[tt] = p_cs[8 * tt];
                }
                smu = *p_mus;


                const unsigned cw0 = nv0, cw1 = nv1, cw2 = nv2, cw3 = nv3;
                const int ppn = pp + nworker;
                if (ppn < P) {
                    const long rown = row_of(ppn);
                    const unsigned* pf = reinterpret_cast<const unsigned*>(
                        v4 + (rown * RNK + gidm) * (DHEAD / 2)) + t4m;
                    nv0 = __ldg(pf);      nv1 = __ldg(pf + 4);
                    nv2 = __ldg(pf + 8);  nv3 = __ldg(pf + 12);
                }


                unsigned A0[4], A1[4], A2[4], A3[4];
                A0[0]=q8e[gidm][t4m];      A0[1]=0u; A0[2]=q8e[gidm][4 + t4m];  A0[3]=0u;
                A1[0]=q8e[gidm][8 + t4m];  A1[1]=0u; A1[2]=q8e[gidm][12 + t4m]; A1[3]=0u;
                A2[0]=q8o[gidm][t4m];      A2[1]=0u; A2[2]=q8o[gidm][4 + t4m];  A2[3]=0u;
                A3[0]=q8o[gidm][8 + t4m];  A3[1]=0u; A3[2]=q8o[gidm][12 + t4m]; A3[3]=0u;
                int acc[4] = {0, 0, 0, 0};
                unsigned b[2];
#ifdef RKI4_MMA_MUC
                b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A0, b);
                b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A1, b);
                b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A2, b);
                b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A3, b);
#else
                b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A0, b);
                b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A1, b);
                b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A2, b);
                b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A3, b);
#endif
#if RNK == 8
                const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                               : (2 * t4m + 0 == RNK ? acc[0] : 0);
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                               : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
                const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(p_vs);
#ifdef RKI4_MMA_MUC
                qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                         : (float)c0 * qsc_m * __bfloat162float(sv2.x);
                qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                         : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
                qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
                qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
                __syncwarp();

                {
                    const float hsa = tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
                    const float hsb = tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
                    if (hmax_pub != nullptr) {
                        hm_a = fmaxf(hm_a, hsa);
                        hm_b = fmaxf(hm_b, hsb);
                    }
                }
                Sp_a += nworker; Sp_b += nworker;
                __syncwarp();
            }
        } else {
        for (int pp = wid; pp < P; pp += nworker) {
            if constexpr (BT) {
                const long row = row_of(pp);
                p_mu = reinterpret_cast<const uint4*>(
                    mu8 + row * DHEAD) + rr * (MUW / 4);
                p_c8 = c8 + (row * PGT + rr) * RNK;
                p_cs = cs + row * PGT + rr;
                p_mus = mus + row;
                p_v4 = reinterpret_cast<const unsigned*>(
                    v4 + (row * RNK + gvc) * (DHEAD / 2)) + t4m;
                p_vs = vs + row * RNK + svc;
            }


            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(p_mu + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if   RNK == 8
                cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                    p_c8 + (8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                    p_c8 + (8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)__ldg(reinterpret_cast<
                    const unsigned short*>(p_c8 + (8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = p_cs[8 * tt];
            }
            smu = *p_mus;

            const unsigned cw0 = __ldg(p_v4),      cw1 = __ldg(p_v4 + 4),
                           cw2 = __ldg(p_v4 + 8),  cw3 = __ldg(p_v4 + 12);
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(p_vs);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();

            {
                const float hsa = tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
                const float hsb = tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa);
                    hm_b = fmaxf(hm_b, hsb);
                }
            }
            if constexpr (!BT) {
                p_mu  += (long)nworker * (DHEAD / 16);
                p_c8  += (long)nworker * PGT * RNK;
                p_cs  += (long)nworker * PGT;
                p_mus += nworker;
                p_v4  += (long)nworker * (RNK * DHEAD / 8);
                p_vs  += (long)nworker * RNK;
            }
            Sp_a += nworker; Sp_b += nworker;
            __syncwarp();
        }
        }
#endif
    } else if constexpr (PIPE) {


        uint4 va[VW4]; uint32_t mua[MUW]; uint2 ca[PGT / 8];
        __nv_bfloat16 vsa, musa, csa[PGT / 8];
        if (p < P) load_page(p, va, mua, ca, vsa, musa, csa);
        while (p < P) {
            const int pn = p + nworker;
            uint4 vb[VW4]; uint32_t mub[MUW]; uint2 cb[PGT / 8];
            __nv_bfloat16 vsb, musb, csb[PGT / 8];
            if (pn < P) load_page(pn, vb, mub, cb, vsb, musb, csb);
            compute_page(p, va, mua, ca, vsa, musa, csa);
            p = pn;
            #pragma unroll
            for (int c = 0; c < VW4; ++c) va[c] = vb[c];
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mua[j] = mub[j];
            vsa = vsb; musa = musb;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                ca[tt] = cb[tt]; csa[tt] = csb[tt];
            }
        }
    } else {


        uint4 va[VW4]; uint32_t mua[MUW]; uint2 ca[PGT / 8];
        __nv_bfloat16 vsa, musa, csa[PGT / 8];
        for (; p < P; p += nworker) {
            load_page(p, va, mua, ca, vsa, musa, csa);
            compute_page(p, va, mua, ca, vsa, musa, csa);
        }
    }


    if (gpmax_out != nullptr) {
        __threadfence_block();
        __syncthreads();
        const int fgroup = r * n_kv + kh;
        for (int i = zsp * (int)blockDim.x + tid; i < 256;
             i += (int)gridDim.z * (int)blockDim.x)
            ghist_out[(long)fgroup * 256 + i] = 0;
        __shared__ float foldm[NW][8];
        const int fg = lane >> 2, fs = lane & 3;
        float fm = -CUDART_INF_F;
        if (fg < G) {
            const float* Sg = S + (((long)r * n_kv + kh) * G + fg) * MP;
            for (long pp = (long)wid + (long)fs * nworker; pp < P;
                 pp += 4L * nworker)
                fm = fmaxf(fm, Sg[pp]);
        }
        fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 1));
        fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 2));
        if (fs == 0) foldm[warp][fg] = fm;
        __syncthreads();
        if (tid < 8) {
            float m2 = -CUDART_INF_F;
            const int nw = ((int)blockDim.x + 31) >> 5;
            for (int w = 0; w < nw; ++w) m2 = fmaxf(m2, foldm[w][tid]);
            gpmax_out[((long)fgroup * 8 + tid) * gridDim.z + zsp] = m2;
        }
    }


    if (hmax_pub != nullptr) {
        if constexpr (HM_FOLD) {


            if (rr == 0) {
                unsigned ua = __float_as_uint(hm_a);
                ua ^= (unsigned)(-(int)(ua >> 31)) | 0x80000000u;
                if (ua != 0x007FFFFFu && gl < G) atomicMax(&s_hm[gl], ua);
                unsigned ub = __float_as_uint(hm_b);
                ub ^= (unsigned)(-(int)(ub >> 31)) | 0x80000000u;
                if (ub != 0x007FFFFFu && gl + 4 < G)
                    atomicMax(&s_hm[gl + 4], ub);
            }
            __syncthreads();
            if (tid < 8 && s_hm[tid] != 0x007FFFFFu)
                atomicMax(&hmax_pub[((long)r * n_kv + kh) * 8 + tid],
                          s_hm[tid]);
        } else {
            __threadfence_block();
            __syncwarp();
            const int hg = lane >> 2, hs = lane & 3;
            float hf = -CUDART_INF_F;
            if (hg < G) {
                const float* Sg = S + (((long)r * n_kv + kh) * G + hg) * MP;
                for (long pp = (long)wid + (long)hs * nworker; pp < P;
                     pp += 4L * nworker)
                    hf = fmaxf(hf, Sg[pp]);
            }
            hf = fmaxf(hf, __shfl_xor_sync(~0u, hf, 1));
            hf = fmaxf(hf, __shfl_xor_sync(~0u, hf, 2));
            if (hs == 0 && hg < G) {
                unsigned u = __float_as_uint(hf);
                u ^= (unsigned)(-(int)(u >> 31)) | 0x80000000u;
                if (u != 0x007FFFFFu) atomicMax(&s_hm[hg], u);
            }
            __syncthreads();
            if (tid < 8 && s_hm[tid] != 0x007FFFFFu)
                atomicMax(&hmax_pub[((long)r * n_kv + kh) * 8 + tid],
                          s_hm[tid]);
        }
    }
}

using KFn = void (*)(const __nv_bfloat16*, const uint8_t*,
                     const __nv_bfloat16*, const int8_t*,
                     const __nv_bfloat16*, const int8_t*,
                     const __nv_bfloat16*, const int*, const int*, float*,
                     const int, const int, const int, const float,
                     const long, const long, const long, float*, int*,
                     const float*, const int*, const __nv_bfloat16*,
                     __nv_bfloat16*, unsigned*);

static KFn pick_kernel(int PGT, int DHEAD, bool bt_mode, bool g4, bool g8,
                       bool pf = false) {


    if (pf && bt_mode && g8 && PGT == 16 && DHEAD == 128)
        return rki4_score_kernel<16, 128, true, false, true,  true>;
#define RKI4_CASE(PGT_, DH_) if (PGT == PGT_ && DHEAD == DH_) { if (bt_mode) { if (g4) return rki4_score_kernel<PGT_, DH_, true, true, false>; if (g8) return rki4_score_kernel<PGT_, DH_, true, false, true>; return rki4_score_kernel<PGT_, DH_, true, false, false>; } if (g4) return rki4_score_kernel<PGT_, DH_, false, true, false>; if (g8) return rki4_score_kernel<PGT_, DH_, false, false, true>; return rki4_score_kernel<PGT_, DH_, false, false, false>; }
    RKI4_CASE(16, 64) RKI4_CASE(16, 128) RKI4_CASE(16, 256)
    RKI4_CASE(32, 64) RKI4_CASE(32, 128) RKI4_CASE(32, 256)
#undef RKI4_CASE
    TORCH_CHECK(false, "rki4: unsupported geometry page=", PGT,
                " d=", DHEAD, " (supported: page {16,32} x d {64,128,256})");
    return nullptr;
}


static inline bool g8_enabled() {


    return true;
}

static void check_common(const torch::Tensor& q, const torch::Tensor& v4,
                         const torch::Tensor& vs, const torch::Tensor& c8,
                         const torch::Tensor& cs, const torch::Tensor& mu8,
                         const torch::Tensor& mus, const torch::Tensor& S,
                         int n_kv, int PGT, int DHEAD, int G) {
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(q.stride(2) == 1, "q innermost dim must be contiguous");
    TORCH_CHECK(q.size(1) == (long)n_kv * G, "q heads != n_kv * G");
    TORCH_CHECK(q.size(2) == DHEAD, "q head dim != summary d");
    TORCH_CHECK(G >= 1 && G <= MAXG, "G must be in [1, ", MAXG, "], got ", G);
    TORCH_CHECK(v4.size(2) == RNK && v4.size(3) == DHEAD / 2, "v4 layout");
    TORCH_CHECK(c8.size(2) == PGT && c8.size(3) == RNK, "c8 layout");
    TORCH_CHECK(v4.is_contiguous() && vs.is_contiguous() && c8.is_contiguous()
                && cs.is_contiguous() && mu8.is_contiguous()
                && mus.is_contiguous(), "summary tensors must be contiguous");
    TORCH_CHECK(S.is_contiguous() && S.dim() == 4 && S.size(1) == n_kv
                && S.size(2) == G, "S must be contiguous (R, n_kv, G, MP)");
    TORCH_CHECK(S.scalar_type() == torch::kFloat, "S must be fp32");
}

void r8score(torch::Tensor q, torch::Tensor v4, torch::Tensor vs,
             torch::Tensor c8, torch::Tensor cs, torch::Tensor mu8,
             torch::Tensor mus, torch::Tensor S, double sm_scale,
             int64_t zsplit)
{

    const int R = q.size(0);
    const int n_kv = v4.size(0);
    const int P = v4.size(1);
    const int PGT = c8.size(2);
    const int DHEAD = v4.size(3) * 2;
    const int G = S.size(2);
    check_common(q, v4, vs, c8, cs, mu8, mus, S, n_kv, PGT, DHEAD, G);
    TORCH_CHECK(S.size(0) == R && S.size(3) == P, "S shape vs (R, P)");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535,
                "zsplit must be in [1, 65535] (CUDA gridDim.z limit)");
    KFn kfn = pick_kernel(PGT, DHEAD,  false,  G <= 4,  (G == 8) && g8_enabled());
    dim3 grid((unsigned)R, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        v4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr()),
        c8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr()),
        mu8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(mus.data_ptr()),
        nullptr, nullptr,
        S.data_ptr<float>(), P,  P, G, (float)sm_scale,
        q.stride(0), q.stride(1), 0L, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "r8score launch: ", cudaGetErrorString(e));
}

void r8score_aos(torch::Tensor q, torch::Tensor rec, torch::Tensor S,
                 double sm_scale, int64_t zsplit)
{

#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)S; (void)sm_scale; (void)zsplit;
    TORCH_CHECK(false, "r8score_aos requires an RKI4_MMA_AOS build");
#else
    const int R = q.size(0);
    const int n_kv = rec.size(0);
    const int P = rec.size(1);
    const int G = S.size(2);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (n_kv, P, 832) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(0) == R && S.size(1) == n_kv && S.size(3) == P, "S");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "r8score_aos covers the G in {4,8} d128 p16 records only");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    KFn kfn = pick_kernel(16, 128,  false,  (G <= 4),  (G == 8));
    dim3 grid((unsigned)R, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr,
        S.data_ptr<float>(), P,  P, G, (float)sm_scale,
        q.stride(0), q.stride(1), 0L, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "r8score_aos launch: ",
                cudaGetErrorString(e));
#endif
}

void rki4_score_bt_aos(torch::Tensor q, torch::Tensor rec, torch::Tensor bt,
                       torch::Tensor nsh, torch::Tensor S, double sm_scale,
                       int64_t zsplit, int64_t n_req)
{

#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)bt; (void)nsh; (void)S; (void)sm_scale;
    (void)zsplit; (void)n_req;
    TORCH_CHECK(false, "rki4_score_bt_aos requires an RKI4_MMA_AOS build");
#else
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (NB, n_kv, 832) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(1) == n_kv, "S must be (R, n_kv, G, MP) fp32");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "rki4_score_bt_aos covers G in {4,8} d128 p16 records only");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    KFn kfn = pick_kernel(16, 128,  true,  (G <= 4),  (G == 8));
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        bt.data_ptr<int>(), nsh.data_ptr<int>(),

        S.data_ptr<float>(), (int)rec.size(0), MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0), nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt_aos launch: ",
                cudaGetErrorString(e));
#endif
}


void rki4_score_bt(torch::Tensor q, torch::Tensor v4, torch::Tensor vs,
                   torch::Tensor c8, torch::Tensor cs, torch::Tensor mu8,
                   torch::Tensor mus, torch::Tensor bt, torch::Tensor nsh,
                   torch::Tensor S, double sm_scale, int64_t zsplit,
                   int64_t n_req,
                   c10::optional<torch::Tensor> sl_rope,
                   c10::optional<torch::Tensor> csc_rope,
                   c10::optional<torch::Tensor> qout_rope,
                   c10::optional<torch::Tensor> hmax_pub)
{


    const int* slp = sl_rope ? sl_rope->data_ptr<int>() : nullptr;
    const __nv_bfloat16* cscp = csc_rope
        ? reinterpret_cast<const __nv_bfloat16*>(csc_rope->data_ptr())
        : nullptr;
    __nv_bfloat16* qoutp = qout_rope
        ? reinterpret_cast<__nv_bfloat16*>(qout_rope->data_ptr())
        : nullptr;
    TORCH_CHECK(!qoutp || (slp && cscp),
                "norope: qout_rope requires sl_rope + csc_rope");


    unsigned* hmp = nullptr;
    if (hmax_pub) {
        TORCH_CHECK(hmax_pub->scalar_type() == torch::kInt
                    && hmax_pub->is_contiguous()
                    && hmax_pub->numel() >= (long)n_req * v4.size(1) * 8,
                    "hmax_pub must be contiguous int32 (R, n_kv, 8)");
        hmp = reinterpret_cast<unsigned*>(hmax_pub->data_ptr<int>());
    }
    if (qoutp) {
        TORCH_CHECK(sl_rope->scalar_type() == torch::kInt
                    && csc_rope->scalar_type() == torch::kBFloat16
                    && csc_rope->stride(1) == 1
                    && csc_rope->size(1) == LOCKS_ROT_DIM
                    && qout_rope->scalar_type() == torch::kBFloat16
                    && qout_rope->is_contiguous(),
                    "norope: sl int32, csc (P,LOCKS_ROT_DIM) bf16 row-contig, "
                    "qout contiguous bf16");
    }

    const int n_kv = v4.size(1);
    const int PGT = c8.size(2);
    const int DHEAD = v4.size(3) * 2;
    const int G = S.size(2);
    const int MP = S.size(3);
    check_common(q, v4, vs, c8, cs, mu8, mus, S, n_kv, PGT, DHEAD, G);
    TORCH_CHECK(c8.size(1) == n_kv, "c8 layout (NB, n_kv, PGT, RNK)");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535,
                "zsplit must be in [1, 65535] (CUDA gridDim.z limit)");


    constexpr int _pf_minreq = 8;
    constexpr bool _pf_off = false;
    const bool g8 = (G == 8) && g8_enabled();
    const bool pf_gate = !_pf_off && g8 && (n_req >= _pf_minreq);
    static bool _pf_announced = false;
    if (!_pf_announced) {
        _pf_announced = true;
        fprintf(stderr, "[locks] FIX1_PF gate: n_req=%d minreq=%d -> PF=%s\n",
                n_req, _pf_minreq, pf_gate ? "true(prefetch)" : "false(deployed)");
    }
    KFn kfn = pick_kernel(PGT, DHEAD,  true,  G <= 4,
                           g8,  pf_gate);
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        v4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr()),
        c8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr()),
        mu8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(mus.data_ptr()),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), 0, MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0), nullptr, nullptr, nullptr,
        slp, cscp, qoutp, hmp);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt launch: ",
                cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m_) {
    m_.def("r8score", &r8score);
    m_.def("r8score_aos", &r8score_aos);
    m_.def("rki4_score_bt", &rki4_score_bt,
           py::arg("q"), py::arg("v4"), py::arg("vs"), py::arg("c8"),
           py::arg("cs"), py::arg("mu8"), py::arg("mus"), py::arg("bt"),
           py::arg("nsh"), py::arg("S"), py::arg("sm_scale"),
           py::arg("zsplit"), py::arg("n_req"),
           py::arg("sl_rope") = py::none(),
           py::arg("csc_rope") = py::none(),
           py::arg("qout_rope") = py::none(),
           py::arg("hmax_pub") = py::none());
    m_.def("rki4_score_bt_aos", &rki4_score_bt_aos);

}
"""
_MOD = None
_SMS = None
_NOROPE = False
GMX = 8


def _get(verbose=False):
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        from .rki4_state import RNK, _AOS

        if not _arch.is_hopper():
            raise RuntimeError(
                "The optimized scorer currently requires a Hopper GPU (SM90)"
            )
        flags = [
            "-O3",
            "--use_fast_math",
            _arch.arch_flag(),
            f"-DRNK={RNK}",
            "-DNW=4",
            "-DRKI4_PFR_STAGES=3",
            "-DRKI4_MMA",
            "-DRKI4_MMA_BIAS",
            "-DRKI4_MMA_PEEL",
            "-DRKI4_MMA_SRED",
            "-DRKI4_MMA_LSE2",
            "-DRKI4_PFR_CONTIG",
            "-DRKI4_PFR_EARLY",
        ]
        if _AOS:
            flags += ["-DRKI4_MMA_AOS", "-DRKI4_MMA_MUC", "-DRKI4_MMA_PFR"]
        _MOD = load_inline(
            name="locks_rki4_score",
            cpp_sources="",
            cuda_sources=_SRC,
            extra_cuda_cflags=flags,
            verbose=verbose,
        )
    return _MOD


def _sm_count() -> int:
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _SMS


def auto_zsplit(n_req: int, n_kv: int, max_pages: int, mult=None) -> int:
    if mult is None:
        mult = 5 if n_req <= 4 else 4
    target = mult * _sm_count()
    z = -(-target // max(n_req * n_kv, 1))
    return max(1, min(z, max(1, max_pages // 8), 65535))


def _lowbatch_mult(n_req, max_pages, G, d):
    if n_req <= 4 and G == 8 and (d == 128):
        return max(5, min(8, max_pages // 1024))
    return None


def rki4_score_only(st, layer, q, block_table, seq_lens, n_req, scale):
    z = st.zsplit or auto_zsplit(
        n_req,
        st.n_kv,
        st.max_pages,
        mult=_lowbatch_mult(n_req, st.max_pages, st.G, st.d),
    )
    if int(block_table.shape[1]) < int(st.max_pages):
        st.score_h.zero_()
    elif st.pp_rec is not None:
        _get().rki4_score_bt_aos(
            q,
            st.pp_rec[layer],
            block_table,
            st.n_score_hi,
            st.score_h,
            float(scale),
            int(z),
            int(n_req),
        )
    else:
        (v4, vs, c8, cs, mu8, mus, _tag) = st.layer_state(layer)
        _get().rki4_score_bt(
            q,
            v4,
            vs,
            c8,
            cs,
            mu8,
            mus,
            block_table,
            st.n_score_hi,
            st.score_h,
            float(scale),
            int(z),
            int(n_req),
        )


def rki4_select_only(
    st, n_req: int, *, q, kv_cache, block_table, seq_lens, scale, layer
) -> None:
    from .shared_average import complete_current_and_combine

    st._hmax_armed = False
    complete_current_and_combine(
        st,
        n_req,
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        layer=layer,
    )


def rki4_score_state(
    st, layer: int, q, block_table, seq_lens, n_req: int, scale: float, *, kv_cache
) -> None:
    rki4_score_only(st, layer, q, block_table, seq_lens, n_req, scale)
    rki4_select_only(
        st,
        n_req,
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        layer=layer,
    )
