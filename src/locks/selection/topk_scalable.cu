#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

__device__ unsigned ordered_score(float f) {
    unsigned x = __float_as_uint(f == 0.f ? 0.f : f);
    return x ^ ((unsigned)(-(int)(x >> 31)) | 0x80000000u);
}

template<int TILE, bool FIRST, bool LAST>
__device__ __forceinline__ void locks_topk_radix_body(
    const float* score, const unsigned long long* input,
    unsigned long long* output, const int* npg, const int* nsh,
    const int* budgets, int* table, int* counts,
    long score_r, long score_h, long table_r, long table_h, long count_r,
    int mp, int input_n, int output_n, int kv, int sink, int cap)
{
    constexpr int BLOCK = 256;
    constexpr int ITEMS = TILE / BLOCK;
    const int tid = threadIdx.x, lane = tid & 31;
    const int chunk = blockIdx.x, h = blockIdx.y, r = blockIdx.z;
    const int base = chunk * TILE;
    const int group = r * kv + h;
    const int n = npg[r], hi = min(n, nsh[r]), lo = min(n, sink);
    const long sb = r * score_r + h * score_h;
    const long tb = r * table_r + h * table_h;
    __shared__ unsigned keys[TILE];
    __shared__ int hist[256];
    __shared__ int total_valid, total_gt, chosen_digit, next_k, bucket_count;
    using Scan = cub::BlockScan<int, BLOCK>;
    __shared__ typename Scan::TempStorage scan;
    if (tid == 0) { total_valid = 0; total_gt = 0; }
    int valid = 0;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        int x = tid + j * BLOCK, i = base + x;
        unsigned u = 0;
        if constexpr (FIRST) {
            if (i < mp) table[tb+i] = -1;
            if (i >= lo && i < hi) u = ordered_score(score[sb+i]);
        } else {
            if (i < input_n) u = unsigned(input[(long)group*input_n+i] >> 32);
        }
        keys[x] = u;
        valid += u != 0;
    }
    __syncthreads();
    if (valid) atomicAdd(&total_valid, valid);
    __syncthreads();
    int wanted = min(total_valid, LAST ? max(0, budgets[r]) : cap);
    unsigned prefix = 0, mask = 0;
    int k = wanted;
    if (wanted && wanted < total_valid) {
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            hist[tid] = 0;
            __syncthreads();
            int shift = 24 - 8*digit;
            #pragma unroll
            for (int j = 0; j < ITEMS; ++j) {
                unsigned u = keys[tid+j*BLOCK];
                if (u && (u & mask) == prefix) {
                    unsigned bin = (u >> shift) & 255;
                    unsigned peers = __match_any_sync(__activemask(), bin);
                    if (lane == __ffs(peers)-1) atomicAdd(hist+bin, __popc(peers));
                }
            }
            __syncthreads();
            if (tid < 32) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane*8+j];
                int suffix = seg;
                #pragma unroll
                for (int d = 1; d < 32; d *= 2) {
                    int up = __shfl_down_sync(0xffffffff, suffix, d);
                    if (lane+d < 32) suffix += up;
                }
                int above = suffix-seg;
                if (above < k && k <= suffix) {
                    int acc = above;
                    for (int j = lane*8+7; j >= lane*8; --j) {
                        int c = hist[j];
                        if (acc+c >= k) {
                            chosen_digit = j; next_k = k-acc; bucket_count = c;
                            break;
                        }
                        acc += c;
                    }
                }
            }
            __syncthreads();
            prefix |= unsigned(chosen_digit) << shift;
            mask |= 255u << shift;
            k = next_k;
            if (k == bucket_count) break;
        }
    }
    unsigned tau = (wanted == total_valid) ? 1 : prefix;
    int greater = 0, equal = 0;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        unsigned u = keys[tid*ITEMS+j];
        greater += u && u > tau;
        equal += u && u == tau;
    }
    if (greater) atomicAdd(&total_gt, greater);
    int eq_offset;
    Scan(scan).ExclusiveSum(equal, eq_offset);
    __syncthreads();
    int keep[ITEMS], kept = 0;
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        unsigned u = keys[tid*ITEMS+j];
        bool eq = u && u == tau;
        keep[j] = wanted && u && (u > tau || (eq && eq_offset < wanted-total_gt));
        eq_offset += eq;
        kept += keep[j];
    }
    int offset;
    Scan(scan).ExclusiveSum(kept, offset);
    if constexpr (!LAST) {
        for (int i = tid; i < cap; i += BLOCK)
            output[(long)group*output_n+chunk*cap+i] = 0;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        if (keep[j]) {
            int x = tid*ITEMS+j, i = base+x;
            unsigned long long packed;
            if constexpr (FIRST)
                packed = (static_cast<unsigned long long>(keys[x]) << 32) | unsigned(~i);
            else packed = input[(long)group*input_n+i];
            if constexpr (LAST) table[tb+lo+offset] = int(~unsigned(packed));
            else output[(long)group*output_n+chunk*cap+offset] = packed;
            ++offset;
        }
    }
    if constexpr (LAST) {
        for (int i = tid; i < lo; i += BLOCK) table[tb+i] = i;
        const int recent_start = max(lo, hi);
        for (int i = recent_start+tid; i < n; i += BLOCK)
            table[tb+lo+wanted+i-recent_start] = i;
        if (tid == 0) counts[r*count_r+h] = lo+wanted+n-recent_start;
    }
}

template<int TILE, bool FIRST, bool LAST>
__global__ void locks_topk_radix_reduce(
    const float* score, const unsigned long long* input,
    unsigned long long* output, const int* npg, const int* nsh,
    const int* budgets, int* table, int* counts,
    long score_r, long score_h, long table_r, long table_h, long count_r,
    int mp, int input_n, int output_n, int kv, int sink, int cap)
{
    if constexpr (TILE == 4096 && FIRST && LAST) {
        if (kv == 8 && gridDim.z == 1 && cap <= 126) {
            int n = npg[blockIdx.z];
            if (n <= 1024) {
                locks_topk_radix_body<1024,true,true>(score,input,output,npg,nsh,budgets,table,counts,score_r,score_h,table_r,table_h,count_r,mp,input_n,output_n,kv,sink,cap);
                long offset=blockIdx.z*table_r+blockIdx.y*table_h;
                for (int i=1024+threadIdx.x;i<mp;i+=256) table[offset+i]=-1;
                return;
            }
            if (n <= 2048) {
                locks_topk_radix_body<2048,true,true>(score,input,output,npg,nsh,budgets,table,counts,score_r,score_h,table_r,table_h,count_r,mp,input_n,output_n,kv,sink,cap);
                long offset=blockIdx.z*table_r+blockIdx.y*table_h;
                for (int i=2048+threadIdx.x;i<mp;i+=256) table[offset+i]=-1;
                return;
            }
        }
    }
    locks_topk_radix_body<TILE,FIRST,LAST>(score,input,output,npg,nsh,budgets,table,counts,score_r,score_h,table_r,table_h,count_r,mp,input_n,output_n,kv,sink,cap);
}

template<int TILE, bool FIRST, bool LAST>
void launch(torch::Tensor score, torch::Tensor input, torch::Tensor output,
    torch::Tensor npg, torch::Tensor nsh, torch::Tensor budgets,
    torch::Tensor table, torch::Tensor counts, int nreq, int sink, int cap, int input_n)
{
    int nkv = score.size(1), mp = score.size(2);
    int chunks = (input_n+TILE-1)/TILE;
    dim3 grid(chunks, nkv, nreq);
    locks_topk_radix_reduce<TILE,FIRST,LAST><<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(
        score.data_ptr<float>(), reinterpret_cast<const unsigned long long*>(input.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(output.data_ptr<int64_t>()),
        npg.data_ptr<int>(), nsh.data_ptr<int>(), budgets.data_ptr<int>(),
        table.data_ptr<int>(), counts.data_ptr<int>(), score.stride(0),score.stride(1),
        table.stride(0),table.stride(1),counts.stride(0),mp,input_n,chunks*cap,nkv,sink,cap);
}

template<int TILE>
void dispatch(torch::Tensor score, torch::Tensor input, torch::Tensor output,
    torch::Tensor npg, torch::Tensor nsh, torch::Tensor budgets,
    torch::Tensor table, torch::Tensor counts, int nreq, int sink, int cap, int input_n,
    bool first, bool last)
{
    if (first && last) launch<TILE,true,true>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n);
    else if (first) launch<TILE,true,false>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n);
    else if (last) launch<TILE,false,true>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n);
    else launch<TILE,false,false>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n);
}

void reduce(torch::Tensor score, torch::Tensor input, torch::Tensor output,
    torch::Tensor npg, torch::Tensor nsh, torch::Tensor budgets,
    torch::Tensor table, torch::Tensor counts, int64_t nreq, int64_t sink,
    int64_t cap, int64_t input_n, int64_t tile, bool first, bool last)
{
    TORCH_CHECK(score.is_cuda() && score.scalar_type()==at::kFloat && score.stride(2)==1);
    TORCH_CHECK(input.scalar_type()==at::kLong && output.scalar_type()==at::kLong);
    TORCH_CHECK(npg.scalar_type()==at::kInt && nsh.scalar_type()==at::kInt && budgets.scalar_type()==at::kInt);
    TORCH_CHECK(table.scalar_type()==at::kInt && counts.scalar_type()==at::kInt && table.stride(2)==1);
    TORCH_CHECK(cap >= 1 && cap < tile && nreq <= score.size(0));
    TORCH_CHECK(!last || input_n <= tile);
    TORCH_CHECK(first || input.numel() >= nreq*score.size(1)*input_n);
    TORCH_CHECK(last || output.numel() >= nreq*score.size(1)*((input_n+tile-1)/tile)*cap);
    switch (tile) {
        case 1024: dispatch<1024>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n,first,last); break;
        case 2048: dispatch<2048>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n,first,last); break;
        case 4096: dispatch<4096>(score,input,output,npg,nsh,budgets,table,counts,nreq,sink,cap,input_n,first,last); break;
        default: TORCH_CHECK(false, "Unsupported tile");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void locks_topk_streaming(
    const float* score, const int* npg, const int* nsh, const int* budgets,
    int* table, int* counts, long sr, long sh, long tr, long th, long cr, int mp, int sink)
{
    const int r=blockIdx.x, h=blockIdx.y, tid=threadIdx.x, lane=tid&31;
    const int n=npg[r], lo=min(n,sink), hi=min(n,nsh[r]);
    const int wanted=min(max(0,budgets[r]),max(0,hi-lo));
    const long sb=r*sr+h*sh, tb=r*tr+h*th;
    __shared__ int hist[256], digit_selected, next_k, bucket_count, gt, eq_seen, out_seen;
    using Scan=cub::BlockScan<int,256>;
    __shared__ typename Scan::TempStorage scan;
    for (int i=tid;i<mp;i+=256) table[tb+i]=-1;
    if (tid==0) {gt=0;eq_seen=0;out_seen=0;}
    unsigned prefix=0,mask=0;
    int k=wanted;
    if (wanted && wanted<hi-lo) {
        for (int digit=0;digit<4;++digit) {
            hist[tid]=0;
            __syncthreads();
            int shift=24-8*digit;
            for (int i=lo+tid;i<hi;i+=256) {
                unsigned u=ordered_score(score[sb+i]);
                if ((u&mask)==prefix) {
                    unsigned bin=(u>>shift)&255;
                    unsigned peers=__match_any_sync(__activemask(),bin);
                    if (lane==__ffs(peers)-1) atomicAdd(hist+bin,__popc(peers));
                }
            }
            __syncthreads();
            if (tid<32) {
                int seg=0;
                for(int j=0;j<8;++j) seg+=hist[lane*8+j];
                int suffix=seg;
                for(int d=1;d<32;d*=2) {
                    int up=__shfl_down_sync(0xffffffff,suffix,d);
                    if(lane+d<32) suffix+=up;
                }
                int above=suffix-seg;
                if(above<k && k<=suffix) {
                    for(int j=lane*8+7;j>=lane*8;--j) {
                        int c=hist[j];
                        if(above+c>=k) {digit_selected=j;next_k=k-above;bucket_count=c;break;}
                        above+=c;
                    }
                }
            }
            __syncthreads();
            prefix|=unsigned(digit_selected)<<shift;
            mask|=255u<<shift;
            k=next_k;
            if(k==bucket_count) break;
        }
    }
    unsigned tau=wanted==hi-lo ? 1 : prefix;
    int local_gt=0;
    for(int i=lo+tid;i<hi;i+=256) local_gt+=ordered_score(score[sb+i])>tau;
    if(local_gt) atomicAdd(&gt,local_gt);
    __syncthreads();
    for(int base=0;base<n;base+=256) {
        int i=base+tid;
        bool candidate=i>=lo && i<hi;
        unsigned u=candidate ? ordered_score(score[sb+i]) : 0;
        int eq=candidate && u==tau, eq_offset, eq_total;
        Scan(scan).ExclusiveSum(eq,eq_offset,eq_total);
        __syncthreads();
        int keep=i<n && (!candidate || (wanted && (u>tau || (eq && eq_seen+eq_offset<wanted-gt))));
        int offset,total;
        Scan(scan).ExclusiveSum(keep,offset,total);
        __syncthreads();
        if(keep) table[tb+out_seen+offset]=i;
        __syncthreads();
        if(tid==0) {eq_seen+=eq_total;out_seen+=total;}
        __syncthreads();
    }
    if(tid==0) counts[r*cr+h]=out_seen;
}

void streaming(torch::Tensor score,torch::Tensor npg,torch::Tensor nsh,torch::Tensor budgets,
    torch::Tensor table,torch::Tensor counts,int64_t nreq,int64_t sink)
{
    dim3 grid(nreq,score.size(1));
    locks_topk_streaming<<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(score.data_ptr<float>(),
        npg.data_ptr<int>(),nsh.data_ptr<int>(),budgets.data_ptr<int>(),table.data_ptr<int>(),counts.data_ptr<int>(),
        score.stride(0),score.stride(1),table.stride(0),table.stride(1),counts.stride(0),score.size(2),sink);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("reduce", &reduce); m.def("streaming", &streaming); }
