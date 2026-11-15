"""Derpseek DERPSEEK_MISS=cut, the served version: one small CUDA kernel between the router and MegaMoE (one CTA, a few
microseconds) instead of exact.cut_host_picks' tensor ops. Same rule: per call, the distinct host-row experts beyond a
window of _cut_now(T) are cut for this step, weakest first by the summed rank price of their picks, never an expert
holding a pick of rank < floor, within a per-step budget of picks; cut picks are counted in the layer's control block
(the residency manager promotes them). With dsv41 R1 the kernel, not the router, releases MegaMoE's ready flag (after
its writes), so MegaMoE never reads the router's indices before the cut. Built on first use (torch cpp extension)."""
import math
import os
import sys

import torch

_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")
_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kMaxRows = 320;
constexpr int kTopk = 6;

__global__ void cut_kernel(int64_t* idx, const float* w, int T, int D, int32_t* cnt, long long* step, int reset,
                           int budget, int now, int floor_, const float* price, uint32_t* ready) {
  __shared__ float s_val[kMaxRows];
  __shared__ int s_npk[kMaxRows], s_prot[kMaxRows], s_cut[kMaxRows];
  __shared__ int s_nprot, s_cutpk;
  __shared__ long long s_left;
  const int tid = threadIdx.x, nt = blockDim.x, P = T * kTopk;
  for (int r = tid; r < D; r += nt) s_val[r] = 0.f, s_npk[r] = 0, s_prot[r] = 0, s_cut[r] = 0;
  if (tid == 0) {
    s_nprot = 0, s_cutpk = 0;
    if (reset) *step = 0;
    const long long left = (long long)budget - *step;
    s_left = left > 0 ? left : 0;
  }
  __syncthreads();
  for (int i = tid; i < P; i += nt) {
    const long long h = idx[i];
    if (h < 0 || h >= D) continue;
    const int t = i / kTopk, k = i % kTopk;
    const float wi = w[i];
    int rank = 0;
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      const float wj = w[t * kTopk + j];
      rank += (wj > wi) || (wj == wi && j < k);
    }
    atomicAdd(&s_val[h], price[rank]);
    atomicAdd(&s_npk[h], 1);
    if (rank < floor_) atomicOr(&s_prot[h], 1);
  }
  __syncthreads();
  for (int r = tid; r < D; r += nt)
    if (s_npk[r] > 0 && s_prot[r]) atomicAdd(&s_nprot, 1);
  __syncthreads();
  const int keep = now - s_nprot > 0 ? now - s_nprot : 0;
  for (int r = tid; r < D; r += nt) {
    if (s_npk[r] == 0 || s_prot[r]) continue;
    const float v = s_val[r];
    int stronger = 0;
    long long weak_pk = 0;  // picks of this candidate and of every weaker one
    for (int q = 0; q < D; ++q) {
      if (s_npk[q] == 0 || s_prot[q]) continue;
      const float vq = s_val[q];
      const bool qs = vq > v || (vq == v && q < r);
      stronger += qs;
      if (!qs) weak_pk += s_npk[q];
    }
    if (stronger >= keep && weak_pk <= s_left) s_cut[r] = 1;
  }
  __syncthreads();
  for (int i = tid; i < P; i += nt) {
    const long long h = idx[i];
    if (h < 0 || h >= D || !s_cut[h]) continue;
    idx[i] = -1;
    atomicAdd(&cnt[h], 1);
    atomicAdd(&s_cutpk, 1);
  }
  __syncthreads();
  if (tid == 0) {
    *step += s_cutpk;
    if (ready != nullptr) asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(ready), "r"(1u) : "memory");
  }
}

void cut(int64_t idx, int64_t w, int64_t T, int64_t D, int64_t cnt, int64_t step, int64_t reset, int64_t budget,
         int64_t now, int64_t floor_, int64_t price, int64_t ready, int64_t stream) {
  cut_kernel<<<1, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int64_t*>(idx), reinterpret_cast<const float*>(w), (int)T, (int)D,
      reinterpret_cast<int32_t*>(cnt), reinterpret_cast<long long*>(step), (int)reset, (int)budget, (int)now,
      (int)floor_, reinterpret_cast<const float*>(price), reinterpret_cast<uint32_t*>(ready));
}
"""
_CPP = "void cut(int64_t idx, int64_t w, int64_t T, int64_t D, int64_t cnt, int64_t step, int64_t reset, int64_t budget, " \
       "int64_t now, int64_t floor_, int64_t price, int64_t ready, int64_t stream);"
_MOD = [None]


def mod():
    if _MOD[0] is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/vllm/torch_extensions")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
        _MOD[0] = load_inline("derpseek_cut", cpp_sources=_CPP, cuda_sources=_SRC, functions=["cut"],
                              extra_cuda_cflags=["-O3"], verbose=False)
        _LOG("cut kernel loaded")
    return _MOD[0]


def cut(st, idx, w, ready_ptr: int = 0) -> None:
    """In place on MegaMoE's [T, 6] int64 indices with their [T, 6] float32 weights (see the module docstring)."""
    from . import exact as ex
    T = idx.shape[0]
    dev = idx.device
    c = ex._CUT.get(dev)
    if c is None:
        c = ex._CUT[dev] = {"price": torch.tensor(ex._CUT_PRICE).pin_memory().to(dev, non_blocking=True),
                            "step": torch.zeros((), dtype=torch.int64, device=dev),
                            "j": torch.arange(6, device=dev)}
    budget = int(math.ceil(ex.CUT_STEP * len(ex.MISS_LAYERS) * T * 6))
    assert idx.dtype == torch.int64 and w.dtype == torch.float32 and idx.is_contiguous() and w.is_contiguous()
    assert st.D <= 320, "cut kernel: at most 320 host rows per layer"
    mod().cut(idx.data_ptr(), w.data_ptr(), T, st.D, st.ctl_buf[ex.HDR:].data_ptr(), c["step"].data_ptr(),
              int(st.idx == min(ex.MISS_LAYERS)), budget, ex._cut_now(T), ex.CUT_FLOOR, c["price"].data_ptr(),
              ready_ptr, torch.cuda.current_stream().cuda_stream)
