// Derpseek: lossless 12-bit verify head GEMV (Carbon round 3 idea 5).
// The BF16 lm_head [V, K] is stored as one byte L = sign | 7 mantissa bits and a 4-bit code c per weight, with the
// exponent e = base + c (base a multiple of 16, so the BF16 high byte is sign | base>>1 | c>>1, the low byte c0 | mant).
// Packed offline in the mma.m16n8k16 A-fragment order: per 16-row tile and 64-column group, lane l holds 32 L bytes
// (2 x uint4: k-step s = words 2s, 2s+1, elements j0..j7) and 16 code bytes (1 x uint4: word s, byte i = c_i | c_{i+4} << 4).
// Weights outside the 16-exponent window (exceptions, ~0.016%) are packed as L = 0, c = 0, i.e. the placeholder
// 2^(base-127); after the K reduction each row adds (true weight - placeholder) x activation from a per-row list (the
// difference is exact in FP32), before the logit is rounded to BF16. The weights are reproduced bit for bit; the FP32
// result differs from a plain FP32 sum of the true products only by rounding, like a change of summation order.
// Block = one 16-row tile, 8 warps split K (10 groups each); fixed-order reduction in shared memory (deterministic).
#include <cuda_bf16.h>
#include <cstdint>

#ifndef H12_PF
#define H12_PF 4
#endif
namespace {
constexpr int kWarps = 8;
constexpr int kGroupCols = 64;   // 4 k-steps of 16
constexpr int kGPW = 10;         // groups per warp (K = 5120)

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t m) {
  // (a & m) | (b & ~m)
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(m));
  return r;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// four weights: L bytes in Lw, codes in the low nibbles of the bytes of Cw -> two bf16x2 (j0|j1, j2|j3)
__device__ __forceinline__ void dec4(uint32_t Lw, uint32_t Cw, uint32_t hconst, uint32_t& p01, uint32_t& p23) {
  const uint32_t ch = ((Cw >> 1) & 0x07070707u) | hconst;      // base>>1 | c>>1 (bit 7 clear)
  const uint32_t hb = lop3_sel(Lw, ch, 0x80808080u);          // sign | exponent high bits
  const uint32_t lb = lop3_sel(Lw, Cw << 7, 0x7F7F7F7Fu);      // c0 | mantissa
  p01 = prmt(lb, hb, 0x5140u);
  p23 = prmt(lb, hb, 0x7362u);
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#ifndef H12_MINB
#define H12_MINB 1
#endif
__global__ void __launch_bounds__(kWarps * 32, H12_MINB) head12_kernel(
    const __nv_bfloat16* __restrict__ x, int M, int K, const uint4* __restrict__ pm, const uint4* __restrict__ pn,
    uint32_t hconst, const int* __restrict__ rptr, const int* __restrict__ ecol, const float* __restrict__ edelta,
    __nv_bfloat16* __restrict__ out, int ldo, int V) {
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int ngroups = K / kGroupCols;
  const int gb = warp * kGPW;
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const size_t tbase = static_cast<size_t>(tile) * ngroups;
  const uint32_t* xr = reinterpret_cast<const uint32_t*>(x + static_cast<size_t>(g) * K);
  const bool xon = g < M;
  uint4 bm0[H12_PF], bm1[H12_PF], bn[H12_PF];
#pragma unroll
  for (int q = 0; q < H12_PF; ++q) {
    const size_t i = (tbase + gb + q) * 32 + lane;
    bm0[q] = __ldcs(pm + 2 * i); bm1[q] = __ldcs(pm + 2 * i + 1); bn[q] = __ldcs(pn + i);
  }
#pragma unroll
  for (int q = 0; q < kGPW; ++q) {
    const uint4 m0 = bm0[q % H12_PF], m1 = bm1[q % H12_PF], nn = bn[q % H12_PF];
    if (q + H12_PF < kGPW) {
      const size_t i = (tbase + gb + q + H12_PF) * 32 + lane;
      bm0[q % H12_PF] = __ldcs(pm + 2 * i); bm1[q % H12_PF] = __ldcs(pm + 2 * i + 1); bn[q % H12_PF] = __ldcs(pn + i);
    }
    const int k0 = (gb + q) * kGroupCols;
    const uint32_t mw[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    const uint32_t nw[4] = {nn.x, nn.y, nn.z, nn.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a[4];
      dec4(mw[2 * s], nw[s] & 0x0F0F0F0Fu, hconst, a[0], a[1]);
      dec4(mw[2 * s + 1], (nw[s] >> 4) & 0x0F0F0F0Fu, hconst, a[2], a[3]);
      uint32_t b0 = 0, b1 = 0;
      if (xon) {
        b0 = __ldg(xr + (k0 + s * 16) / 2 + t);
        b1 = __ldg(xr + (k0 + s * 16) / 2 + 4 + t);
      }
      mma16816(c, a, b0, b1);
    }
  }
  // reduction over the 8 warps in a fixed order
  __shared__ float red[kWarps][16][8];
  red[warp][g][2 * t] = c[0];
  red[warp][g][2 * t + 1] = c[1];
  red[warp][g + 8][2 * t] = c[2];
  red[warp][g + 8][2 * t + 1] = c[3];
  __syncthreads();
  if (threadIdx.x < 128) {
    const int r = threadIdx.x >> 3, m = threadIdx.x & 7;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += red[w][r][m];
    const int row = tile * 16 + r;
    // exceptions of this row: weight - placeholder (exact in FP32), in column order
    if (m < M && row < V) {
      const int e0 = rptr[row], e1 = rptr[row + 1];
      for (int e = e0; e < e1; ++e)
        s += edelta[e] * __bfloat162float(x[static_cast<size_t>(m) * K + ecol[e]]);
      out[static_cast<size_t>(m) * ldo + row] = __float2bfloat16_rn(s);
    }
  }
}
}  // namespace

extern "C" int head12_abi() { return 3; }

extern "C" int head12_launch(const void* x, int M, int K, const void* pm, const void* pn, int base, const int* rptr,
                             const int* ecol, const float* edelta, void* out, int ldo, int V, cudaStream_t stream) {
  if (K != kWarps * kGPW * kGroupCols || M < 1 || M > 8 || (base & 15) != 0) return -1;
  const uint32_t hb = static_cast<uint32_t>(base >> 1) & 0x78u;
  const uint32_t hconst = hb * 0x01010101u;
  head12_kernel<<<(V + 15) / 16, kWarps * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), M, K, static_cast<const uint4*>(pm), static_cast<const uint4*>(pn),
      hconst, rptr, ecol, edelta, static_cast<__nv_bfloat16*>(out), ldo, V);
  return static_cast<int>(cudaGetLastError());
}
