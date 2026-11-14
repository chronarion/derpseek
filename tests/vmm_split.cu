// One virtual range, first half backed by HBM pages, second half by Grace (host NUMA) pages: does TMA stream it?
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
#define CD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); printf("DRV %s line %d: %s\n", #x, __LINE__, s_); exit(1);} } while (0)
constexpr int BOX_R = 128, BOX_C = 128, TILE = BOX_R * BOX_C, STAGES = 6;
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(b))); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) { asm volatile("{\n.reg .pred p;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, uint64_t* b, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" :: "r"(sa(dst)), "l"(m), "r"(sa(b)), "r"(c0), "r"(c1) : "memory"); }
// streams rows [r0, r1) of a [rows, cols] uint8 tensor
__global__ void tma_rows(const __grid_constant__ CUtensorMap map, int cols, int r0, int r1, unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem[]; __shared__ uint64_t bars[STAGES];
  const int tc = cols / BOX_C, total = ((r1 - r0) / BOX_R) * tc;
  if (threadIdx.x == 0) { for (int s = 0; s < STAGES; s++) mbar_init(&bars[s]); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  int n = blockIdx.x < total ? (total - blockIdx.x + gridDim.x - 1) / gridDim.x : 0; unsigned long long acc = 0;
  if (threadIdx.x == 0) for (int s = 0; s < STAGES && s < n; s++) { int t = blockIdx.x + s * gridDim.x; mbar_expect(&bars[s], TILE); tma2d(smem + s * TILE, &map, &bars[s], (t % tc) * BOX_C, r0 + (t / tc) * BOX_R); }
  for (int i = 0; i < n; i++) { int s = i % STAGES; mbar_wait(&bars[s], (i / STAGES) & 1);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(smem + s * TILE); for (int k = threadIdx.x; k < TILE / 4; k += blockDim.x) acc += p[k];
    __syncthreads();
    if (threadIdx.x == 0 && i + STAGES < n) { int t = blockIdx.x + (i + STAGES) * gridDim.x; mbar_expect(&bars[s], TILE); tma2d(smem + s * TILE, &map, &bars[s], (t % tc) * BOX_C, r0 + (t / tc) * BOX_R); } }
  atomicAdd(out, acc);
}
int main() {
  CK(cudaSetDevice(0)); CK(cudaFree(0)); CUdevice dev; CD(cuDeviceGet(&dev, 0));
  CUmemAllocationProp pd = {}; pd.type = CU_MEM_ALLOCATION_TYPE_PINNED; pd.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pd.location.id = dev;
  CUmemAllocationProp ph = {}; ph.type = CU_MEM_ALLOCATION_TYPE_PINNED; ph.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; ph.location.id = 0;
  size_t gd = 0, gh = 0; CD(cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM)); CD(cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  printf("granularity: device %zu KiB, host NUMA %zu KiB\n", gd >> 10, gh >> 10);
  const size_t half = 512ull << 20, total = 2 * half; const int cols = 131072, rows = (int)(total / cols), hrow = (int)(half / cols);
  CUdeviceptr va; CD(cuMemAddressReserve(&va, total, 0, 0, 0));
  CUmemGenericAllocationHandle hd, hh; CD(cuMemCreate(&hd, half, &pd, 0)); CD(cuMemCreate(&hh, half, &ph, 0));
  CD(cuMemMap(va, half, 0, hd, 0)); CD(cuMemMap(va + half, half, 0, hh, 0));
  CUmemAccessDesc acc = {}; acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = dev; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CD(cuMemSetAccess(va, total, &acc, 1));
  // fill through the device (both halves), pattern by byte index
  uint8_t* h = (uint8_t*)malloc(total); for (size_t i = 0; i < total; i++) h[i] = (uint8_t)(i * 2654435761u >> 24);
  CK(cudaMemcpy((void*)va, h, total, cudaMemcpyHostToDevice));
  CUtensorMap map; cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, str[1] = {(cuuint64_t)cols}; cuuint32_t box[2] = {BOX_C, BOX_R}, es[2] = {1, 1};
  CD(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)va, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  unsigned long long* d_out; CK(cudaMalloc(&d_out, 8)); CK(cudaFuncSetAttribute(tma_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * TILE));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  struct { const char* name; int r0, r1; } parts[] = {{"HBM half", 0, hrow}, {"host-NUMA half", hrow, rows}, {"whole range", 0, rows}};
  for (auto& p : parts) for (int nsm : {152, 16}) {
    float ms = 0; unsigned long long got = 0;
    for (int it = 0; it < 3; it++) { CK(cudaMemset(d_out, 0, 8)); CK(cudaEventRecord(a)); tma_rows<<<nsm, 256, STAGES * TILE>>>(map, cols, p.r0, p.r1, d_out); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b)); }
    CK(cudaMemcpy(&got, d_out, 8, cudaMemcpyDeviceToHost));
    unsigned long long want = 0; for (size_t i = (size_t)p.r0 * cols; i < (size_t)p.r1 * cols; i += 4) want += *(uint32_t*)(h + i);
    printf("%-16s SMs %3d  %8.1f GB/s  checksum %s\n", p.name, nsm, (double)(p.r1 - p.r0) * cols / ms / 1e6, got == want ? "ok" : "WRONG");
  }
  // a row copy from the host half into the HBM half (promotion), then re-check
  CK(cudaMemcpy((void*)va, (void*)(va + half), 64ull << 20, cudaMemcpyDeviceToDevice));
  CK(cudaEventRecord(a)); CK(cudaMemcpy((void*)va, (void*)(va + half), 256ull << 20, cudaMemcpyDeviceToDevice)); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); printf("promotion copy host->HBM within the range: %.1f GB/s\n", (256ull << 20) / ms / 1e6);
  return 0;
}
