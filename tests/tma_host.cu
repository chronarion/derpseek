// Can TMA (cp.async.bulk.tensor) read Grace host memory on the GB300, and how fast? Streams 1 GiB in 16 KiB tiles.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
constexpr int BOX_R = 128, BOX_C = 128, TILE = BOX_R * BOX_C, STAGES = 6;
constexpr size_t ROWS = 8192, COLS = 131072, BYTES = ROWS * COLS;   // 1 GiB of uint8

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(b))); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, uint64_t* b, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(sa(dst)), "l"(m), "r"(sa(b)), "r"(c0), "r"(c1) : "memory"); }

__global__ void tma_stream(const __grid_constant__ CUtensorMap map, unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint64_t bars[STAGES];
  const int tc = COLS / BOX_C, total = (ROWS / BOX_R) * tc;
  if (threadIdx.x == 0) { for (int s = 0; s < STAGES; s++) mbar_init(&bars[s]); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  int n = blockIdx.x < total ? (total - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  unsigned long long acc = 0;
  if (threadIdx.x == 0)
    for (int s = 0; s < STAGES && s < n; s++) { int t = blockIdx.x + s * gridDim.x; mbar_expect(&bars[s], TILE); tma2d(smem + s * TILE, &map, &bars[s], (t % tc) * BOX_C, (t / tc) * BOX_R); }
  for (int i = 0; i < n; i++) {
    int s = i % STAGES; mbar_wait(&bars[s], (i / STAGES) & 1);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(smem + s * TILE);
    for (int k = threadIdx.x; k < TILE / 4; k += blockDim.x) acc += p[k];
    __syncthreads();
    if (threadIdx.x == 0 && i + STAGES < n) { int t = blockIdx.x + (i + STAGES) * gridDim.x; mbar_expect(&bars[s], TILE); tma2d(smem + s * TILE, &map, &bars[s], (t % tc) * BOX_C, (t / tc) * BOX_R); }
  }
  atomicAdd(out, acc);
}
__global__ void ldg_stream(const uint4* __restrict__ src, size_t n16, unsigned long long* out) {
  unsigned long long acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldg(src + i); acc += (unsigned long long)v.x + v.y + v.z + v.w; }
  atomicAdd(out, acc);
}
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode() {
  void* f = nullptr; cudaDriverEntryPointQueryResult q;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q));
  return (EncodeFn)f;
}
static void run(const char* name, uint8_t* buf, int nsm) {
  CUtensorMap map; cuuint64_t dims[2] = {COLS, ROWS}, strides[1] = {COLS}; cuuint32_t box[2] = {BOX_C, BOX_R}, es[2] = {1, 1};
  CUresult r = encode()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, buf, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  unsigned long long *d_out, h1 = 0, h2 = 0; CK(cudaMalloc(&d_out, 8));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); float ms_t = -1, ms_l = -1;
  int smem = STAGES * TILE; CK(cudaFuncSetAttribute(tma_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if (r == CUDA_SUCCESS) {
    for (int it = 0; it < 3; it++) {
      CK(cudaMemset(d_out, 0, 8)); CK(cudaEventRecord(a)); tma_stream<<<nsm, 256, smem>>>(map, d_out); CK(cudaEventRecord(b));
      cudaError_t e = cudaEventSynchronize(b); if (e != cudaSuccess) { printf("%-22s SMs %3d  TMA kernel failed: %s\n", name, nsm, cudaGetErrorString(e)); exit(2); }
      CK(cudaEventElapsedTime(&ms_t, a, b)); }
    CK(cudaMemcpy(&h1, d_out, 8, cudaMemcpyDeviceToHost));
  }
  for (int it = 0; it < 3; it++) {
    CK(cudaMemset(d_out, 0, 8)); CK(cudaEventRecord(a)); ldg_stream<<<nsm * 4, 256>>>((const uint4*)buf, BYTES / 16, d_out); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms_l, a, b)); }
  CK(cudaMemcpy(&h2, d_out, 8, cudaMemcpyDeviceToHost));
  printf("%-22s SMs %3d  tensor map %s  TMA %7.1f GB/s  LDG %7.1f GB/s  checksums %s\n", name, nsm,
         r == CUDA_SUCCESS ? "ok  " : "FAIL", r == CUDA_SUCCESS ? BYTES / ms_t / 1e6 : 0.0, BYTES / ms_l / 1e6,
         (r != CUDA_SUCCESS || h1 == h2) ? "match" : "DIFFER");
  CK(cudaFree(d_out));
}
int main() {
  int dev = 0, nsm = 0; CK(cudaSetDevice(dev)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  int pageable_access = 0; CK(cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, dev));
  printf("SMs %d, pageable memory access %d\n", nsm, pageable_access);
  uint8_t *dmem, *hpin, *hreg, *hpage;
  CK(cudaMalloc(&dmem, BYTES)); CK(cudaHostAlloc(&hpin, BYTES, cudaHostAllocMapped));
  hreg = (uint8_t*)aligned_alloc(1 << 21, BYTES); hpage = (uint8_t*)aligned_alloc(1 << 21, BYTES);
  for (size_t i = 0; i < BYTES; i++) { uint8_t v = (uint8_t)(i * 2654435761u >> 24); hpin[i] = v; hreg[i] = v; hpage[i] = v; }
  CK(cudaHostRegister(hreg, BYTES, cudaHostRegisterMapped));
  CK(cudaMemcpy(dmem, hpin, BYTES, cudaMemcpyHostToDevice));
  for (int s : {nsm, 32, 16}) {
    run("HBM (cudaMalloc)", dmem, s);
    run("pinned (HostAlloc)", hpin, s);
    run("registered (malloc)", hreg, s);
    run("pageable (malloc)", hpage, s);
  }
  return 0;
}
