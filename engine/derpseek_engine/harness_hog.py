"""Fable round 20's D0 for deferred-fast: run one of dsv41's harnesses (engine/harness/*.py) while DERPSEEK_HOG_SMS SMs
(default 8) are held for the whole run by a kernel spinning on a side stream, as a late lane (a second, 8-SM MegaMoE
call) would hold them, to see what the step's dense chain loses when those SMs are gone. DERPSEEK_HOG_SMS=0: no hog
(the control). Each hog CTA asks for the whole SM (1,024 threads, most of the shared memory) so nothing else shares it.
   python3 /derpseek/derpseek_engine/harness_hog.py /engine/harness/whole_step.py OUT [harness args]"""
import atexit
import os
import runpy
import sys

import torch

sys.path.insert(0, "/derpseek")
_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
// 2-CTA clusters, as a late-lane MegaMoE instance would launch: it holds whole SM pairs. In this harness MegaMoE runs on
// all 76 pairs (kNumSMs 152), so its clusters that find no free pair wait until the hog ends. Served decode calls run
// at kNumSMs 144 (dsv41 E3, DSV41_MIX_OFFPATH), which leaves 4 pairs to the FFN-side mix
__global__ void __cluster_dims__(2, 1, 1) hog(volatile int* stop) {
  extern __shared__ char smem[];
  if (threadIdx.x == 0) {
    smem[0] = 1;
    while (*stop == 0) __nanosleep(2000);
  }
  __syncthreads();
}
int launch(int64_t stop, int64_t nblk, int64_t smem, int64_t stream) {
  cudaFuncSetAttribute(hog, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  hog<<<(int)nblk, 1024, (size_t)smem, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<volatile int*>(stop));
  return (int)cudaGetLastError();
}
"""
_STATE = {}


def _start():
    n = int(os.environ.get("DERPSEEK_HOG_SMS", "8"))
    n -= n % 2
    if n <= 0 or _STATE:
        return
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/vllm/torch_extensions")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
    m = load_inline("derpseek_hog", cpp_sources="int launch(int64_t stop, int64_t nblk, int64_t smem, int64_t stream);",
                    cuda_sources=_SRC, functions=["launch"], extra_cuda_cflags=["-O3"], verbose=False)
    stop = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    s = torch.cuda.Stream()
    smem = 200 * 1024
    rc = m.launch(stop.data_ptr(), n, smem, s.cuda_stream)
    _STATE.update(m=m, stop=stop, stream=s)
    sys.stderr.write(f"DERPSEEK hog: {n} CTAs of 1,024 threads and {smem >> 10} KB shared memory spinning (rc {rc})\n")

    def _end():
        stop[0] = 1
        torch.cuda.synchronize()
        sys.stderr.write("DERPSEEK hog: released\n")

    atexit.register(_end)


def _install():
    """Hold the SMs only in the harness's timing loop (whole_step.py replays a second, fresh capture S times back to back):
    a late lane exists only inside decode-step graphs. In this harness every MegaMoE call launches one cluster per SM
    pair (kNumSMs 152) and its fair first wave waits for every gated cluster, so a hog that never ends hangs the first
    call it overlaps (both D0 runs did). Served decode MegaMoE runs at kNumSMs 144; this does not measure what a lane
    that ends, or one kept out of MegaMoE's way, costs there."""
    orig = torch.cuda.CUDAGraph.replay
    seen, count = set(), [0]

    def replay(self, _orig=orig):
        if not torch.cuda.is_current_stream_capturing():
            count[0] += 1
            if id(self) not in seen:
                seen.add(id(self))
                if not _STATE and count[0] > int(os.environ.get("DERPSEEK_HOG_AFTER", "100")):
                    torch.cuda.synchronize()
                    _start()     # a new graph after the generation runs: the timing capture
        _orig(self)

    torch.cuda.CUDAGraph.replay = replay


if __name__ == "__main__":
    target = sys.argv[1]
    _install()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")
