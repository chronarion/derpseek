"""Prompt-chunk MegaMoE, standalone (Carbon round 5 check): the stock fp8xfp4 MegaMoE call with all 384 experts in
HBM + the fused shared expert, skewed top-6 routing (Zipf-like expert popularity, every expert picked at 8K tokens),
timed at T = 3842 and 8162 tokens (CUDA events, L2 flushed), so ncu can read DRAM bytes and the L2 hit rate of one call.
   FORK=... python3 prefill_moe_bench.py [T ...]      env ZIPF (popularity exponent, default 0.8), ITERS"""
import os, shutil, sys, torch
import torch.distributed as dist
from vllm.utils.deep_gemm import _import_deep_gemm
dg = _import_deep_gemm()
FORK = os.environ.get("FORK")
root = "/tmp/drp-root"
shutil.rmtree(root, ignore_errors=True)
shutil.copytree(os.path.join(os.path.dirname(dg.__file__), "include"), os.path.join(root, "include"))
if FORK:
    for rel in ("include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh", "include/deep_gemm/scheduler/mega_moe.cuh"):
        shutil.copyfile(os.path.join(FORK, rel), os.path.join(root, rel))
dg._C.init(root)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29535")
dist.init_process_group("nccl", rank=0, world_size=1)
group = dist.group.WORLD
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmSm100MegaMoEBackend
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
H, I, K, TMAX = 5120, 2304, 6, 8192
E = int(os.environ.get("E", "384"))      # weight rows (served prompt calls: 480 = 384 experts + 96 cache rows)
NR = 384                                  # routed experts the picks land on
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
be = DeepGemmSm100MegaMoEBackend()
ws = []
for e0 in range(0, E, 64):   # build in slices to bound the transient memory
    n = min(64, E - e0)
    w13 = torch.randint(-128, 128, (n, 2 * I, H // 2), dtype=torch.int8, device=dev, generator=g)
    s13 = torch.randint(118, 124, (n, 2 * I, H // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
    w2 = torch.randint(-128, 128, (n, H, I // 2), dtype=torch.int8, device=dev, generator=g)
    s2 = torch.randint(118, 124, (n, H, I // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
    ws.append((w13, s13, w2, s2))
w13, s13, w2, s2 = (torch.cat([w[i] for w in ws]) for i in range(4))
del ws
out = be.transform_weights(w13_weight=w13, w13_weight_scale=s13, w2_weight=w2, w2_weight_scale=s2,
                           num_local_experts=E, hidden_size=H, intermediate_size=I)
del w13, s13, w2, s2
W = [out[0][0], out[0][1], out[1][0], out[1][1]]
gu = (torch.randn(2 * I, H, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
dn = (torch.randn(H, I, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
gs = torch.full((H // 128, 2 * I), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()
ds = torch.full((I // 128, H), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()
l1, l2 = dg.transform_weights_for_mega_moe((gu, gs), (dn, ds))
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=1, mma_type="fp8xfp4")
ZIPF = float(os.environ.get("ZIPF", "0.8"))
pop = 1.0 / torch.arange(1, NR + 1, device=dev).float() ** ZIPF
pop = pop[torch.randperm(NR, device=dev, generator=g)]
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)

def run(T):
    x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.multinomial(pop.expand(T, NR), K, replacement=False, generator=g).to(torch.int32)
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1)
    cnt = torch.bincount(ids.flatten().long(), minlength=NR)
    blocks = int(((cnt + 127) // 128).sum())
    sbm = int(dg.get_block_m_for_mega_moe(1, E, TMAX, T, K, "fp8xfp4"))
    y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)

    def call():
        prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T],
                               shared_x_sf=symm.shared_l1_acts_sf, shared_block_m=sbm)
        dg.fp8_fp4_mega_moe(y, (W[0], W[1]), (W[2], W[3]), symm, activation_clamp=10.0, fast_math=True,
                            shared_l1_weights=(l1[0], l1[1]), shared_l2_weights=(l2[0], l2[1]))
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(int(os.environ.get("ITERS", "10"))):
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); call(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    print(f"E={E} block_m {sbm} T={T}: experts picked {int((cnt > 0).sum())}, max tokens per expert {int(cnt.max())}, 128-row blocks "
          f"{blocks} (full would be {(T * K + 127) // 128}); prepare + MegaMoE median {ts[len(ts) // 2]:.3f} ms", flush=True)

for T in [int(a) for a in sys.argv[1:]] or [3842, 8162]:
    run(T)
