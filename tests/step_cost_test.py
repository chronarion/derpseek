"""Where does exact mode's step time go? MegaMoE call time at serving shapes:
  stock   : NH HBM rows only (dsv41-like call, no host tier)
  split0  : D host rows first + the same NH HBM rows, picks only to HBM rows (host tier present, no misses)
  split k : the same with k distinct host-row experts picked (one pick each)
Random FP4 weights; routing uniform over the HBM rows (distinct experts ~21 at T=4, ~83 at T=32 with NH=96).
Served shapes (Fable round 21, 0b): NDIST=15,60,100 (per T in TS: exactly that many distinct HBM experts per call),
SHARED=1 (the fused shared expert, as served), NSMS=144 (kNumSMs of served decode calls), LANE=4 (host-lane clusters)."""
import os, shutil, sys, torch
import torch.distributed as dist
from vllm.utils.deep_gemm import _import_deep_gemm
dg = _import_deep_gemm()
FORK = os.environ["FORK"]
root = "/tmp/drp-root"
shutil.rmtree(root, ignore_errors=True)
shutil.copytree(os.path.join(os.path.dirname(dg.__file__), "include"), os.path.join(root, "include"))
for rel in ("include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh", "include/deep_gemm/scheduler/mega_moe.cuh"):
    shutil.copyfile(os.path.join(FORK, rel), os.path.join(root, rel))
dg._C.init(root)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29534")
dist.init_process_group("nccl", rank=0, world_size=1)
group = dist.group.WORLD
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmSm100MegaMoEBackend
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
H, I, K, TMAX = 5120, 2304, 6, 256
D, NH = int(os.environ.get("D", "64")), int(os.environ.get("NH", "96"))
E = D + NH
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
be = DeepGemmSm100MegaMoEBackend()
def make(n):
    w13 = torch.randint(-128, 128, (n, 2 * I, H // 2), dtype=torch.int8, device=dev, generator=g)
    s13 = torch.randint(118, 124, (n, 2 * I, H // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
    w2 = torch.randint(-128, 128, (n, H, I // 2), dtype=torch.int8, device=dev, generator=g)
    s2 = torch.randint(118, 124, (n, H, I // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
    out = be.transform_weights(w13_weight=w13, w13_weight_scale=s13, w2_weight=w2, w2_weight_scale=s2,
                               num_local_experts=n, hidden_size=H, intermediate_size=I)
    del w13, s13, w2, s2
    return [out[0][0], out[0][1], out[1][0], out[1][1]]

def make_shared():
    """The fused shared expert as vLLM builds it (transform_shared_expert_weights): FP8 e4m3 gate_up [2I, H] and down
    [H, I], one UE8M0 scale per 32 weights of a row packed four to an int32 (unit scales here), then DeepGEMM's
    transform_weights_for_mega_moe. Small values: random bytes would include e4m3's NaN codes."""
    gu = (torch.randn(2 * I, H, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    dn = (torch.randn(H, I, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    # MN-major scale views (stride(-2) == 1), as DeepGEMM's TMA layout wants them
    gs = torch.full((H // 128, 2 * I), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()
    ds = torch.full((I // 128, H), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()
    l1, l2 = dg.transform_weights_for_mega_moe((gu, gs), (dn, ds))
    return [l1[0], l1[1], l2[0], l2[1]]

hbm = make(NH)                     # the HBM rows (stock call uses exactly these)
hst = make(D)                      # host rows: transformed on the GPU, then moved to pinned memory
host = []
for t in hst:
    h = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, pin_memory=True); h.copy_(t); host.append(h)
del hst; torch.cuda.empty_cache()
TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4"}
class _Arr:
    def __init__(self, t, rows):
        self.__cuda_array_interface__ = {"shape": (rows,) + tuple(t.shape[1:]), "typestr": TS[t.dtype], "data": (t.data_ptr(), False),
                                         "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}
def span(t, rows):
    out = torch.as_tensor(_Arr(t, rows), device=dev)
    return out.view(t.dtype) if out.dtype != t.dtype else out
sp = [span(t, E) for t in hbm]
SHARED = os.environ.get("SHARED", "0") == "1"
NSMS = int(os.environ.get("NSMS", "0"))
shw = make_shared() if SHARED else None
symm_s = dg.get_symm_buffer_for_mega_moe(group, NH, TMAX, K, H, I, num_shared_experts=int(SHARED), mma_type="fp8xfp4")
symm_x = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=int(SHARED), mma_type="fp8xfp4")
ctl_buf = torch.zeros(16 + E, dtype=torch.int32, device=dev)
ctl = ctl_buf[:E]
ctl[0] = 0x53505244; ctl[1] = D; ctl[10] = int(os.environ.get("LANE", "0"))
desc_tbl = torch.zeros(160 * 4 * 128, dtype=torch.uint8, device=dev)   # per-layer clone table (ctl words 14-15)
ctl.view(torch.int64)[7] = desc_tbl.data_ptr()
ctl.view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64, device=dev)
def call(x, ids, w, mode, T):
    symm = symm_s if mode == "stock" else symm_x
    prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    kw = dict(shared_l1_weights=(shw[0], shw[1]), shared_l2_weights=(shw[2], shw[3])) if SHARED else {}
    prev = dg.get_num_sms()
    if NSMS:
        dg.set_num_sms(min(NSMS, prev))
    try:
        if mode == "stock":
            dg.fp8_fp4_mega_moe(y, (hbm[0], hbm[1]), (hbm[2], hbm[3]), symm, activation_clamp=10.0, fast_math=True, **kw)
        else:
            dg.fp8_fp4_mega_moe(y, (sp[0], sp[1]), (sp[2], sp[3]), symm, cumulative_local_expert_recv_stats=ctl,
                                activation_clamp=10.0, fast_math=True, **kw)
    finally:
        dg.set_num_sms(prev)
    return y
def timeit(fn, n=30):
    for _ in range(5): fn()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1000
NDIST = [int(v) for v in os.environ.get("NDIST", "").split(",") if v]
for i, T in enumerate(int(t) for t in os.environ.get("TS", "4,8,32").split(",")):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1).float()
    if NDIST:   # exactly NDIST[i] distinct HBM experts: picks walk a random pool, distinct within each row
        pool = torch.randperm(NH, device=dev, generator=g)[:NDIST[i]]
        base = pool[(torch.arange(T, device=dev)[:, None] * K + torch.arange(K, device=dev)[None, :]) % NDIST[i]]
    else:
        base = torch.stack([torch.randperm(NH, device=dev, generator=g)[:K] for _ in range(T)])   # HBM row indices 0..NH-1
    nd = len(set(base.flatten().tolist()))
    t_stock = timeit(lambda: call(x, base.long(), w, "stock", T))
    line = f"T={T:3d} ({nd:3d} distinct HBM experts): stock {t_stock:7.1f} us"
    for k in (0, 1, 2, 3, 4, 6):
        ids = (base + D).clone()
        for j in range(min(k, T)):
            ids[j, K - 1] = j                          # one pick of token j goes to host row j
        if k > T:
            continue
        line += f" | split{k} {timeit(lambda: call(x, ids.long(), w, 'split', T)):7.1f}"
    print(line, flush=True)
