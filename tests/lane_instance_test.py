"""Derpseek deferred-fast, standalone (Fable round 21, plan step 2): the late lane is a second instance of the same
MegaMoE fork at kNumSMs 8 (the 4 SM pairs a 144-SM decode call leaves), with no shared expert and its own symmetric
buffer and control block (own lane counters and clone table), over the same weight tensors and host rows.
At each T in TS (NDIST distinct HBM experts per call, k = 0..4 host experts with one pick each):
  split : the main call (kNumSMs MAIN_SMS, shared expert, host picks masked to -1) plus the late call (kNumSMs LANE_SMS,
          host picks only) against the full main call with every pick: max |y_now + y_late - y_full| over max |y_full|
          (two bf16 roundings against one, so small but not zero)
  time  : the full call, the masked main call, the late call alone, and the masked main call with the late call beside
          it on a second stream (both launched together, timed to the later end)
Patched fork headers from $FORK, as tests/step_cost_test.py."""
import os, shutil, torch
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
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29535")
dist.init_process_group("nccl", rank=0, world_size=1)
group = dist.group.WORLD
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmSm100MegaMoEBackend
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
H, I, K, TMAX = 5120, 2304, 6, 256
D, NH = int(os.environ.get("D", "64")), int(os.environ.get("NH", "128"))
E = D + NH
MAIN_SMS = int(os.environ.get("MAIN_SMS", "144"))
LANE_SMS = int(os.environ.get("LANE_SMS", "8"))
LANE = int(os.environ.get("LANE", "4"))          # host-lane clusters of the main call (serving default)
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

hbm = make(NH)
hst = make(D)
host = []
for t in hst:
    h = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, pin_memory=True); h.copy_(t); host.append(h)
del hst
shw = make_shared()
torch.cuda.empty_cache()
TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4"}


class _Arr:
    def __init__(self, t, rows):
        self.__cuda_array_interface__ = {"shape": (rows,) + tuple(t.shape[1:]), "typestr": TS[t.dtype],
                                         "data": (t.data_ptr(), False),
                                         "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}


def span(t, rows):
    out = torch.as_tensor(_Arr(t, rows), device=dev)
    return out.view(t.dtype) if out.dtype != t.dtype else out


sp = [span(t, E) for t in hbm]
symm_main = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=1, mma_type="fp8xfp4")
symm_late = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=0, mma_type="fp8xfp4")


def ctl_block(lane):
    """A control block of its own (lane counters, clone table); the same host rows."""
    buf = torch.zeros(16 + E, dtype=torch.int32, device=dev)
    c = buf[:E]
    c[0] = 0x53505244; c[1] = D; c[10] = lane
    tbl = torch.zeros(160 * 4 * 128, dtype=torch.uint8, device=dev)
    c.view(torch.int64)[7] = tbl.data_ptr()
    c.view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64, device=dev)
    return c, buf, tbl


ctl_main = ctl_block(LANE)
ctl_main0 = ctl_block(0)       # DIAG: the main instance with its host lane off
ctl_late = ctl_block(int(os.environ.get("LATE_LANE", "0")))   # 0: every cluster takes host tasks in order (round 21)
DIAG = os.environ.get("DIAG", "0") == "1"


def call(x, ids, w, which, T, out=None):
    symm, ctl, sms = {"main": (symm_main, ctl_main[0], MAIN_SMS), "main0": (symm_main, ctl_main0[0], MAIN_SMS),
                      "late": (symm_late, ctl_late[0], LANE_SMS)}[which]
    prepare_megamoe_inputs(x, w, ids.clamp(min=0), symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    symm.topk_idx[:T].copy_(ids)                       # -1: no pick (as the cut kernel leaves them)
    y = out if out is not None else torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    kw = dict(shared_l1_weights=(shw[0], shw[1]), shared_l2_weights=(shw[2], shw[3])) if which != "late" else {}
    prev = dg.get_num_sms()
    dg.set_num_sms(min(sms, prev))
    try:
        dg.fp8_fp4_mega_moe(y, (sp[0], sp[1]), (sp[2], sp[3]), symm, cumulative_local_expert_recv_stats=ctl,
                            activation_clamp=10.0, fast_math=True, **kw)
    finally:
        dg.set_num_sms(prev)
    return y


def timeit(fn, n=30):
    for _ in range(5):
        fn()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); st.record()
    for _ in range(n):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1000


side = torch.cuda.Stream()


def both(x, ids_now, ids_late, w, T):
    """The masked main call and the late call launched together (the lane beside MegaMoE), joined at the end."""
    main = torch.cuda.current_stream()
    ev = torch.cuda.Event(); ev.record(main)
    side.wait_event(ev)
    call(x, ids_now, w, "main", T)
    with torch.cuda.stream(side):
        call(x, ids_late, w, "late", T)
    main.wait_stream(side)


NDIST = [int(v) for v in os.environ.get("NDIST", "15,60,100").split(",")]
for i, T in enumerate(int(t) for t in os.environ.get("TS", "4,32,64").split(",")):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1).float()
    pool = torch.randperm(NH, device=dev, generator=g)[:NDIST[i]]
    base = D + pool[(torch.arange(T, device=dev)[:, None] * K + torch.arange(K, device=dev)[None, :]) % NDIST[i]]
    for k in (0, 1, 2, 3, 4):
        if k > T:
            continue
        ids = base.clone().long()
        for j in range(k):
            ids[j, K - 1] = j                         # one pick of token j goes to host row j
        host_pick = ids < D
        ids_now = torch.where(host_pick, torch.full_like(ids, -1), ids)
        ids_late = torch.where(host_pick, ids, torch.full_like(ids, -1))
        y_full = call(x, ids, w, "main", T).float()
        y_now = call(x, ids_now, w, "main", T).float()
        y_late = call(x, ids_late, w, "late", T).float()
        err = (y_now + y_late - y_full).abs().max().item() / max(y_full.abs().max().item(), 1e-9)
        if DIAG and k:
            rel = lambda u, v: (u - v).abs().max().item() / max(v.abs().max().item(), 1e-9)
            fin = {n: bool(torch.isfinite(t).all()) for n, t in (("full", y_full), ("now", y_now), ("late", y_late))}
            amax = {n: f"{t.abs().max().item():.3g}" for n, t in (("full", y_full), ("now", y_now), ("late", y_late))}
            y_full0 = call(x, ids, w, "main0", T).float()
            y_shared = call(x, torch.full_like(ids, -1), w, "main0", T).float()
            y_late_main = call(x, ids_late, w, "main0", T).float() - y_shared
            nolate = ~host_pick.any(1)
            print(f"  DIAG T={T} host {k}: finite {fin} max {amax}; full(lane 4) vs full(lane 0) {rel(y_full, y_full0):.2e}; "
                  f"now+late vs full(lane 0) {rel(y_now + y_late, y_full0):.2e}; late vs main-instance late "
                  f"{rel(y_late, y_late_main):.2e}; late rows without a late pick max "
                  f"{y_late[nolate].abs().max().item() if nolate.any() else 0:.3g}", flush=True)
        t_full = timeit(lambda: call(x, ids, w, "main", T))
        t_now = timeit(lambda: call(x, ids_now, w, "main", T))
        t_late = timeit(lambda: call(x, ids_late, w, "late", T))
        t_both = timeit(lambda: both(x, ids_now, ids_late, w, T))
        print(f"T={T:3d} distinct {NDIST[i]:3d} host {k}: split err {err:.2e} | full {t_full:7.1f} us | main masked "
              f"{t_now:7.1f} | late alone {t_late:7.1f} | main + late beside {t_both:7.1f}", flush=True)
