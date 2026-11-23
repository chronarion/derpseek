"""(b) GEMM1 + stage fusion, first measurement: on real layer weights, the production pre-attention chain (FlashInfer
LL GEMM for wq_a/wkv, the W3 qkv_kt stage, FlashInfer LL GEMM for wq_b) against W3's own fused chain (K1 wqkv_kt GEMM
with in-task quantization -> fp32 split partials, K2 qkv reduce + norms + MXFP8 q + RoPE/SWA insert, K3 wq_b GEMM), as
CUDA graphs with an L2 flush before every replay (cold weights, as in a served step). Per-kernel profiles too."""
import sys

import torch

sys.path.insert(0, "/w/src")
from w3 import timing  # noqa: E402
from w3.fused import FusedChain  # noqa: E402
from w3.preattn import PreattnChain  # noqa: E402
from w3.state import cos_sin_slice, make_state  # noqa: E402
from w3.weights import load_layer  # noqa: E402

dev = torch.device("cuda")
fl = timing.Flusher(dev)
for layer in (3, 10, 30):
    w = load_layer(layer, dev)
    for T in (4, 6):
        st = make_state(T, T, 2048, w.compress_ratio, dev)
        cs = cos_sin_slice(w.compress_ratio, 1 << 20, dev) if False else cos_sin_slice(w.compress_ratio, 8192, dev)
        x = (torch.randn(T, 5120, device=dev) * 0.5).to(torch.bfloat16)
        res = {}
        for name, mk in (("production", lambda: PreattnChain(w, cs, T, kt_mode="side")),
                         ("w3-fused", lambda: FusedChain(w, cs, T))):
            ch = mk()
            g, _ = timing.capture(lambda: ch.run(x, st))
            t = timing.time_graph(g, fl, iters=200)
            prof = timing.profile_graph(g, fl, reps=15)
            res[name] = (t, prof)
        print(f"layer {layer} (ratio {w.compress_ratio}) T={T}: production {res['production'][0]['median']:.1f} us, "
              f"W3 fused chain {res['w3-fused'][0]['median']:.1f} us (graph replay, L2 flushed)")
        for name in res:
            print(f"  {name}:\n" + timing.fmt_profile(res[name][1]))
        sys.stdout.flush()
