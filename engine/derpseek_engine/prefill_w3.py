"""Prompt chunks skip dsv41's W3 fused q/kv stage (DERPSEEK_W3_MAX_T, default 256 rows).

W3's stage kernel is built for decode-sized steps; on an 8K-token prompt chunk it takes ~290 us per layer (two blocks
per token) for ~50 MB of work. Above DERPSEEK_W3_MAX_T rows the attention takes vLLM's own q/kv split, norms and
fused RoPE + KV insert instead (the path W3 replaces; W3 stays on for decode and graph-captured steps).
DERPSEEK_W3_MAX_T=0 keeps W3 for every size."""
import importlib
import os
import sys

MAX_T = int(os.environ.get("DERPSEEK_W3_MAX_T", "256") or 0)
_LOG = lambda m: sys.stderr.write(f"DERPSEEK prefill_w3: {m}\n")


def install():
    if MAX_T <= 0:
        return
    try:
        integ = importlib.import_module("w3preattn.integrate")
    except ImportError as e:
        _LOG(f"w3preattn.integrate not importable ({e}); nothing to patch")
        return
    if getattr(integ, "_derpseek_w3_max_t", None) is not None:
        return
    orig = integ._eligible

    def _eligible(self, qr_kv):
        return qr_kv.shape[0] <= MAX_T and orig(self, qr_kv)

    integ._eligible = _eligible
    integ._derpseek_w3_max_t = MAX_T
    _LOG(f"W3 fused q/kv stage only for steps of <= {MAX_T} rows; prompt chunks take vLLM's own path")
