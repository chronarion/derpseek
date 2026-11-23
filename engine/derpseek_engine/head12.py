"""Lossless 12-bit verify head (Carbon round 3 idea 5): the target's BF16 lm_head stored as one byte (sign + 7 mantissa
bits) and a 4-bit exponent code per weight, 1.5 bytes instead of 2, read by a CUDA GEMV (overlay/head12/head12.cu)
that decodes straight into tensor-core fragments.

The exponent is base + a 4-bit code (base a multiple of 16: 112 for this head, so a code is the exponent's low nibble).
Weights outside that 16-exponent window (~0.016% of the head) decode to a placeholder (2^(base-127)); each row then adds
(true weight - placeholder) x activation from a per-row list after the K reduction, before the logit is rounded to BF16. Every weight is
reproduced bit for bit; the dot accumulates in FP32 as in the stock head (only the FP32 summation order differs: the
exact tier's summation-order class).

DERPSEEK_HEAD12=1 swaps model.compute_logits for batches of up to 64 rows (larger ones keep the stock head)."""
import ctypes
import os
import sys

import torch

_LOG = lambda m: sys.stderr.write(f"DERPSEEK head12: {m}\n")
_LIB = os.environ.get("DERPSEEK_HEAD12_LIB", "/derpseek/build/head12/libhead12.so")
MAX_M = 8


def _lane_map(dev):
    """[32 lanes, 4 k-steps, 8 elements] -> (row in the 16-row tile, column in the 64-column group), mma A order"""
    lane = torch.arange(32, device=dev)[:, None, None]
    s = torch.arange(4, device=dev)[None, :, None]
    j = torch.arange(8, device=dev)[None, None, :]
    g, t = lane // 4, lane % 4
    row = g + 8 * ((j // 2) % 2)
    col = 16 * s + 2 * t + (j % 2) + 8 * (j // 4)
    return row.expand(32, 4, 8), col.expand(32, 4, 8)


class Head12:
    def __init__(self, weight: torch.Tensor):
        assert weight.dtype == torch.bfloat16 and weight.dim() == 2
        V, K = weight.shape
        assert V % 16 == 0 and K == 5120
        dev = weight.device
        tiles, groups = V // 16, K // 64
        u = weight.detach().contiguous().view(torch.int16)
        e = ((u >> 7) & 0xFF).to(torch.int16)
        hist = torch.bincount(e.flatten().to(torch.int32), minlength=256)
        base = 16 * int(torch.stack([hist[b:b + 16].sum() for b in range(0, 256, 16)]).argmax())
        code = e - base
        exc = (code < 0) | (code > 15)
        code = torch.where(exc, torch.zeros_like(code), code).to(torch.uint8)
        mant = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
        mant = torch.where(exc, torch.zeros_like(mant), mant)
        del e
        R, C = _lane_map(dev)
        pm = torch.empty(tiles, groups, 32, 32, dtype=torch.uint8, device=dev)
        pn = torch.empty(tiles, groups, 32, 16, dtype=torch.uint8, device=dev)
        step = 512
        for t0 in range(0, tiles, step):
            t1 = min(tiles, t0 + step)
            sl = slice(t0 * 16, t1 * 16)
            view = lambda a: a[sl].view(t1 - t0, 16, groups, 64).permute(0, 2, 1, 3)[:, :, R, C]   # [T, G, 32, 4, 8]
            pm[t0:t1] = view(mant).reshape(t1 - t0, groups, 32, 32)
            cc = view(code).reshape(t1 - t0, groups, 32, 4, 2, 4)
            pn[t0:t1] = (cc[..., 0, :] | (cc[..., 1, :] << 4)).reshape(t1 - t0, groups, 32, 16)
        rows, cols = exc.nonzero(as_tuple=True)                    # row-major: sorted by row, then column
        place = 2.0 ** (base - 127)                                # L = 0, c = 0 decodes to +2^(base-127)
        delta = weight.detach()[rows, cols].float() - place        # exact in FP32
        counts = torch.bincount(rows, minlength=V)
        rptr = torch.zeros(V + 1, dtype=torch.int32, device=dev)
        rptr[1:] = counts.cumsum(0).to(torch.int32)
        self.pm, self.pn = pm, pn
        self.rptr, self.ecol, self.edelta = rptr, cols.to(torch.int32).contiguous(), delta.contiguous()
        self.base, self.V, self.K = base, V, K
        self.n_exc = int(rows.numel())
        self.bytes = pm.numel() + pn.numel() + rptr.numel() * 4 + self.n_exc * 8
        del u, code, exc, mant
        self.lib = ctypes.CDLL(_LIB)
        assert self.lib.head12_abi() == 3
        self.lib.head12_launch.restype = ctypes.c_int
        self.lib.head12_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p]

    def __call__(self, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
        M = x.shape[0]
        assert M <= MAX_M
        x = x.to(torch.bfloat16).contiguous()
        out = torch.empty(M, self.V, dtype=torch.bfloat16, device=x.device)
        if M == 0:
            return out
        rc = self.lib.head12_launch(x.data_ptr(), M, self.K, self.pm.data_ptr(), self.pn.data_ptr(), self.base,
                                    self.rptr.data_ptr(), self.ecol.data_ptr(), self.edelta.data_ptr(), out.data_ptr(),
                                    out.stride(0), self.V,
                                    torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"head12_launch failed: {rc}")
        return out if out_dtype == torch.bfloat16 else out.to(out_dtype)


def install(runner):
    """after load_model: build the packed head and route decode-sized compute_logits through it"""
    lm = getattr(runner.model, "language_model", runner.model)
    w = lm.lm_head.weight
    head = Head12(w)
    lp = lm.logits_processor
    vocab = getattr(lp, "org_vocab_size", None) or getattr(lp, "vocab_size", head.V)
    orig = lm.compute_logits

    def compute_logits(hidden_states):
        if hidden_states.shape[0] <= MAX_M:
            return head(hidden_states)[:, :vocab]
        return orig(hidden_states)

    lm.compute_logits = compute_logits
    _LOG(f"on: [{head.V}, {head.K}] exponent window {head.base}..{head.base + 15}, {head.n_exc} exceptions, "
         f"{head.bytes / 2**20:.0f} MiB packed (BF16 {w.numel() * 2 / 2**20:.0f} MiB, kept for batches > {MAX_M})")
    return head
