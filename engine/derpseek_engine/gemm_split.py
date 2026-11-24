"""Large prompt chunks: FlashInfer CuTe-DSL MXFP8 dense GEMMs above DERPSEEK_GEMM_SPLIT rows run as equal row pieces
of at most that many rows (multiples of 128), each on the tactic FlashInfer picks for <= 8192 rows.

Measured (11-23, exact, 12,258-token chunks): the three dense GEMMs of a layer took ~[115, 490, 510] us against
~[56, 240, 220] us at 8,162 rows, i.e. ~2.05-2.3x the time for 1.5x the rows. Rows are independent, so each output row
is the same FP32 sum over the same products (the exact tier's summation-order class at most); the slicing itself is
bit-identical (one 8192-row call vs two 4096-row pieces, tests in the scratch notes). The swizzled MXFP8 activation
scales (FlashInfer F8_128x4) are sliced in whole 128-row tiles. DERPSEEK_GEMM_SPLIT=0 turns it off."""
import dataclasses
import importlib
import os
import sys

import torch

SPLIT = int(os.environ.get("DERPSEEK_GEMM_SPLIT", "0") or 0)
_LOG = lambda m: sys.stderr.write(f"DERPSEEK gemm_split: {m}\n")


def _scale_rows(scale, K, r0, r1):
    kp = ((K // 32 + 3) // 4) * 4
    return scale.reshape(-1)[r0 * kp:r1 * kp]


def install():
    if SPLIT <= 0:
        return
    assert SPLIT % 128 == 0, "DERPSEEK_GEMM_SPLIT must be a multiple of 128"
    mod = importlib.import_module("vllm.model_executor.kernels.linear.mxfp8.flashinfer")
    qmod = importlib.import_module("vllm.model_executor.layers.fusion.quant_activation")
    cls = mod.FlashInferCutedslMxfp8LinearKernel
    if getattr(cls, "_derpseek_split", False):
        return
    orig = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        K, N = layer.weight.shape
        qa = qmod.as_quantized_activation(x, self.input_quant_key())
        data = qa.data if qa is not None else x
        m = data.numel() // K
        if m <= SPLIT or bias is not None:
            return orig(self, layer, x, bias)
        n_pieces = (m + SPLIT - 1) // SPLIT
        piece = ((m + n_pieces - 1) // n_pieces + 127) // 128 * 128
        shape = qa.orig_shape if qa is not None else x.shape
        out = None
        for r0 in range(0, m, piece):
            r1 = min(m, r0 + piece)
            if qa is not None:
                s1 = (r1 + 127) // 128 * 128 if r1 == m else r1
                sub = dataclasses.replace(qa, data=qa.data.reshape(-1, K)[r0:r1],
                                          scale=_scale_rows(qa.scale, K, r0, s1), orig_shape=torch.Size((r1 - r0, K)))
            else:
                sub = x.reshape(-1, K)[r0:r1]
            y = orig(self, layer, sub, None).reshape(r1 - r0, N)
            if out is None:
                out = torch.empty(m, N, dtype=y.dtype, device=y.device)
            out[r0:r1].copy_(y)
        return out.view(*shape[:-1], N)

    cls.apply_weights = apply_weights
    cls._derpseek_split = True
    _LOG(f"FlashInfer CuTe-DSL MXFP8 GEMMs above {SPLIT} rows run as row pieces of <= {SPLIT}")
