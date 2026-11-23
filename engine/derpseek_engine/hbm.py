"""Derpseek: GPU memory handed back to static expert rows (C1 push, 11-22).

DERPSEEK_EMBED_HOST=1: the input embedding tables (the target's embed_tokens and the DSpark drafter's, when it keeps
its own) move to pinned Grace memory, read in place over NVLink-C2C. A decode step gathers 1-8 rows (5120 x BF16 =
10 KB each, ~1 us over the link); a prompt chunk of 8,192 tokens reads 84 MB (~0.25 ms against a ~120 ms chunk). The
table's values are copied bit for bit, so every gathered embedding is unchanged. Tables that hold identical values
share one host copy. Runs right after the model (and drafter) load, before any graph capture, so captured kernels see
the host address from the start. The GPU copies are freed; the rowmap spends the room on static expert rows."""
import os
import sys

import torch

_LOG = lambda m: sys.stderr.write(f"DERPSEEK_HBM {m}\n")
_KEEP = []          # pinned host tensors (their aliases do not own them)


class _CudaView:
    """a CUDA array interface over a pinned host buffer (same address on the GB300: UVA / ATS)"""

    def __init__(self, host: torch.Tensor):
        self.__cuda_array_interface__ = {"shape": tuple(host.shape), "typestr": "<f2" if host.dtype == torch.float16 else "<V2",
                                         "data": (host.data_ptr(), False), "version": 3,
                                         "strides": tuple(s * host.element_size() for s in host.stride())}


def _alias(host: torch.Tensor, dtype, device) -> torch.Tensor:
    """a cuda tensor aliasing the pinned host buffer"""
    raw = host.view(torch.int16)
    t = torch.as_tensor(_CudaView(raw.view(torch.float16)), device=device)   # fp16 typestr: torch reads the bits
    return t.view(dtype)


def _embeddings(*roots):
    seen = {}
    for root in roots:
        if root is None:
            continue
        for name, m in root.named_modules():
            if name.endswith("embed_tokens") and isinstance(getattr(m, "weight", None), torch.Tensor):
                w = m.weight
                if w.is_cuda and w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_contiguous():
                    seen.setdefault(id(m), (name, m))
    return list(seen.values())


def embed_to_host(runner) -> int:
    """move every input embedding table to pinned host memory; returns the GPU bytes freed"""
    drafter = None
    for attr in ("speculator", "drafter", "spec_decoder"):
        s = getattr(runner, attr, None)
        if s is not None and getattr(s, "model", None) is not None:
            drafter = s.model
            break
    mods = _embeddings(getattr(runner, "model", None), drafter)
    if not mods:
        _LOG("no embedding table found; nothing moved")
        return 0
    hosts = []      # (gpu tensor it was copied from, host buffer)
    freed = 0
    for name, m in mods:
        w = m.weight.data
        host, shared = None, False
        for src, h in hosts:
            if src.shape == w.shape and (src.data_ptr() == w.data_ptr() or torch.equal(src, w)):
                host, shared = h, True
                break
        if host is None:
            host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
            host.copy_(w)
            torch.cuda.synchronize()
            if not torch.equal(host.to(w.device), w):
                raise RuntimeError(f"DERPSEEK_EMBED_HOST: host copy of {name} differs")
            hosts.append((w, host))
            _KEEP.append(host)
        alias = _alias(host, w.dtype, w.device)
        if not torch.equal(alias[:4096], w[:4096]) or not torch.equal(alias[-4096:], w[-4096:]):
            raise RuntimeError(f"DERPSEEK_EMBED_HOST: the device view of {name} reads other values")
        nbytes = w.numel() * w.element_size()
        m.weight.data = alias
        freed += nbytes
        _LOG(f"{name}: {tuple(w.shape)} {w.dtype} -> pinned host memory ({nbytes / 2**30:.2f} GiB of HBM"
             f"{', shares an earlier host copy' if shared else ''})")
    del hosts
    torch.cuda.empty_cache()
    _LOG(f"freed {freed / 2**30:.2f} GiB of HBM ({len(_KEEP)} host copies, {sum(h.numel() * h.element_size() for h in _KEEP) / 2**30:.2f} GiB)")
    return freed
