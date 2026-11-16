"""Derpseek 6-bit scale codes: the engine side of the MegaMoE fork's namespace derpseek_sf6 (Fable round 23a, v5).

A routed weight scale (UE8M0, one byte per 32 weights of a row) is stored as scale - E0 in 6 bits. The fork loads the
codes with a TMA and the scale copy's 6-bit form unpacks them into tensor memory, so nothing runs per K block; the
block-scaled MMA multiplies by 2^(weight scale + token scale - 254), and the kernel adds E0 to the token scales instead
(the pull, the all-token head, the first GEMM's epilogue). Measured on the GB300 in stage A: that shift is bit-exact, and
the 6-bit copy leaves each code as one byte.

A coded bank is a uint8 tensor [rows, row_bytes], one row per expert (HBM rows, cache rows, host rows, staging slots:
all alike, copied as whole rows). For a matrix with N output rows and KB 128-wide K blocks, a tile is 128 stored
positions (one CTA's box): 32 chunks of 16 scales = 12 packed bytes (scale i of a chunk at bit 6 i, little-endian).
  layout 1 (bytes)     [N / 4 chunk rows][pitch]: a chunk row holds its 12 bytes for every K block in order, pitch =
                       12 KB rounded up to 16 (the TMA's stride rule). The fork's box is 16 bytes x 32 rows at byte
                       12 k: 12 bytes of codes and 4 bytes the copy ignores (the next K block's; zeros past the end).
  layout 2 (16U6)      [N / 128 tiles][KB][4 rows of 96 bytes]: the box is 128 codes x 4 rows of the TMA's own 6-bit
                       type (CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B: box width 128, strides and address multiples of
                       32, a 4-byte gap after every 12 bytes in shared memory). Dense: exactly 3/4 of today's bytes.
The fork header's DRP_SF6_E0 and DRP_SF6_LAYOUT must equal E0 and LAYOUT here."""
import ctypes
import os

import torch

E0 = int(os.environ.get("DERPSEEK_SF6_E0", "96"))
LAYOUT = int(os.environ.get("DERPSEEK_SF6_LAYOUT", "2"))
PROF = os.environ.get("DERPSEEK_SF6_PROF", "0") == "1"
# layout 1 only: 4 spare bytes at the end of every chunk row, so that no box ends past the tensor (the last K block's
# box otherwise does, by 4 bytes that the TMA fills with zeros and the copy ignores); costs 12 rows of the 165
SLACK = os.environ.get("DERPSEEK_SF6_SLACK", "0") == "1"
CLONE_BYTES = 160 * 4 * 128      # derpseek_host::kMaxSMs CTAs x 4 maps x 128 B
NUM_MAPS = 4                     # after the clone table: host L1, host L2, HBM L1, HBM L2 scale maps
TAIL_BYTES = NUM_MAPS * 128 + 128   # the maps, then one int64 (the profile counters' address) in a slot of its own
CTL_ON, CTL_PROF = 1, 2          # control word 11
HOST_ROWS_CLAIMED = 4096         # rows the host-row maps claim: the fork clones them per call onto the call's bases
#                                  (control words 2-9), so staging buffers of any row count are read through them
STAT_WORDS = 8                   # profile counters per CTA: routed [K blocks, waited, wait cycles, issue cycles], shared [...]
_DT_UINT8, _DT_16U6_ALIGN16B = 0, 15   # CUtensorMapDataType


def pitch(KB: int) -> int:
    """layout 1: bytes of a chunk row (12 per K block, rounded up to the TMA's stride multiple of 16)"""
    return -(-(KB * 12 + (4 if SLACK else 0)) // 16) * 16


def row_bytes(N: int, KB: int) -> int:
    """bytes of one expert's coded scales for a matrix with N output rows and KB K blocks"""
    T = N // 128
    return T * 32 * pitch(KB) if LAYOUT == 1 else T * KB * 384


def pack6(codes: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 16] (values 0..63) -> uint8 [..., 12]: code i at bit 6 i"""
    c = codes.to(torch.int32).unflatten(-1, (4, 4))
    v = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    return torch.stack((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF), -1).flatten(-2).to(torch.uint8)


def unpack6(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 12] -> uint8 [..., 16]"""
    b = packed.to(torch.int32).unflatten(-1, (4, 3))
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    return torch.stack([(v >> (6 * i)) & 63 for i in range(4)], -1).flatten(-2).to(torch.uint8)


def _tiles(t: torch.Tensor) -> torch.Tensor:
    """transformed scale tensor int32 [n, N, KB] (memory [n][KB][N], four scale bytes per word) -> uint8
    [n, tiles, KB, 32 chunks, 16]: every (tile, K block) box in the order of its 512 bytes in memory"""
    n, N, KB = t.shape
    assert t.dtype == torch.int32 and N % 128 == 0, (t.dtype, t.shape)
    b = t.permute(0, 2, 1).contiguous().view(torch.uint8)
    return b.view(n, KB, N // 128, 32, 16).permute(0, 2, 1, 3, 4)


def encode(t: torch.Tensor, e0: int = None, step: int = 16) -> torch.Tensor:
    """transformed scale tensor int32 [n, N, KB] -> coded bank uint8 [n, row_bytes] on the same device"""
    e0 = E0 if e0 is None else e0
    n, N, KB = t.shape
    T = N // 128
    out = torch.zeros(n, row_bytes(N, KB), dtype=torch.uint8, device=t.device)
    for i in range(0, n, step):
        x = _tiles(t[i:i + step]).to(torch.int16) - e0
        lo, hi = int(x.min()), int(x.max())
        if lo < 0 or hi > 63:
            raise ValueError(f"scale bytes {lo + e0}..{hi + e0} do not fit 6-bit codes with E0 = {e0}")
        p = pack6(x.to(torch.uint8))                              # [m, T, KB, 32, 12]
        m = p.shape[0]
        if LAYOUT == 1:
            o = out[i:i + m].view(m, T * 32, pitch(KB))
            o[:, :, :KB * 12] = p.permute(0, 1, 3, 2, 4).reshape(m, T * 32, KB * 12)
        else:
            out[i:i + m] = p.reshape(m, -1)
    return out


def decode(coded: torch.Tensor, N: int, KB: int, e0: int = None) -> torch.Tensor:
    """coded bank -> int32 [n, N, KB] with the transform's strides (a view of contiguous [n][KB][N] memory)"""
    e0 = E0 if e0 is None else e0
    n, T = coded.shape[0], N // 128
    if LAYOUT == 1:
        p = coded.view(n, T, 32, pitch(KB))[..., :KB * 12].reshape(n, T, 32, KB, 12).permute(0, 1, 3, 2, 4)
    else:
        p = coded.view(n, T, KB, 32, 12)
    b = (unpack6(p.contiguous()).to(torch.int16) + e0).to(torch.uint8)       # [n, T, KB, 32, 16]
    m = b.permute(0, 2, 1, 3, 4).reshape(n, KB, N * 4).contiguous().view(torch.int32)
    return m.permute(0, 2, 1)


def span(coded: torch.Tensor, rows: int, raw_shape, raw_stride) -> torch.Tensor:
    """The tensor the launcher is given for a coded bank: it checks an uncoded int32 scale tensor [rows, N, KB] with the
    transform's strides (raw_shape, raw_stride: those of the uncoded tensor) and builds its own tensor map from the
    data pointer. The fork loads nothing through that map in a coded call (routed tiles go through the maps of
    write_tail); the launcher's checks are satisfied, nothing more."""
    class _A:
        __cuda_array_interface__ = {"shape": (rows,) + tuple(raw_shape[1:]), "typestr": "<i4", "data": (coded.data_ptr(), False),
                                    "strides": tuple(s * 4 for s in raw_stride), "version": 3}
    out = torch.as_tensor(_A(), device=coded.device)
    return out if out.dtype == torch.int32 else out.view(torch.int32)


_lib = None


def _encode_map(dtype: int, addr: int, dims, strides, box) -> bytes:
    """cuTensorMapEncodeTiled (no interleave, no swizzle, L2 promotion 256 B, zero fill): the 128-byte tensor map.
    Needs this thread's CUDA context (callers on a thread without one: `with torch.cuda.device(dev):`)."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
        _lib.cuTensorMapEncodeTiled.restype = ctypes.c_int
        _lib.cuTensorMapEncodeTiled.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    rank = len(dims)
    buf = (ctypes.c_uint64 * 32)()
    base = ctypes.addressof(buf)
    base += (-base) % 128                                        # CUtensorMap: alignas(128) (the encoder asks for 64)
    r = _lib.cuTensorMapEncodeTiled(base, dtype, rank, addr, (ctypes.c_uint64 * rank)(*dims),
                                    (ctypes.c_uint64 * max(rank - 1, 1))(*strides), (ctypes.c_uint32 * rank)(*box),
                                    (ctypes.c_uint32 * rank)(*([1] * rank)), 0, 0, 3, 0)
    if r != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled error {r}: type {dtype} address {addr:#x} dims {list(dims)} "
                           f"strides {list(strides)} box {list(box)}")
    return ctypes.string_at(base, 128)


def map_args(addr: int, rows: int, N: int, KB: int):
    """(data type, address, dims, strides, box) of the tensor map over a coded bank of `rows` experts"""
    T = N // 128
    if LAYOUT == 1:
        return _DT_UINT8, addr, [pitch(KB), rows * T * 32], [pitch(KB)], [16, 32]
    assert addr % 32 == 0
    return _DT_16U6_ALIGN16B, addr, [128, rows * T * KB * 4], [96], [128, 4]


def write_tail(tail: torch.Tensor, host1: torch.Tensor, host2: torch.Tensor, hbm1: torch.Tensor, hbm2: torch.Tensor,
               shape1, shape2, stats: torch.Tensor = None) -> None:
    """Fill the clone table's tail (uint8 view of TAIL_BYTES): the four coded scale maps and the counters' address.
    host*: the pinned coded host banks (the fork re-bases its per-call clones of these two maps onto control words
    4-5 / 8-9, so the same maps serve staging buffers); hbm*: the coded HBM banks (HBM row r = kernel row r + D);
    shape1 / shape2: (N, KB) of the first and second GEMM's weights."""
    assert tail.numel() == TAIL_BYTES and tail.dtype == torch.uint8
    blobs = [_encode_map(*map_args(host1.data_ptr(), HOST_ROWS_CLAIMED, *shape1)),
             _encode_map(*map_args(host2.data_ptr(), HOST_ROWS_CLAIMED, *shape2)),
             _encode_map(*map_args(hbm1.data_ptr(), hbm1.shape[0], *shape1)),
             _encode_map(*map_args(hbm2.data_ptr(), hbm2.shape[0], *shape2))]
    tail[:NUM_MAPS * 128].copy_(torch.frombuffer(bytearray(b"".join(blobs)), dtype=torch.uint8))
    tail[NUM_MAPS * 128:NUM_MAPS * 128 + 8].view(torch.int64)[0] = 0 if stats is None else stats.data_ptr()


def ctl_word(prof: bool = None) -> int:
    return CTL_ON | (CTL_PROF if (PROF if prof is None else prof) else 0)


def new_stats(device) -> torch.Tensor:
    return torch.zeros(160 * STAT_WORDS, dtype=torch.int64, device=device)


def stats_line(stats: torch.Tensor) -> str:
    s = stats.view(160, STAT_WORDS).sum(0).tolist()
    part = lambda name, k, w, cw, ci: (f"{name}: {k} K blocks, had to wait for {100.0 * w / k:.1f}%, {cw / k:.0f} cycles per K block "
                                       f"in the wait, {ci / k:.0f} from the wait to the commit") if k else f"{name}: none"
    return part("routed tiles", *s[0:4]) + "; " + part("shared tiles", *s[4:8])
