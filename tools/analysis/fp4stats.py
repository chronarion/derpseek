import json, struct, sys, numpy as np
M = sys.argv[1]
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
hdr_cache = {}
def tensor(name):
    f = idx[name]
    if f not in hdr_cache:
        with open(f"{M}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr_cache[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = hdr_cache[f]
    e = h[name]
    a, b = e["data_offsets"]
    mm = np.memmap(f"{M}/{f}", dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
    return e["dtype"], e["shape"], np.array(mm)
LAY = [2, 10, 20, 30, 39]; EXP = [0, 100, 200, 383]
codes = np.zeros(16, np.int64); scales = np.zeros(256, np.int64); dsc = np.zeros(512, np.int64)
distinct = np.zeros(17, np.int64); printed = False
for l in LAY:
    for e in EXP:
        for w in ("w1", "w2", "w3"):
            dt, sh, q = tensor(f"layers.{l}.ffn.experts.{e}.{w}.weight")
            dts, shs, s = tensor(f"layers.{l}.ffn.experts.{e}.{w}.scale")
            if not printed:
                print("weight", dt, sh, "scale", dts, shs); printed = True
            lo, hi = q & 15, q >> 4
            nib = np.stack([lo, hi], -1).reshape(sh[0], -1)       # [rows, K] codes
            codes += np.bincount(nib.ravel(), minlength=16)
            scales += np.bincount(s.ravel(), minlength=256)
            s2 = s.reshape(shs[0], -1).astype(np.int16)
            dsc += np.bincount((np.diff(s2, axis=1) + 256).ravel(), minlength=512)
            g = nib.reshape(-1, 32)
            # distinct code values per 32-group (sample of groups)
            gs = g[:: max(1, len(g) // 20000)]
            d = np.array([len(np.unique(x)) for x in gs])
            distinct += np.bincount(d, minlength=17)
p = codes / codes.sum()
H = -(p[p > 0] * np.log2(p[p > 0])).sum()
e2m1 = [0, .5, 1, 1.5, 2, 3, 4, 6]
print("code histogram (code: value: share)")
for c in range(16):
    v = (-1 if c & 8 else 1) * e2m1[c & 7]
    print(f"  {c:2d} {v:+5.1f} {p[c]*100:6.2f}%")
print(f"FP4 code entropy {H:.3f} bits (vs 4)")
ps = scales / scales.sum(); Hs = -(ps[ps > 0] * np.log2(ps[ps > 0])).sum()
nz = np.nonzero(scales)[0]
print(f"scale (e8m0) range {nz.min()}..{nz.max()} (2^{nz.min()-127}..2^{nz.max()-127}), entropy {Hs:.3f} bits")
pd = dsc / dsc.sum(); Hd = -(pd[pd > 0] * np.log2(pd[pd > 0])).sum()
print(f"entropy of scale delta along K {Hd:.3f} bits; P(delta=0) {pd[256]*100:.1f}%, |delta|<=1 {pd[255:258].sum()*100:.1f}%")
print("distinct codes per 32-group:", {k: round(v / distinct.sum() * 100, 1) for k, v in enumerate(distinct) if v})
bpw_lossless = (H * 32 + Hd) / 32
print(f"ideal lossless bpw (code entropy + delta-coded scales): {bpw_lossless:.3f} vs 4.25")
