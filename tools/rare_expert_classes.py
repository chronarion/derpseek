"""Token classes of cold picks vs all picks vs all tokens (same inputs as analyze.py)."""
import collections, json, re, sys
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
CORPUS, REF, ROWMAP, TOKJ = sys.argv[1:5]
tok = Tokenizer.from_file(TOKJ)
W = json.load(open(CORPUS)); R = json.load(open(ROWMAP))["layers"]
L = 40
cold = {l: torch.tensor(sorted(set(R[str(l)]["host"]) - set(R[str(l)]["cache"]))) for l in range(L)}
CJKP = set("，。、！？：；“”‘’（）《》「」『』—…·")
def cls(s):
    if re.fullmatch(r"<｜.*｜>|<.*>", s): return "special token"
    t = s.strip()
    if not t: return "newline" if "\n" in s else "whitespace"
    if all(c in CJKP for c in t): return "CJK punctuation"
    if re.search(r"[一-鿿]", t): return "CJK text"
    if re.fullmatch(r"\d+", t): return "digits"
    if re.fullmatch(r"[A-Za-z]+", t): return "Latin word" if s[0] == " " else "Latin word piece"
    if all(not c.isalnum() for c in t): return "ASCII punctuation / symbols"
    return "other"
cache = {}
def c_of(i):
    if i not in cache: cache[i] = cls(tok.decode([i]))
    return cache[i]
band = {"0-19": range(0, 20), "20-39": range(20, 40)}
tokc = collections.Counter(); allc = {b: collections.Counter() for b in band}; coldc = {b: collections.Counter() for b in band}
for w in W:
    with safe_open(f"{REF}/{w['id']}.safetensors", "pt") as f:
        ex = f.get_tensor("experts").long()
    ids = w["ids"][:ex.shape[1]]
    cl = [c_of(i) for i in ids]
    tokc.update(cl)
    for b, ls in band.items():
        for l in ls:
            m = torch.isin(ex[l], cold[l]).sum(1).tolist()   # cold picks per token
            for c, k in zip(cl, m):
                allc[b][c] += 6
                coldc[b][c] += k
nt = sum(tokc.values())
for b in band:
    tc = sum(coldc[b].values())
    print(f"layers {b}: token class | share of tokens | share of cold picks | cold picks per token of that class")
    for c, n in sorted(tokc.items(), key=lambda kv: -coldc[b][kv[0]]):
        print(f"  {c:<28} {n / nt:6.1%} {coldc[b][c] / tc:7.1%} {coldc[b][c] / n / len(band[b]):8.3f}")
