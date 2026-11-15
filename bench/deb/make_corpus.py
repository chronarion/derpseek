"""Degenerate expert bench (DEB), step 1: a 16-window corpus in the KLD corpus format that pushes expert routing into
the cases the one-conversation KLD measurement never sees. Every window is 2,048 tokens of the model's own tokens:
  wiki-0, code-0, chat-0, zh-0   4 ordinary windows, copied: the control (and the only ones with official references)
  rand-*    4 windows of random tokens drawn from every token the corpus uses: the widest spread of experts
  shuf-*    4 ordinary windows with their tokens shuffled: an ordinary token mix with no structure
  switch-*  4 windows that switch family every 64 tokens (wiki, code, chat, math, zh, long in turn): constant churn
Decoded 16 at a time (decode_kld.py --batch 16, 4 rows per window per step = 64 rows, the served 16-request shape),
the lossy profiles see a full step of unrelated, hard-to-cache requests.
   python3 -I make_corpus.py KLD_CORPUS.json OUT.json"""
import json
import random
import sys

src, out = sys.argv[1:3]
W = {w["id"]: w for w in json.load(open(src))}
rng = random.Random(20261115)
N = 2048


def win(id_, ids, source):
    assert len(ids) == N, (id_, len(ids))
    return {"id": id_, "ids": ids, "partition": "deb", "score_from": 0, "sha256": "", "source": source,
            "stratum": id_.split("-")[0]}


out_w = []
for fam in ("wiki", "code", "chat", "zh"):
    w = W[f"{fam}-0"]
    out_w.append(win(f"{fam}-0", list(w["ids"][:N]), "copy (the official references match it)"))
pool = sorted({t for w in W.values() for t in w["ids"]})
for k in range(4):
    out_w.append(win(f"rand-{k}", [rng.choice(pool) for _ in range(N)], "uniform over the corpus's tokens"))
for k, base in enumerate(("wiki-1", "code-1", "chat-1", "math-1")):
    ids = list(W[base]["ids"][:N])
    rng.shuffle(ids)
    out_w.append(win(f"shuf-{k}", ids, f"{base} shuffled"))
fams = [["wiki-2", "wiki-3"], ["code-2", "code-3"], ["chat-2", "chat-3"], ["math-0", "math-1"], ["zh-0", "zh-1"],
        ["long-0", "long-1"]]
for k in range(4):
    ids, i = [], 0
    while len(ids) < N:
        f = fams[(i + k) % len(fams)]
        src_ids = W[f[(i // len(fams)) % 2]]["ids"]
        off = rng.randrange(0, len(src_ids) - 64)
        ids += src_ids[off:off + 64]
        i += 1
    out_w.append(win(f"switch-{k}", ids[:N], "64-token pieces of the six families in turn"))
json.dump(out_w, open(out, "w"))
print(f"{len(out_w)} windows -> {out}: {', '.join(w['id'] for w in out_w)}")
