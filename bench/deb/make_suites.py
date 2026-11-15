"""Expert-pinning bench, step 1: one suite of 2,048-token windows (KLD corpus format) labelled by group, from the KLD
corpus (ordinary text), the degenerate windows (make_corpus.py) and the varied corpus (tools/varied_corpus.py), plus
optionally the windows that use rarely used experts most (--cold: a JSON {window id: cold-pick rate} from routing).
Window order is shuffled, so a KLD run 16 windows at a time mixes groups the way unrelated requests do.
   python3 -I make_suites.py OUT.json --kld CORPUS.json --deb DEB.json --varied VARIED.json [--cold RATES.json]
       [--per-group 12]"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--kld")
ap.add_argument("--deb")
ap.add_argument("--varied")
ap.add_argument("--cold", help="JSON {window id: share of picks on rarely used experts} from routing")
ap.add_argument("--cold-n", type=int, default=16)
ap.add_argument("--per-group", type=int, default=12)
ap.add_argument("--holdout", action="store_true", help="leave each varied category's first window (-0) out of the suite "
                "(it stays free for calibrating approaches, e.g. a rowmap, so the suite tests them on unseen text)")
ap.add_argument("--calib-dir", help="with --holdout: link each held-out window's routing file here as GROUP-ID (a "
                "family per bench group for tools/make_rowmap.py)")
ap.add_argument("--calib-routing", help="routing dir of the varied corpus (dsv41 ref_logits.py output)")
a = ap.parse_args()
rng = random.Random(20261115)

LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "sv", "fi", "pl", "tr", "vi", "id", "sw", "yo", "spa", "eng"}
CJK = {"ja", "ko", "zh", "cmn", "jpn"}
CODE_RARE = {"haskell", "commonlisp", "fortran", "cobol", "x86assembly", "prolog", "erlang", "ocaml", "forth", "ada",
             "algol68", "j"}


def group_of(stratum: str) -> str:
    kind, _, rest = stratum.partition("_")
    if kind in ("wiki", "web"):
        return "latin-script" if rest in LATIN else "cjk" if rest in CJK else "other-scripts"
    if kind == "code":
        return "code-rare" if rest in CODE_RARE else "code-main"
    if kind in ("struct", "chat"):
        return {"struct": "structured", "chat": "chat"}[kind]
    if kind in ("math", "sci"):
        return "math-science"
    return kind


groups = {}
if a.kld:
    for w in json.load(open(a.kld)):
        if len(w["ids"]) == 2048:
            groups.setdefault("ordinary", []).append(w)
if a.deb:
    for w in json.load(open(a.deb)):
        if w["id"].split("-")[0] in ("rand", "shuf", "switch"):
            groups.setdefault("degenerate", []).append(w)
varied = json.load(open(a.varied)) if a.varied else []
if a.holdout:
    if a.calib_dir and a.calib_routing:
        import os
        os.makedirs(a.calib_dir, exist_ok=True)
        for w in varied:
            src = os.path.join(a.calib_routing, w["id"] + ".safetensors")
            dst = os.path.join(a.calib_dir, f"{group_of(w['stratum'])}-{w['id']}.safetensors")
            if w["id"].endswith("-0") and os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
    varied = [w for w in varied if not w["id"].endswith("-0")]
for w in varied:
    groups.setdefault(group_of(w["stratum"]), []).append(w)
out, seen = [], set()
if a.cold:
    rates = json.load(open(a.cold))
    pool = [w for w in varied if w["id"] in rates]
    for w in sorted(pool, key=lambda w: -rates[w["id"]])[:a.cold_n]:
        out.append(dict(w, group="cold-heavy"))
        seen.add(w["id"])
for g, ws in sorted(groups.items()):
    ws = [w for w in ws if w["id"] not in seen]
    rng.shuffle(ws)
    for w in ws[:a.per_group]:
        out.append(dict(w, group=g))
rng.shuffle(out)
json.dump(out, open(a.out, "w"))
count = {}
for w in out:
    count[w["group"]] = count.get(w["group"], 0) + 1
print(f"{len(out)} windows -> {a.out}: " + ", ".join(f"{g} {n}" for g, n in sorted(count.items())))
