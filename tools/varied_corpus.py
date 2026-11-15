"""A varied corpus for the rare-experts study (docs/rare-experts.md), in the KLD corpus format: windows of 2,048
DeepSeek-V4.1 tokens (BOS + 2,047), each packed from several documents of one kind so that no single document
dominates, scored at the last position only (so dsv41's ref_logits.py writes routing for every position and almost no
logits). Streams small samples from public Hugging Face datasets (datasets, tokenizers).
   python3 build: varied_corpus.py TOKENIZER_JSON OUT.json [--per 3]"""
import argparse
import json
import random
import sys

import datasets
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer")
ap.add_argument("out")
ap.add_argument("--per", type=int, default=3, help="windows per natural-language kind")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
BOS = tok.token_to_id("<｜begin▁of▁sentence｜>")
USER, ASSIST, EOS = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
N = 2048
rng = random.Random(20261115)
out = []


def pack(kind, texts, nwin, source):
    """Windows of N tokens from consecutive documents (each window starts on a fresh document)."""
    made, buf = 0, []
    for t in texts:
        if made >= nwin:
            break
        if not t or not t.strip():
            continue
        buf += tok.encode(t.strip() + "\n\n", add_special_tokens=False).ids
        if len(buf) >= N - 1:
            out.append({"id": f"{kind}-{made}", "ids": [BOS] + buf[:N - 1], "partition": "varied", "score_from": N - 1,
                        "sha256": "", "source": source, "stratum": kind})
            made, buf = made + 1, []
    sys.stderr.write(f"{kind}: {made} windows\n")


def stream(name, cfg=None, split="train", skip=0):
    ds = datasets.load_dataset(name, cfg, split=split, streaming=True)
    return iter(ds.skip(skip)) if skip else iter(ds)


# natural language: Wikipedia in 30 languages (15 scripts)
WIKI = ["en", "de", "fr", "es", "it", "pt", "nl", "sv", "fi", "pl", "tr", "vi", "id", "sw", "yo",
        "ru", "uk", "el", "ar", "fa", "he", "hi", "bn", "ta", "th", "ka", "am", "ja", "ko", "zh"]
for lang in WIKI:
    try:
        it = stream("wikimedia/wikipedia", f"20231101.{lang}", skip=rng.randrange(200, 3000))
        pack(f"wiki_{lang}", (r["text"] for r in it), a.per, f"wikimedia/wikipedia 20231101.{lang}")
    except Exception as e:
        sys.stderr.write(f"wiki {lang} failed: {e!r}\n")
# web text
for cfg in ("rus_Cyrl", "cmn_Hani", "jpn_Jpan", "arb_Arab", "hin_Deva", "spa_Latn"):
    try:
        pack(f"web_{cfg.split('_')[0]}", (r["text"] for r in stream("HuggingFaceFW/fineweb-2", cfg)), 2,
             f"HuggingFaceFW/fineweb-2 {cfg}")
    except Exception as e:
        sys.stderr.write(f"web {cfg} failed: {e!r}\n")
try:
    pack("web_eng", (r["text"] for r in stream("HuggingFaceFW/fineweb", "sample-10BT")), 2, "HuggingFaceFW/fineweb")
except Exception as e:
    sys.stderr.write(f"web eng failed: {e!r}\n")
# code: Rosetta Code, the same tasks in many languages
CODE = ["Python", "C", "C++", "Java", "JavaScript", "Go", "Rust", "Haskell", "Common Lisp", "Fortran", "COBOL",
        "X86 Assembly", "Perl", "Ruby", "PHP", "SQL", "UNIX Shell", "Lua", "R", "Julia", "Prolog", "Erlang", "OCaml",
        "Kotlin", "Swift", "Scala", "Forth", "Ada", "ALGOL 68", "J"]
code = {c: [] for c in CODE}
try:
    for n, r in enumerate(stream("christopher/rosetta-code")):
        if n == 0:
            sys.stderr.write(f"rosetta fields: {sorted(r)}\n")
        lang = r.get("language_name") or (r.get("language_url") or "").lstrip("#").replace("_", " ")
        if lang in code and sum(len(x) for x in code[lang]) < 60000:
            code[lang].append(f"// {r['task_name']}\n{r.get('code') or ''}")
    for lang, texts in code.items():
        rng.shuffle(texts)
        name = lang.lower().replace("c++", "cpp").replace("c#", "csharp")
        pack("code_" + "".join(ch for ch in name if ch.isalnum()), texts, 2, f"christopher/rosetta-code {lang}")
except Exception as e:
    sys.stderr.write(f"rosetta failed: {e!r}\n")
# math in LaTeX, chats, biomedical text
try:
    pack("math_latex", (f"{USER}{r['problem']}{ASSIST}</think>{r['solution']}{EOS}"
                        for r in stream("HuggingFaceH4/MATH-500", split="test")), 6, "HuggingFaceH4/MATH-500")
except Exception as e:
    sys.stderr.write(f"math failed: {e!r}\n")
try:
    pack("chat_ultra", ("".join(f"{USER if m['role'] == 'user' else ASSIST}{m['content']}" for m in r["messages"]) + EOS
                        for r in stream("HuggingFaceH4/ultrachat_200k", split="train_sft")), 4,
         "HuggingFaceH4/ultrachat_200k")
except Exception as e:
    sys.stderr.write(f"ultrachat failed: {e!r}\n")
try:
    by = {}
    for r in stream("OpenAssistant/oasst1"):
        by.setdefault(r["lang"], []).append(f"{USER if r['role'] == 'prompter' else ASSIST}{r['text']}")
    for lang in ("en", "es", "ru", "de", "zh", "fr", "pt-BR", "ca", "uk", "ja"):
        if lang in by:
            pack(f"chat_oasst_{lang.replace('-', '').lower()}", by[lang], 2, f"OpenAssistant/oasst1 {lang}")
except Exception as e:
    sys.stderr.write(f"oasst failed: {e!r}\n")
try:
    pack("sci_pubmed", (r["article"] for r in stream("ccdv/pubmed-summarization", "document")), 4,
         "ccdv/pubmed-summarization")
except Exception as e:
    sys.stderr.write(f"pubmed failed: {e!r}\n")
# structured data, generated from records of the sources above (no private data)
recs = [r for _, r in zip(range(400), stream("HuggingFaceH4/MATH-500", split="test"))]
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
def j(i):
    r = recs[i % len(recs)]
    return json.dumps({"id": i, "subject": r["subject"], "level": r["level"], "answer": r["answer"],
                       "tags": rng.sample(words, 3), "score": round(rng.random(), 4)}, indent=2, ensure_ascii=False)
pack("struct_json", (j(i) for i in range(5000)), 3, "generated JSON records")
pack("struct_csv", ("id,subject,level,score,ratio\n" + "\n".join(
    f"{i},{recs[i % len(recs)]['subject']},{recs[i % len(recs)]['level']},{rng.randint(0, 999)},{rng.random():.5f}"
    for i in range(k * 40, k * 40 + 40)) for k in range(500)), 3, "generated CSV")
pack("struct_yaml", (f"service:\n  name: {rng.choice(words)}-{k}\n  replicas: {rng.randint(1, 9)}\n  ports:\n"
                     f"    - {rng.randint(1024, 65535)}\n  env:\n    LEVEL: {rng.choice(words)}\n    TIMEOUT_MS: "
                     f"{rng.randint(100, 9000)}\n  enabled: {rng.choice(['true', 'false'])}" for k in range(5000)), 3,
     "generated YAML")
pack("struct_html", (f"<div class=\"card\"><h2>{recs[k % len(recs)]['subject']}</h2><p>{recs[k % len(recs)]['problem']}"
                     f"</p><ul><li>level {recs[k % len(recs)]['level']}</li></ul></div>" for k in range(5000)), 3,
     "generated HTML")
pack("struct_log", ("\n".join(f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}:"
                              f"{rng.randint(10, 59)}Z {rng.choice(['INFO', 'WARN', 'ERROR', 'DEBUG'])} "
                              f"{rng.choice(['api', 'db', 'cache', 'auth'])}[{rng.randint(100, 9999)}] "
                              f"GET /v{rng.randint(1, 3)}/{rng.choice(words)}/{rng.randint(1, 99999)} "
                              f"{rng.choice([200, 201, 304, 404, 500])} {rng.randint(1, 900)}ms" for _ in range(20))
                    for _ in range(5000)), 3, "generated logs")
pack("struct_mdtable", ("| id | subject | level | answer |\n|---|---|---|---|\n" + "\n".join(
    f"| {i} | {recs[i % len(recs)]['subject']} | {recs[i % len(recs)]['level']} | {recs[i % len(recs)]['answer'][:20]} |"
    for i in range(k * 20, k * 20 + 20)) for k in range(500)), 2, "generated Markdown tables")
json.dump(out, open(a.out, "w"))
sys.stderr.write(f"{len(out)} windows, {len(out) * N} tokens -> {a.out}\n")
