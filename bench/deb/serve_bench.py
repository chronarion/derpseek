#!/usr/bin/env python3
"""Expert-pinning bench, step 2: prompt reading and decode speed per group of a suite (make_suites.py), against a
running Derpseek server, with token-id prompts (every approach reads the same tokens).
  prefill  per group: prompts of --prefill tokens from the group's windows behind a unique random lead-in (no prefix
           cache hits), max_tokens 1; prompt tokens per second end to end (the first run is a warm-up)
  decode   per group: --c streams, each continuing a random 1,024-token slice of one of the group's windows
           (temperature 1, --max-tokens new tokens), for --seconds; total and per-stream decode tokens per second
  misses   with --docker NAME, the server's residency lines logged during each decode phase: share of non-static
           picks read from CPU memory and host-row picks per step
Standard library only.
   python3 -I serve_bench.py SUITE.json OUT.json [--url http://127.0.0.1:8112/v1] [--groups a,b] [--prefill 32768]
       [--c 8] [--seconds 45] [--docker derpseek]"""
import argparse
import json
import random
import re
import subprocess
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("suite")
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8112/v1")
ap.add_argument("--model", default="derpseek")
ap.add_argument("--groups", default="")
ap.add_argument("--prefill", type=int, default=32768)
ap.add_argument("--prefill-runs", type=int, default=3)
ap.add_argument("--c", type=int, default=8)
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--max-tokens", type=int, default=384)
ap.add_argument("--docker", default="")
a = ap.parse_args()
if a.docker:
    a.seconds = max(a.seconds, 65)   # residency lines come every ~30 s: a decode phase must hold at least one whole one
W = json.load(open(a.suite))
groups = {}
for w in W:
    groups.setdefault(w["group"], []).append(w["ids"])
if a.groups:
    groups = {g: v for g, v in groups.items() if g in a.groups.split(",")}
pool = sorted({t for ws in groups.values() for ids in ws for t in ids[1:]})
rng = random.Random(time.time_ns())
BOS = W[0]["ids"][0]


def post(body, stream=False, timeout=900):
    req = urllib.request.Request(a.url + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def prefill(ws):
    """Prompt tokens per second of runs 2.. (run 1 warms up the shapes)."""
    rates = []
    for r in range(a.prefill_runs + 1):
        lead = [rng.choice(pool) for _ in range(16)]
        body_ids, order = [BOS] + lead, list(range(len(ws)))
        rng.shuffle(order)
        i = 0
        while len(body_ids) < a.prefill:
            body_ids += ws[order[i % len(order)]][1:]
            i += 1
        body_ids = body_ids[:a.prefill]
        t0 = time.perf_counter()
        with post({"model": a.model, "prompt": body_ids, "max_tokens": 1, "temperature": 0}) as resp:
            out = json.load(resp)
        dt = time.perf_counter() - t0
        if r:
            rates.append(out["usage"]["prompt_tokens"] / dt)
    return rates


def decode(ws):
    """C streams for --seconds: (total decode tok/s over the window, mean per-stream tok/s, requests)."""
    stop = time.time() + a.seconds
    lock, toks, per, reqs = threading.Lock(), [0], [], [0]

    def one():
        while time.time() < stop:
            ids = ws[rng.randrange(len(ws))]
            s = rng.randrange(1, len(ids) - 1024)
            body = {"model": a.model, "prompt": [BOS] + ids[s:s + 1024], "max_tokens": a.max_tokens,
                    "temperature": 1.0, "stream": True, "stream_options": {"include_usage": True}}
            n, t_first, t_last = 0, None, None
            try:
                with post(body, stream=True) as resp:
                    for line in resp:
                        line = line.strip()
                        if not line.startswith(b"data: ") or line == b"data: [DONE]":
                            continue
                        ev = json.loads(line[6:])
                        now = time.perf_counter()
                        if ev.get("choices") and ev["choices"][0].get("text"):
                            t_first = t_first or now
                            t_last = now
                        if ev.get("usage"):
                            n = ev["usage"]["completion_tokens"]
            except Exception:
                continue
            with lock:
                if time.time() <= stop + 5:
                    toks[0] += n
                    reqs[0] += 1
                    if t_first and t_last and t_last > t_first and n > 1:
                        per.append((n - 1) / (t_last - t_first))

    th = [threading.Thread(target=one) for _ in range(a.c)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    return toks[0] / (time.time() - t0), (sum(per) / len(per) if per else 0.0), reqs[0]


def misses(t0, t1):
    """Residency lines (logged every ~30 s, each covering the time since the previous one) printed between t0 + 31 and
    t1 + 1, i.e. covering decode time only (a prompt chunk's picks would swamp the decode steps'): mean miss share and
    host-row picks per step."""
    if not a.docker:
        return None
    try:
        cp = subprocess.run(["docker", "logs", "--timestamps", "--since", str(int(t0)), a.docker], capture_output=True,
                            text=True, timeout=60)
        txt = cp.stdout + cp.stderr
    except Exception:
        return None
    import datetime
    keep = []
    for line in txt.splitlines():
        if "residency:" not in line:
            continue
        try:
            ts = datetime.datetime.fromisoformat(line.split(" ", 1)[0][:26]).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        if t0 + 31 <= ts.timestamp() <= t1 + 1:
            keep.append(line)
    m = re.findall(r"misses ([0-9.]+)% of non-static picks, ([0-9.]+) per step", "\n".join(keep))
    if not m:
        return None
    return dict(miss_pct=sum(float(x) for x, _ in m) / len(m), host_picks_per_step=sum(float(y) for _, y in m) / len(m),
                lines=len(m))


res = {}
for g, ws in sorted(groups.items()):
    r = {"windows": len(ws)}
    if a.prefill:
        rates = prefill(ws)
        r["prefill_tok_s"] = sum(rates) / len(rates)
        r["prefill_runs"] = [round(x) for x in rates]
    if a.c:
        time.sleep(31)   # a residency line closes the prompt-reading phase
        t0 = time.time()
        tot, per, n = decode(ws)
        t1 = time.time()
        time.sleep(31)   # and one closes the decode phase
        r.update(decode_tok_s=tot, decode_per_stream=per, decode_requests=n, decode_c=a.c)
        r["misses"] = misses(t0, t1)
    res[g] = r
    print(f"{g:<14} prefill {r.get('prefill_tok_s', 0):8.0f} tok/s | decode C{a.c} {r.get('decode_tok_s', 0):7.1f} tok/s "
          f"({r.get('decode_per_stream', 0):5.1f} per stream, {r.get('decode_requests', 0)} requests) | misses "
          f"{(r.get('misses') or {}).get('miss_pct', float('nan')):5.1f}%", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
