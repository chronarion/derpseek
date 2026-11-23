import json, re, sys, statistics as S
ev = json.load(open(sys.argv[1])); ev = ev.get("traceEvents", ev) if isinstance(ev, dict) else ev
ks = sorted((e for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda e: e["ts"])
nm = lambda e: re.sub(r"\(.*|<.*", "", e["name"].replace("void ", "").replace("(anonymous namespace)::", "")).split("::")[-1]
N = [nm(e) for e in ks]
LL = lambda n: "low_latency" in n
g_router, g_mega, g_stage, n_layers = [], [], [], 0
for i, n in enumerate(N):
    if n == "sm100_fp8_fp4_mega_moe_impl":
        # the router and the LL GEMM (wo_b) before this MegaMoE
        r = next((j for j in range(i - 1, max(i - 12, 0), -1) if N[j] == "router_kernel"), None)
        w = next((j for j in range(i - 1, max(i - 12, 0), -1) if LL(N[j])), None)
        if r is None or w is None: continue
        wend = ks[w]["ts"] + ks[w]["dur"]
        g_router.append(ks[r]["ts"] + ks[r]["dur"] - wend)
        g_mega.append(ks[i]["ts"] - wend)
        n_layers += 1
    if n == "stage_kernel":
        p = next((j for j in range(i - 1, max(i - 6, 0), -1) if LL(N[j])), None)
        if p is not None:
            g_stage.append(ks[i]["ts"] + ks[i]["dur"] - (ks[p]["ts"] + ks[p]["dur"]))
steps = sum(1 for n in N if n == "markov_kernel")
f = lambda v: f"median {S.median(v):.1f} us, mean {S.mean(v):.1f} (x {len(v) / max(steps, 1):.1f} per step = {S.mean(v) * len(v) / max(steps, 1):.0f} us/step)"
print(f"{steps} steps, {n_layers} MoE layers with a router")
print("wo_b end -> router end:", f(g_router))
print("wo_b end -> MegaMoE launch:", f(g_mega))
print("q_a+kv_a end -> stage end:", f(g_stage))
