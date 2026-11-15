# Derpseek design notes

Status (2026-11-14): exact mode serves. Sections 1-9 are the design as written before any GPU run; section 10 is
what was built and measured. Figures without "measured" are estimates.

## 1. Goal

Serve DeepSeek-V4.1-Flash-1017 on one DGX Station GB300 with no second GPU, as fast as possible, with at least one
profile whose KLD against the official-precision model stays below 0.08 (decode-path KLD, the same harness as dsv41).
The bar is dsv41, which uses the GB300 plus an RTX PRO 6000 for the coldest experts.

## 2. Why it is hard

- **Capacity.** About 285 GiB of weights have to sit on a GPU; the GB300 has 250.7 GiB. Routed experts are 269.0 GiB
  of that (15,360 expert slots of 35.39M weights in MXFP4 = 4.25 bits per weight).
- **Bandwidth.** Decode is bandwidth-bound at every batch size served. Today the RTX adds ~1.8 TB/s beside the GB300's
  ~7 TB/s, but it is mostly idle: it holds the coldest third of the experts, which are ~2.3 of the ~21 distinct
  experts a batch-1 step reads per layer (measured cost of an expert on the GB300: ~3 us, from a MegaMoE sweep).

## 3. Architecture facts that drive the design

- **Prefill and decode use different halves of the model.** In prefill, layers 0-19 run on every prompt token; the
  last KV-source layer (20) and everything above it run only on each request's last 128 rows (vLLM 0.32's decoder
  replay; DeepSeek's report: 8B active parameters per token at prefill, 16B at decode). Layers 20-39 are
  decode-only in practice.
- **KV is small** (one shared 512-d latent, KV shared across layers from 4 source layers, FP4 KV): ~2.4 KB per token
  with today's pool layout. HBM is a weight problem.
- **Experts are trained in FP4.** CPU: the e2m1 code histogram is nearly flat (entropy 3.895 of 4 bits; a 32-group
  uses 10-13 distinct codes), so the codes are close to incompressible. The UE8M0 scales are not: entropy 0.99 bits,
  and neighbouring scales differ by -1, 0 or +1 in every group measured. A row base plus 2-bit deltas stores them in
  0.0625 bits per weight instead of 0.25: experts 4.25 -> 4.0625 bits per weight, -4.4% of expert bytes, lossless.
- **Experts are nearly orthogonal and full rank** (pair cosine ~0.001): merging and low-rank sharing are out.
- **The drafter** (DSpark, 7.4 GiB) can be compressed with no KLD cost, since speculative decoding keeps the target
  distribution; only acceptance moves.
- **Expert use is skewed but shifts with the text.** Leave-one-window-out on 18 reference windows (CPU): with the
  hottest 76% of expert slots resident, 7.4% of picks miss (2.8% wiki .. 18.9% Chinese). Usage-driven bit allocation
  overfits: damage on held-out text families reached 12x its in-sample value.

## 4. The plan in one paragraph

Split the model at layer 20 in every profile. Layers 0-19 (the prefill path) stay native FP4, always resident, never
coded, so prefill keeps full tensor-core speed. Capacity is recovered in layers 20-39 only. One engine-owned memory
arena replaces vLLM's per-graph pools (~12 GiB today) and the KV pools get a compact layout. Dense FP8 matrices move
to FP6 (E2M3 with a scale per 32: 1.15% relative error against the shipped FP8, -24% of their bytes; the TMA can
unpack it, no ALU). The sidecar, peer protocol and warm slots go away.

## 5. Two profiles (estimates against dsv41 today; lil-bench standard, tokens per second)

| profile | C1 (0 context) | C8 | C16 | prefill | KLD batch 1 / 16 |
|---|---|---|---|---|---|
| dsv41 `default` (GB300 + RTX, measured) | 376 | 1,337 | 1,953 | 75-76K | 0.021 / 0.029 |
| dsv41 `fast` (GB300 + RTX, measured) | 397 | 1,517 | 2,270 | 65K | 0.034 / 0.039 |
| `solo`: exact fit, GB300 only | 377 | 1,285 | 1,865 | 72-78K | 0.021 / 0.029 |
| `solo-fast`: max speed, GB300 only | 428 | 1,620 | 2,390 | 72-78K | ~0.053 (0.045-0.066) |

- `solo`: layers 20-39 keep 288 of 384 experts per layer in HBM; the rest stay exact in Grace memory. The MoE kernel
  reads cold experts over NVLink-C2C first, so that stream runs beside the HBM stream, and a miss promotes the expert
  into HBM (least recently used slot out). Step model at C1: +0.27 ms for the cold experts' bytes back on the
  GB300, -0.25 ms for the peer protocol gone, -0.09 ms for the scale deltas.
- `solo-fast`: layers 20-39 in an on-grid trellis code (TC-FP4: 3.25 bits per weight on 20-29, 3.0 on 30-39) that
  decodes into native e2m1 tiles with idle ALU, plus singleton drop-2, an MXFP8 head and FP6 dense. CPU-measured
  TC-FP4 relative error: 0.045 / 0.070 / 0.110 / 0.133 at 3.75 / 3.5 / 3.25 / 3.0 bits.
- Where the estimates can fail: the Grace tier must overlap with the HBM stream (not yet shown that the TMA can read a
  host address at all; serialized, `solo` loses ~10% at C1 and ~24% at C16); sixteen unrelated streams miss ~2% of
  picks and saturate the link (+13% per step); prefill is the least certain number.

## 6. The C2C link as a constant-bitrate pipe ("wavefront" quantization)

Idea: the Grace -> HBM link (NVLink-C2C, ~340-360 GB/s for in-kernel reads of registered host memory) is idle during
decode. Treat it as a constant-bitrate channel that is always worth filling, racing the layer wavefront of each step,
and pay storage up front so fetches are cheap and well aimed.

Format: embedded bit planes of the FP4 code. HBM keeps the top planes of every non-resident expert (sign and exponent:
a 3-bit base with ~20% relative error, or sign and top exponent bit: a 2-bit base with ~38%); Grace keeps the
remaining planes pre-packed so a fetch is one contiguous copy; base plus refinement reassembles to the exact FP4 code
with a few bit operations, so the tensor-core path is unchanged. Every expert has a ready flag and the MoE kernel never
waits: a late refinement costs precision for that step, never time.

Measured (CPU, leave-one-window-out on routing traces):
- What predicts a cold pick: the same expert was used in the previous 32 / 256 / 2,048 positions for 65 / 84 / 92.5%
  of cold picks. A token -> expert table explains only 27-33%. The request's working set predicts; n-grams do not
  (for cold experts).
- A prompt's prefill routing covers 80-94% of its continuation's cold picks on layers 0-19 (layers 20-39 see only the
  128 replay rows at prefill: 33-80%).
- A refinement cache vs concurrency (static lossless core of the hottest 10,000 slots + 2-bit base + an LRU cache of
  1,654 refinement slots, link budget per step from today's step times): share of all picks at base precision C1
  0.00%, C2 0.01%, C4 0.03%, C8 0.02%, C16 1.19% (the link saturates at C16). Without a pinned core (plain LRU over
  everything) it degrades to 9.3% at C16. Intra-step timing is not modelled yet (a first-use miss found at the router
  has only tens of microseconds).
- Not free: every streamed byte is also an HBM write (~0.5-2.5% of HBM time at C1-C8, ~5% at C16), and host traffic
  (Engram lookups, KV offload) needs priority over the stream.

## 7. Closed by evidence

Variable-length entropy coding of the codes (no decoder within 12x of stream rate); the hardware decompression engine
(0.16-0.6 TB/s, and the codes do not compress); rotated trellis and VQ codes; 2:4 sparse FP4; INT8 (no INT8 MMA on
sm_103a); expert merging and low rank; usage-driven bit allocation; a Grace-CPU compute tier. FP4 x FP4 at prefill is
3x per MMA by the ISA but 1.24-1.36x measured on a fused MoE on B200: unpriced here.

## 8. First experiments (each with a gate)

1. KLD of simulated lattice noise by layer band (gate: layers 20-39 at 0.11 relative error cost <= +0.012).
2. FP6 dense KLD on the official FP8 path (gate: <= +0.003).
3. C2C exposure: cold experts in registered Grace memory, default order vs cold-first (first confirm TMA reads a host
   address; gate: <= 10 us exposed for one cold expert). This decides whether `solo` is real.
4. Routing-count shadow log on the served engine; cold-deferral KLD at 1-2% cold share.
5. TC-FP4 decoder microbenchmark (gate: >= 6 TB/s of code bytes with the MMA running, no clock drop at the C16 shape).

## 9. References

- DeepSeek, Engram: Conditional Memory via Scalable Lookup, arXiv 2601.07372 (n-gram tables prefetched from host
  memory because their addresses depend only on input tokens).
- MoE-SpeQ, arXiv 2511.14102; SP-MoE, arXiv 2510.10302 (speculative decoding with expert prefetch).
- SiDA-MoE, arXiv 2310.18859; fMoE / FineMoE, arXiv 2502.05370; ST-MoE, arXiv 2606.15453; ExpertFlow, arXiv
  2410.17954 (expert prediction for offloading).
- QTIP and successors (trellis-coded weights); Any-Precision LLM and Matryoshka quantization (nested / bit-plane
  precision) — to be checked when the formats are built.

## 10. Implementation status (2026-11-14, measured)

**Dropping cold experts does not work** (decode-path / prefill-path KLD against the official reference,
leave-one-window-out expert ranks): forcing the router past the coldest 12.5% of expert slots costs KLD 0.120; 17.5%
costs 0.153. **A naive 3-bit cold tier is too lossy**: the coldest 22.5% of slots with the mantissa bit dropped costs
0.066 on its own. **TMA reads Grace memory at full link speed** from pinned, registered or pageable memory (348 / 339 /
331 GB/s with all SMs, ~300-340 with 16 SMs); VMM-mapped host pages only reach 91 GB/s.

**Exact mode** (`DERPSEEK_MODE=exact`). Per layer 192 static experts (ranked max-min: the highest lowest pick share over
the calibration families) and 96 cache rows in HBM; every non-static expert keeps an exact copy in pinned host memory
(144 GB). MegaMoE (fork in `engine/overlay/megamoe_fork`) reads host rows through per-CTA tensor-map clones whose
global address is the host base; the rows are numbered host first, and a host lane in the scheduler sends host-row
tasks only to the last 4 two-CTA clusters, so link-bound tiles never hold more than 8 SMs while the rest stream HBM.
Standalone (tests/step_cost_test.py, shared_lane_test.py): bit-identical to the all-HBM call for T = 1..1024; extra
call time for k distinct host experts at T=32: 0 / 0 / 0 / +23 / +151 us for k = 1 / 2 / 3 / 4 / 6 (no lane:
+41 / +89 / +148 / +198 / +292). The residency manager (`residency.py`) promotes host experts that get picks into
the least recently used cache row between steps (remap first, copy on a side stream, map in when landed).

Served (lil quick cells, 30 s, 0 context, GB300 only):

| profile | C1 | C8 | misses (share of non-static picks) | decode KLD |
|---|---|---|---|---|
| exact, 256 static + 32 cache (count-ranked) | 305-319 | 941 | 9.5% | |
| exact, 192 static (max-min) + 96 cache | 340 | 1,075 | 2.1% | lossless |
| exact-fast, 192 + 96 (+ singleton drop-2, MXFP8 head) | 374 | 1,335 | 1.0-1.4% | 0.028 (8,160 rows) |

Per-shape GPU step (exact-fast): 1x1 5.70 ms, 1x3 6.29 ms, 8x3 12.59 ms; dual-GPU dsv41 default 5.9-6.0 / 6.1-6.7 /
11.6-14.8 ms. What the first served build lost and why (worth knowing before touching the integration again):
- a bare `(*a, **k)` wrapper around `deep_gemm.fp8_fp4_mega_moe` hid `shared_l1_weights` from vLLM's signature
  check, which disabled shared-expert fusion and with it dsv41's router path on every layer (~2 ms per C1 step);
- the scheduler's fair first wave deadlocked when a call was mostly host experts (only the lane may claim them);
- the clone table was shared by all layers while a call writes it before its dependency wait (PDL).

**Precision mode** (`DERPSEEK_MISS=skip`, profile `skip-fast`): in layers 20-38 a host-row pick is dropped for that step
(counted, so it is promoted) instead of read; decode-sized calls only. Not yet measured.

The exact-fast server passed dsv41's release checks (README). Prompt reading runs at 34-39K tok/s for 32K-131K
prompts (dsv41: 75K): a prompt chunk reads most host experts over the link.

Measured after 1.0 (lil-bench standard/1 and per-shape GPU step periods):
- `exact-fast-kv` (--gpu-memory-utilization 0.98 instead of 0.96): KV cache 1.44M -> 3.64M tokens (3.27 -> 8.26 GiB),
  every cell of the matrix runs (16 requests at 128K: 1,759 tok/s); decode 1-5% lower than the 0.96 run.
- Compressible scale banks (dsv41's a12_retire ported to exact mode, `DERPSEEK_A12=1`): batch 1 x 3 drafts 6.52 ms
  (6.29 without), batch 8 x 3 12.49 ms (12.59 without); off by default.
- `skip-fast` (precision mode): decode KLD 0.041 with the residency manager ticking in the harness (0.221 with the
  cache frozen); C1 338.9 / C8 1,297.2, batch 1 x 3 6.55 ms (exact-fast 6.29): the skip's three extra kernels per layer
  (and MegaMoE no longer starting on the router's ready flag) cost more than the ~0.03 ms of misses it removes, and
  acceptance drops (2.28 vs 2.42 at C1). Worth doing only inside MegaMoE's dispatch, and where misses are frequent.
- 16K-token prefill chunks (at 0.98): the KV cache falls to 1.64M tokens and the engine hits an illegal memory access at
  the first long prompt (dsv41's components assume steps of at most 8,192 tokens); not pursued.
- 32K-token prefill chunks at 0.975 do not fit (the profiling run's activations need ~5 GiB more than the KV cache had).

- No-peer combine in the seam (`DERPSEEK_SEAM_COMBINE`, T <= 8) and R1 (MegaMoE starts on the router's ready flag):
  batch 1 x 3 / x 4 drafts 6.25 / 6.40 ms (6.29 / 6.44-6.49 without), C1 cell 377. dsv41 had closed R1 because its gain
  moved into the RTX ack wait; with no peer it stays.

Must decode layers synchronize? (Fable round 19, CPU port of the official forward, user's question 11-14): not
strictly, but every late join is approximate (each nonlinear read of the residual streams must see every additive
term), and exact slack is scheduling, mostly used already. Prices per 1% of picks in layers 20-38: joining at the next
FFN-side seam 0.0007-0.0009 (it needs its own post-mix coefficients), carrying into the next MoE call 0.008, skipping
0.013-0.016; four to five times more in layers 0-19. A late join moves bytes in time, so it pays only where a resource
is idle (the C2C link here). Its ranked list for Derpseek: prefill with host experts streamed ahead (exact), finer
hand-offs (the two above), the verify width chosen on the device instead of the host (+1.7-3.8% at C1, estimated),
rank-gating the skip exit (~3x cheaper), and a pre-attention route predictor (90% top-6 recall in layers 20-39).

Next: prefill (staging a layer's host experts during the layers before it); skip-fast KLD and speed
(the KLD harness needs the residency manager ticking: `derpseek_engine/harness_tick.py`); more HBM rows (2-bit scale
deltas, the drafter's experts, memory headroom); the full lil-bench standard/1 matrix (C16, 64K and 128K contexts).
