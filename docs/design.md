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

Profile names: `lossy-fast` was called `exact-fast` before 1.4 (it is not exact: singleton drop-2 and the MXFP8
verify head change the output distribution slightly).

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
| lossy-fast, 192 + 96 (+ singleton drop-2, MXFP8 head) | 374 | 1,335 | 1.0-1.4% | 0.028 (8,160 rows) |

Per-shape GPU step (lossy-fast): 1x1 5.70 ms, 1x3 6.29 ms, 8x3 12.59 ms; dual-GPU dsv41 default 5.9-6.0 / 6.1-6.7 /
11.6-14.8 ms. What the first served build lost and why (worth knowing before touching the integration again):
- a bare `(*a, **k)` wrapper around `deep_gemm.fp8_fp4_mega_moe` hid `shared_l1_weights` from vLLM's signature
  check, which disabled shared-expert fusion and with it dsv41's router path on every layer (~2 ms per C1 step);
- the scheduler's fair first wave deadlocked when a call was mostly host experts (only the lane may claim them);
- the clone table was shared by all layers while a call writes it before its dependency wait (PDL).

**Precision mode** (`DERPSEEK_MISS=skip`, profile `skip-fast`): in layers 20-38 a host-row pick is dropped for that step
(counted, so it is promoted) instead of read; decode-sized calls only. Not yet measured.

The lossy-fast server passed dsv41's release checks (README). Prompt reading runs at 34-39K tok/s for 32K-131K
prompts (dsv41: 75K): a prompt chunk reads most host experts over the link.

Measured after 1.0 (lil-bench standard/1 and per-shape GPU step periods):
- `exact-fast-kv` (1.1-1.3: --gpu-memory-utilization 0.98 instead of 0.96): KV cache 1.44M -> 3.64M tokens (3.27 -> 8.26 GiB),
  every cell of the matrix runs (16 requests at 128K: 1,759 tok/s); decode 1-5% lower than the 0.96 run.
- Compressible scale banks (dsv41's a12_retire ported to exact mode, `DERPSEEK_A12=1`): batch 1 x 3 drafts 6.52 ms
  (6.29 without), batch 8 x 3 12.49 ms (12.59 without); off by default.
- `skip-fast` (precision mode): decode KLD 0.041 with the residency manager ticking in the harness (0.221 with the
  cache frozen); C1 338.9 / C8 1,297.2, batch 1 x 3 6.55 ms (lossy-fast 6.29): the skip's three extra kernels per layer
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

- Layer split (same 11,520 HBM rows, every layer 480 kernel rows and 96 cache rows; only the static / host boundary
  moves): split A (layers 0-19: 224 static, 20-39: 160) lil prefill 62.7K / 69.9K tok/s (192 everywhere: 56.2K / 63.0K),
  release-check prompts +12-24%, decode unchanged within noise -> the default. Split B (256 / 128) prefill 69.2K /
  74.7K (dsv41 75K), decode -3-4% -> profile `lossy-fast-prefill`. Fable round 19's item 1 (host experts of layers 0-19
  resident or streamed ahead for long prompts) would close the rest without the decode cost.

- Split C (layers 0-19: 224 static + 128 cache, 20-39: 96 + 128; the same 11,520 HBM rows, a larger adaptive cache):
  lil prefill 70.6K / 75.0K (split B's), decode C1 364 / 362 / 359, C8 1,314 / 1,250 / 1,268, C16 1,870 / 1,786 /
  1,823 (0 / 64K / 128K context): between A and B, down to -4.7% at C16 64K. A bigger cache does not replace static
  rows for decode under concurrency. Not adopted.

**Prompt reading depends on the text** (1.4, per-layer CUDA events around every prompt-chunk MegaMoE call,
`DERPSEEK_PREFILL_PROF=1`, `derpseek_engine/prefill.py`; probe: fresh 32K prompts, max_tokens 1). lil-bench's prefill
text cycles 20 sentences: once the residency manager has promoted what it uses, a chunk reads ~3 distinct host experts
per layer (layers 0-19) and the MoE takes ~2.3 ms per layer. Varied text reads 30-58 per layer (MMLU-Pro questions
~37, Python source ~36, most in layers 0-5) and the MoE takes 4.6-7.5 ms per layer: 33-37K tok/s against ~54K for
lil's text, and a fresh server reads even lil's text at ~35K on its first prompt. (Freezing the residency manager
during prompts makes it worse: lil's text 25K.)

**Prompt staging** (1.4, `DERPSEEK_STAGE=1`, default in `exact`): for a prompt chunk (T >= 1024) each layer of 0-19
reads its host-mapped experts from an HBM staging slot instead of host memory. Two slots of 68 rows (2.4 GB, reserved
at load so vLLM sizes the KV cache around them: 3.66M -> 2.60M tokens). When layer l's MoE call is enqueued, the copy
of layer l+2's host-mapped rows into the slot layer l just used goes onto a copy stream (after that slot's last
reader): one `cudaMemcpyBatchAsync` per layer (0.14-0.2 ms of CPU, ~350-366 GB/s; 256 single copies took 2.2 ms of CPU
and ran at ~290 GB/s, and that CPU time made the GPU wait). The call points control words 2-9 at the slot and turns the
host lane off (word 10), sends the picks of host-mapped experts to their staging positions, then restores the words
and credits the per-row counts back to the true host rows (the residency manager keeps learning from prompts). Layers
0-1 are copied during the previous chunk; when the residency manager has moved a few of their experts since, the slot
is topped up with them instead of dropped. Bit for bit equal to the host-row and all-HBM calls for T = 1..4096
(`tests/stage_test.py`). The main stream waits 0.008 ms per staged layer for its copy. Probe, same boot (exact):
lil's text 41.1K -> 53.1K (that boot's baseline was low; another boot read it at 53.6K without staging, so about
level), MMLU-Pro 24.2K -> 42.1K, Python source 28.2K -> 42.7K, MMLU-Pro at 128K 40.1K -> 48.1K.
Final code (top-up reusing freed positions; same boot, staging switched live, after a lil run): lil's text 61.0K ->
62.6K, MMLU-Pro 38.8K -> 50.7K, Python source 40.7K -> 53.1K, MMLU-Pro at 128K 43.4K -> 58.9K (+31-36% on varied text;
this boot's baseline was higher than boot 31's, so the figures above overstate the ratio). No stale-slot fallbacks
(~420 top-ups per 30 s window), 0.03-0.04 ms of copy wait per staged layer, copies at 373-375 GB/s. Release-check
prompts (dsv41 bench.py): 32K 43.5K -> 57.0K, 131K 48.2K -> 63.6K tok/s; needles at 129K-975K read at 60-84K tok/s end
to end (1.3: 44-64K). Overlapping the next layers' weight loads with the current layer's compute is FlexGen's (Sheng et
al. 2023, section 4.2).

`exact` 1.4 with prompt staging, lil-bench standard/1 (full matrix, GB300 only, final code): decode C1 330.4 / 326.5 /
326.5, C8 1,077.9 / 1,038.8 / 1,042.4, C16 1,529.3 / 1,505.8 / 1,469.9 (0 / 64K / 128K context); prefill 66.2K (32K) /
70.3K (128K); KV cache 2.60M tokens (a first run before the top-up fix: C1 333.5 / 317.1 / 325.9, C8 1,060 / 1,039 /
1,049, C16 1,521 / 1,453 / 1,471, prefill 66.0K / 70.8K). lil's prompt text is repetitive, so staging moves its prefill number little (1.3
lossy-fast without staging: 64.7K / 70.7K). `exact` reads every picked expert; `lossy-fast`'s singleton drop-2 is
worth +11% at C1 and +24% at C8 (1.3: 372.8 / 1,337.8 / 1,888.0). In that run layers 0-1 still fell back to host reads
on most chunks (their slots, copied during the previous chunk, had more residency moves than the 4 spare rows held); the top-up now reuses the positions of staged experts that were mapped into cache rows meanwhile.

**`exact` at 8-16 requests** (1.5; quick lil cells, 30 s, no context, the same boot conditions; three `exact` 1.4
controls: C1 332, C8 1,075-1,080, C16 1,516-1,548): `exact` ran the MoE glue on the stock path above 24 rows (the W4
router, its masked verify rows and R1 only to 24 rows) and stock steps above batch 8, both of which `lossy-fast` lifts
and which dsv41 lists as speed-only (Fable round 20, item 0). The W4 router to 64 rows (bit-exact with the stock
gate): C1 334, C8 1,140, C16 1,552; plus whole-step graphs to batch 16: C1 341, C8 1,147 (+6.4%), C16 1,666 (+8.5%).
Drafts per request at batch 5-32 (dsv41's 3 was never swept; 4 and 5 read many more experts per step): 2 drafts C8
1,109 / C16 1,561 (acceptance 2.09 against 2.27-2.38), 4 drafts 1,029 / 1,149, 5 drafts 964 / 1,030. The batch-1
32-row copy verify keeps the stock path (DERPSEEK_WIDE_B1_STOCK: wide calls take the W4 router only as masked verify
steps). Misses are 0.08 ms of an `exact` 8x3 step (Fable round 20), so the rest of the gap to `lossy-fast` is its drop.
1.5 `exact` full lil matrix: C1 336.7 / 331.9 / 326.5, C8 1,140.7 / 1,071.3 / 1,086.7, C16 1,661.1 / 1,633.5 / 1,616.1
(0 / 64K / 128K context; 1.4: +2/+2/0%, +6/+3/+4%, +9/+8/+10%), prefill 66.4K / 70.6K; release checks pass (GSM8K
196/194 of 200, needles to 975K).

After 1.5 (quick lil cells unless noted): the per-request verify width lambdas (dsv41's K3: 0.25 at batch >= 2, 0.3 at
batch 1) stay: 0.15 / 0.20 cost C8 -5%, 0.35 / 0.40 and 0.50 / 0.50 are within noise (C8 1,129-1,141, C16
1,654-1,699 against 1,139-1,153 / 1,665). **Unrelated concurrent streams** (`mixed_bench`: each request continues a
different 1,024-token slice of the KLD corpus, wiki / code / chat / math / Chinese / long, 768 tokens, temperature 1)
are where `exact` loses: 8 streams 641 tok/s aggregate, 16 streams 702-739 (lil C8 1,140 / C16 1,673 in the same
boots), with 12-15% of non-static picks missing (600-1,500 host picks per step): the C2C link, not the SMs, is the
limit. The host lane at 6 clusters changes nothing; at 8 clusters 16 unrelated streams give 771 (+7% over the two lane-4
runs, which differ by 5%), lil cells unchanged; the default's own 16-stream runs span 668-767, so that is noise, not a
gain. The residency manager under the same traffic (live tune file): promotions off 491 tok/s (33.6% of non-static picks
miss), fewer (promote_tick 8) 736, more (48 per tick, 8 per layer) 719, min_misses 3 706, default 668-767: the
defaults stay. For `exact`, unrelated traffic is bound by the link; only fewer bytes over it (the lossy profiles) moves it.

**The cut** (`DERPSEEK_MISS=cut`, Fable round 20's v0 of `wavefront-fast`): per decode call in layers 20-38, the
distinct host-row experts beyond a window (1 / 2 / 3 / 5 experts at up to 8 / 16 / 32 / 64 rows) are skipped for the
step, weakest first by the summed rank price of their picks, never an expert holding one of a token's two strongest
picks, and at most 4% of the band's picks per step (beyond that the call waits, as in exact). Cut picks are counted,
so the residency manager promotes them. Decode-path KLD with the residency manager ticking (8,160 rows): on
`lossy-fast` options 0.0296 (0.0281 without), on `exact` options 0.0295 (the engine's floor is ~0.021: `exact` keeps the
weak picks that `lossy-fast` drops, so it has more host picks to cut); `skip-fast` (every host pick of the band skipped)
measured 0.041. So the cut belongs on `lossy-fast`. Served form: one 512-thread kernel between the router and MegaMoE
(`derpseek_engine/cutk.py`, agrees with the tensor-op rule on 300 random calls, capture-safe); MegaMoE, launched after
it, reads the cut indices. A bit-plane base does not fit: 2-plane bases for a band layer's 224 non-static experts need
2.23 GB against its 1.80 GB cache (Fable round 20).
`wavefront-fast` (lossy-fast + the cut), quick cells against lossy-fast in the same session: C1 362.6 / 363.5, C8 1,328.9
/ 1,335.4, C16 1,898.5 / 1,852.9; unrelated streams (`mixed_bench`, 90 s) 8: 836 / 813 (+3%), 16: 1,040 / 951 (+9%).
With a 10% budget instead of 4%: 845 / 1,087 (+1% / +4.5% more), but the harness, which verifies one request at a
time, cannot price a budget that binds only under heavy traffic; Fable's model puts 4% near +0.016 at 16 unrelated
streams, so 4% stays.

**`deferred-fast` is blocked** (Fable round 20's D0): a late lane would be a second MegaMoE call holding 8 SMs while the
step goes on. Holding 8 SMs with a spinning 2-CTA-cluster kernel during the whole-step harness hangs MegaMoE in its fair
first wave ("target 76 claimed 72": the call is sized for all 76 SM pairs and waits for every cluster), both in the prompt
prefill and in the decode replays alone. A lane would need MegaMoE at 144 SMs in every decode graph and the step's other
full-width kernels to tolerate a second wave, for the +1-3% Fable expects under agent traffic. Shelved; the cut takes the
miss case without a lane.

Quality on `exact` 1.4 (lil-bench client): GPQA Diamond 90.4% (179/198, 262K-token cap, 3 truncated; lossy-fast 1.1:
92.4%), tool-eval-bench 89/100 (lossy-fast 1.0: 91/100). Both within run-to-run noise of the lossy-fast runs.

Served runs are not bit-reproducible run to run (greedy outputs and first-token logprobs differ between two identical
requests in one boot, residency frozen or not), so staging's exactness is shown on the kernel, not by greedy text.

Next: the cut inside the router kernels (no extra launch; Fable round 20's v0 proper), split-B / H3 rowmaps under the
cut (prompt-side layers resident, KV back), more HBM rows (2-bit scale deltas, the drafter's experts).
