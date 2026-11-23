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

## 10. Implementation status (2026-11-22, measured)

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

**Precision mode** (`DERPSEEK_MISS=skip`, profile `skip-fast` until 1.6): in layers 20-38 a host-row pick is dropped for that step
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

**`deferred-fast`: D0 as run did not test a lane.** Fable round 20's D0 held 8 SMs with a spinning 2-CTA-cluster
kernel during dsv41's whole-step harness: MegaMoE hung in its fair first wave ("target 76 claimed 72"), both with the
hog held through the prompt prefill and with it held only from the first decode replay. In that harness MegaMoE ran on
all 76 SM pairs (kNumSMs 152), and its fair first wave waits until every cluster has a task, so a kernel holding SMs
leaves clusters unscheduled and the call waits for them; the hog never ended, so it hung. The served decode path
differs: with the FFN-side mix off MegaMoE's path (dsv41 E3, `DSV41_MIX_OFFPATH=1`, the default), boundary-path
MegaMoE calls run at kNumSMs 144 (72 clusters) and leave 4 SM pairs to the mix, so a lane of up to 8 SMs can run beside
decode MegaMoE (competing with the mix), and a lane confined to the window from MegaMoE(l)'s end to seam F(l+1) never
meets MegaMoE at all, as dsv41's `engram_side` side GEMM does. Prompt chunks run MegaMoE at 152. Reopened: Fable round
21, and a probe in the served engine.

Quality on `exact` 1.4 (lil-bench client): GPQA Diamond 90.4% (179/198, 262K-token cap, 3 truncated; lossy-fast 1.1:
92.4%), tool-eval-bench 89/100 (lossy-fast 1.0: 91/100). Both within run-to-run noise of the lossy-fast runs.

Served runs are not bit-reproducible run to run (greedy outputs and first-token logprobs differ between two identical
requests in one boot, residency frozen or not), so staging's exactness is shown on the kernel, not by greedy text.

**D0' in the served engine** (`derpseek_engine/lane.py`, 11-15): a finite kernel on 8 SMs (four 2-CTA clusters of
1,024 threads and 200 KB of shared memory, as a MegaMoE instance) reading pinned CPU memory, forked in every band
layer, its work switched by the tune file inside one boot, every graph replay timed with CUDA events. Plain loads read
135 GB/s on 8 SMs (MegaMoE's TMA lane ~350), so the read size sets how long it holds the SMs. Step time against
probe off, C1 graphs / C8 / C16:
- forked after MegaMoE(l), joined when attention(l+1) returns: 35 us held +0.2-1.2% / 0 / 0; 70 us +5.5-7.6% / 0 /
  0; 140 us +21-29% / +8.7% / +4.4%. The window between MegaMoE(l) and seam F(l+1) holds ~40-45 us at C1, 70+ above;
- forked after router(l) and the off-path mix (beside MegaMoE(l) on the 4 SM pairs the 144-SM call leaves), joined
  when attention(l+1) returns: 70 us +0.2-2.2% / +0.6% / 0; 140 us +2.5-14% / +0.8% / 0. Nothing stalls;
- the empty fork / join (probe off) against a boot without it: +0.5% at C1 (router fork), +0.8% (after MegaMoE).
Fable round 21: D0 had hung on the drafter's MoE call, which takes all 152 SMs; decode MegaMoE is a 144-SM call.

**Fast promotion** (since 1.7; the only path since 1.8, the A/B switch removed): a promoted expert is mapped in from the copy
stream right behind its copy (a stream memory write of its row-map word) instead of at the next tick on the main
stream, so it serves a step sooner; the bytes are the same. On `exact`, one boot, tune toggles, two runs per cell: C1
+0.6% (noise), C8 1,141 -> 1,182 (+3.6%), C16 1,670 -> 1,745 (+4.5%), unrelated streams 8: 613 -> 637, 16: 708 -> 756.
Keeping 2 pre-evicted slots per layer as well (`free_ahead`) was no better. The first A/B showed a residency
accounting race: the per-step count snapshot was a host copy on the copy stream, which ran whenever that stream got to
it, sometimes inside the next step's prompt chunk, where staging's count credit briefly raises and restores some
counters, so the next delta went negative and wrapped ("misses 100%" lines, promotions driven by garbage). The
snapshot is now a device copy on the main stream between steps. Share of host-row picks on rows already being
promoted, with the fix: lil C1 17-22%, unrelated streams 6-9% -> 4% with fast promotion.

**`deferred-fast` (closed in 1.8 by the pinning bench below; `derpseek_engine/defer.py` stays for `=wave`)**, Fable
round 21's design: in the band layers of decode
calls at 1-8 and 17-64 rows (not 9-16: the attention kernel is a cooperative launch there), every host-row pick moves
to a late lane (`cutk.late`, counted as a miss), a second MegaMoE instance at kNumSMs 8 with no shared expert and its
own symmetric buffer and control block, forked after the router and the off-path mix; at the next attention-side seam
the seam's post-only mode adds X2 = X + post_F(l) (x) y_late on the lane's stream, and the next FFN seam waits for it.
Eager forwards (the decoder layer drops its mHC stream outside FULL graphs) join on the stock seam path. Decode-path KLD
(TICK = 1, 8,160 rows): 0.0297 (lossy-fast 0.0284, the cut 0.0296). Speed (fast promotion off in that boot): lil C1
364.5, C8 1,346.9, C16 1,883.4; unrelated streams 8: 808, 16: 950: the same as `lossy-fast` (813 / 951), below the cut
(836 / 1,038). The link is the limit: under unrelated streams a band layer needs 20-40 host experts per step, 1-2 ms
of link, and the late window hides one or two. Its use is as the LATE class in front of the cut (Fable's `=wave`).

Quality, 1.6: lil-bench's `estonia` (134,187-token packet, 30 samples at 30 at once, temperature 1) and
`hotel-lights` 30/30 on `exact`, `lossy-fast` and `wavefront-fast`.

**The pinning bench (1.8, `bench/deb`)**: 11 kinds of text (held-out varied windows, cold-heavy, ordinary, degenerate),
KLD against `exact` at batch 16 (decode path, full routes) and serving per kind (prompt reading at 32K, decode with 8
conversations of one kind, decode-only miss share). KLD, all kinds (CJK in brackets): noise (`exact` again) 0.0168
(0.0176); varied rowmap 0.0167; `lossy-fast` 0.0279 (0.0309); `wavefront-fast` 0.0348 (0.0505); its 2% step cap
(`DERPSEEK_CUT_STEP=0.02`) 0.0305 (0.0398) but 7% slower on ordinary text; `CUT_NOW=2,2,3,5` 0.0361; `deferred-fast`
0.0354 (0.0462) at `lossy-fast`'s speed or below (ordinary 693 against 707, CJK 885 against 922): closed. Decode with 8
of a kind: `lossy-fast` +10-26% over `exact`; `wavefront-fast` within a few percent of it, +9% on ordinary text, -5% on
CJK. The varied rowmap on `exact`: +2-8% on most kinds (CJK +8%, rare code +7%, structured +7%), -4% degenerate,
prompt reading +1-3%.

**Fable round 22 (2026-11-15, five consults on `exact`; plan in the scratch notes)**: the binding resource under
unrelated streams is CPU-memory bandwidth through the host lane (289 GB/s, 79% of a 16-stream step), and 94-99% of
host reads are re-reads of a working set larger than the 96 cache rows. Agreed: a GiB of HBM as static rows in layers
20-39 is worth about +2% at 8-16 unrelated streams (nothing on lil); K3's batch lambda 0.25 -> 0.4-0.5 is worth +3-9%
there (distribution-exact); 2-bit scale codes free ~525 rows (+16-20% unrelated, C1 +1.9%; 2-3 weeks of kernel).
Closed: Grace-CPU expert compute (72 cores only match the lane, on the same memory), lossless coding of the FP4 codes,
units finer than an expert, expert bundles, row-split waves (0.3-0.6% of rows need no CPU-memory expert under
streams), time-slicing streams, write-through as a TMA store (the padded 4-bit shared-memory type is load-only).
Measured: the repeat share (of the distinct host experts a decode step reads, the share the step before also read;
new residency-line field) is 28-37% at mixed 8 / 16 and 13-18% on lil cells, the low-reuse case: write-through in its
copy-first form would add 1-3% over a manager rule, so it is closed. Prompt reading on varied text loses 20-30 ms per
8K chunk to host work per staged layer (copy plans rebuilt when promotions move a staged layer's set), not to the link.

**The overnight A/Bs (1.8)**, all on `exact`, with the measurement lesson first: identical serving runs spread 3-7%
(mixed traffic up to ~10%: a setting that cannot act at 8 conversations gave the best 8-conversation figure), so 2-5%
effects need the teacher-forced decode harness (`bench/deb/run_kld.sh`, batch 16, 240 rows: the same tokens in every
arm, 2.6% between identical runs, the KLD as a gross-error check).
- Rowmaps: the varied-text max-min map is ~3.5% faster on the harness (1,002 -> 967 window-seconds over five runs) and
  the same served (lil C16 +1.9%, the bench and mixed traffic within noise): `exact`'s rowmap in 1.8. Round 22e's
  mixed-traffic blends (85 / 70% max-min) were worse served (mixed 16: 686 / 683 against 734-752).
- Row value (round 22d's curve): 80 cache rows everywhere (-12 GiB) costs -17% / -22% at 8 / 16 unrelated
  conversations, as predicted (-16% / -21%); 8 more static rows in layers 20-39 (+3 GiB) +6% / +1%. A GiB of HBM as
  rows is worth ~1.5-2% there.
- Cache policy: the short-memory rate rule (24 / 48 / 96 copies per tick) reads more experts from CPU memory than
  least recently used (15-17% of non-static picks against 13-15%) and gains nothing beyond noise: closed. A copy-engine
  copy beside a saturated host lane takes the link from it, more than 1:1 (the lane drops from ~408 to ~99 GB/s against
  copies at 236-333 GB/s): promotions and copy-engine prefetch compete with in-call reads.
- Verify width (K3's batch lambda 0.40 / 0.55, 2 drafts at batch 9-32): inside the noise. Open until a lower-noise
  mixed test exists (the harness fixes the drafts).
- Prompt reading: counting prompt chunks makes promotions reshuffle the staged layers (staging copy issue 0.51-0.63 ms
  of engine-thread time with 313-397 top-ups per 30 s, against 0.21-0.29 ms and 0-35 without; ~43 against ~17 ms from
  one chunk's last MoE call to the next chunk's first). Not counting them (`prefill_t` 1024) reads varied prompts
  8-12% faster and matches the all-resident ceiling for layers 0-19, so block lending would add little. Ring-light
  (`light` 1: staged layers frozen during prompts; 2: only the layers a prompt stages) keeps +0-7% on varied text but
  costs lil's prompt 4-6% (its few host experts, read in-kernel, are promoted for its later chunks): off by default.
  The ~17 ms left between chunks (against ~2.5 ms between layers) is GPU idle time.
- The in-call window in graph replays (`tests/step_cost_test.py` GRAPH=1): at T = 4 even the first host expert costs
  24 us; two are free at T = 32, four at T = 64. The eager loop's "two free at T = 4" was its own floor.
- Tail lane widening (a development fork): bit-identical, but 12-22% slower with 8-16 host experts; a static
  36-cluster lane is slower too (73-77 against 61-65 us per expert): the memory side limits the lane. Closed.
- 2-bit scale codes (Fable round 23a; the same family as b12x's MXFP4-CSF and local-inference-lab's lossless-CSF
  checkpoint): a base byte per output row and a 2-bit delta per UE8M0 scale, 0.007% of rows in a per-layer exception
  store; an expert row shrinks 4.26% (8.6 GiB = 512 more rows). Three kernel designs, all bit-identical in every test
  (uncoded and coded rows, T = 1-8,192, 8 / 144 / 152 SMs); speed at T = 64 against today's call: 1.45x (the expansion
  on the MMA warp), 2.2x (a scale warp waiting on its loads), 1.85x (a stream of code words; the scale warp still works
  ~365 cycles per K block, ~5.7 cycles per instruction against the stock MMA chain's 2.3); prompt chunks 1.36-1.56x
  (a stock prompt-chunk call delivers a K block every ~420 cycles per cluster, so the expansion alone costs about the
  whole call). Not shipped; software expansion closed. The rows are worth it (above).

**6-bit scale codes (1.9; Fable round 23a's fifth pass, `derpseek_engine/sf6.py`, the fork's namespace
`derpseek_sf6`)**: every routed UE8M0 weight scale of the checkpoint lies in 117..127 (the shared expert's in
114..121). Each is stored as `scale - 96` in 6 bits, 16 codes in 12 bytes (per expert, 128-row tile and K block: four
rows of 96 bytes), in HBM rows, cache rows, host rows and staging slots alike, so every copy stays a whole-row copy. A
TMA of the 6-bit type (`CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B`, box 128 codes x 4 rows: 384 bytes, a 4-byte gap after
every 12 bytes in shared memory) loads them, and the scale copy's unpack form (`tcgen05.cp ... .b8x16.b6x16_p32`) widens
them into tensor memory: nothing runs per K block. The MMA multiplies by 2^(weight scale + token scale - 254), so the 96
moves to the token scales: the pull adds it to each routed token's scale word, the first GEMM's epilogue to the second
GEMM's token scales, and with the all-token head (T <= 16) every SM writes a +96 copy of the token scales before the
routed tasks start; the shared expert keeps its own scales. The 4-bit unpack (`.b4x16_p64`, which would have freed 349
rows) places each value shifted left by two bits (PTX ISA Figure 198; the fourth pass checked it on the GPU), so it
cannot carry exponents. An expert row is 18,524,160 bytes (-1.47%): 2.97 GiB over the 11,520 HBM rows, spent as 8 more
static rows in each of layers 20-39 (+160, rowmap `exact-splitA-varied-sf6-mm.json`); the KV pool keeps its size.
- Exactness: bit-identical to the uncoded all-HBM call in every kernel test (single MMAs; whole calls at T = 1-8,192
  with host rows, the host lane, staging and the shared expert; 8 / 144 / 152 SMs), and the same KLD on the harness
  (against the pinbench's exact run: 0.0167 / 0.0170, against 0.0169 for 1.8 and 0.0168 for exact against itself).
- Kernel cost against today's call (served decode shapes in CUDA graphs): T = 4 +1.7 us at first (57.9 -> 59.5,
  five runs), almost all of it the GPU-scope `__threadfence` after the all-token head's copy: each SM's own TMA reads
  the copy (every SM writes the same words), so a `fence.proxy.async.global` suffices, and as shipped the cost is
  +0.5-0.7 us (58.4-58.5, four runs; prefetching the coded maps instead of acquiring them changed nothing: 59.2 / 59.4).
  Calls that read host experts: within ~1 us of today's at T = 4 (split 1 / 2 / 4: 84.1 / 145.5-146.6 / 273.3
  against 82.4-83.3 / 144.6 / 272.9-273.8 us); T = 32 and 64 the same or slightly less. Prompt chunks (T = 8,192, under the power cap): -4% to +10% depending on the routing
  draw (120 / 170 / 250 / 350 distinct experts, two runs each; the MMA warp's K-block counters unchanged; neither the
  plain scale copy nor dropping the +96 adds, timing-only variants, recovers more than half of the worst draw's gap).
  Served prompt reading is the same (32K: lil 63.3K against 63.1K, MMLU 50.7K / 51.5K, code 53.2K / 52.7K).
- Served against 1.8 (the same session, warm kernel caches): the teacher-forced harness 911.2 / 903.9 against 951.2 /
  938.9 window-seconds (-4.0%; median window -3.7% / -4.5%); mixed traffic 8 conversations 652-679 (four runs, two
  on the release server) against 598-648 (three runs; +7.4% on the means), 16 conversations 724-792 against 719-743
  (+5.0%); lil C1 / C8 / C16 330.5 / 1,171.9 / 1,716.7 against 331.9 / 1,174.2 / 1,715.4; host-row picks on the
  harness -3.9%. The KV pool takes the ~0.2 GiB left over: 2,682,981 tokens (2,602,934). The release check's speed
  sets: equal or slightly better (decode c=1 / 2 / 4: 401.6 / 506.7 / 702.4 against 399.3 / 501.7 / 687.3).
- Measurement lesson: the coded fork's MegaMoE variants compiled on first use inside measured windows in the first
  A/B (five in the harness's first batch, one in a mixed run); keep the kernel caches warm in both arms.

**The C1 push (1.10, 2026-11-22): where a batch-1 step goes, and what was tried.** Goal: lil C1 400 tok/s on `exact`
(1.9: ~331-338). Two consults with an outside model (Gemini 4 Carbon) proposed the levers; every one was measured.

- **Anatomy** (vLLM's torch profiler on the served engine, 365 steps of a lil C1 cell; nsys's capture stop deadlocks
  the engine): the GPU is busy ~6.96 ms per step and idle only 29 us median / 41 us mean between step graphs
  (`timing.py` now logs that gap), so the host is not the limit. Fair-share time per step: MegaMoE 3.0 ms (43 calls,
  ~88 us each), low-latency MXFP8 GEMMs 0.82, the mHC mix 0.76 (side stream; 3 us of it alone), wo_a 0.49, attention
  0.31 (16 us alone), router 0.29, seams 0.27, BF16 verify head 0.19 (at 6.8 TB/s, the roofline), drafter head 0.12,
  Markov loop 0.09. One layer at 4 rows is ~150 us: ~95 MegaMoE + ~55 dense chain. The step moves ~20-26 GB, i.e.
  ~3-3.8 TB/s averaged: busy but not bandwidth-saturated; the dense chain is a sequence of ~10 dependent small kernels
  per layer. Host-row misses at C1: ~10.6 picks and ~90 us per step.
- **Drafts.** Position-1 acceptance is 0.63-0.85 by kind of text while the target's mass on the drafter's top-8 is
  0.89-0.98 (top-32 0.94-0.99): an oracle that knew the target's probabilities over the drafter's top-32 would lift lil
  from 2.41 to ~4.6 tokens per step. Measured on a dump of 83K training steps (160 prompts, disjoint from the 48
  evaluation prompts; `derpseek_engine/acc3.py`, tools in `tools/draft_shape/`), scored on held-out kinds of text:
  a confidence-head tilted proposal +0.0-0.2%; short-suffix copy splices 0%; sibling candidates at position 1
  (recursive rejection, measured row costs) -3.0..+0.6%; an online per-request token bias -1.6..+0.3%; a low-rank
  correction of the drafter's hidden state before its head +0.6% on validation, -0.3..+0.2% held out (it memorizes
  without early stopping); a vocabulary shortlist for the drafter loses 1-13%; an MXFP4 drafter head saves its 60 us
  but loses as much acceptance. What generalizes is a **context-free reshape of the proposal** (18 coefficients over
  log q, -log q_max, five probability bins and the draft position, fitted on the training prompts): +1.2-2.3% on
  every held-out kind offline (lil +1.9-2.1%). It runs inside Derpseek's copy of the fused Markov kernel
  (`overlay/w6mk`, built by `derpseek.sh`): per position one more grid barrier for the softmax normalizer, then the
  reshaped logits are both drawn from and written to the q cache, so the rejection test stays exact (kernel tests:
  zero coefficients bit-identical to off; the cache equals a torch reshape; draws follow q'). Since it sharpens q,
  K3b would read inflated confidences (mean 0.64 -> 0.79; widths 3.66 -> 5.14 drafts); the kernel therefore hands K3b
  the drafter's own pre-reshape max and normalizer (`draftshape.py`). Served (two alternating pairs of boots, 3 lil C1 cells each): accepted tokens per step +1.4% (2.457 against 2.422), K3b's widths unchanged (3.61 against 3.64 drafts), tok/s +0.6% (337.3 against 335.4), inside the run-to-run spread: the extra barrier (~50 us per step) takes most of the gain. Both pairs were ahead, so 1.10 ships it on in `exact` (`DERPSEEK_DRAFT_SHAPE=` turns it off, and the Markov loop is then dsv41's own; the lossy profiles keep it off, unmeasured with their options).
- **Kernel-side levers, measured and closed:** a device-side width switch (conditional graph; ceiling = the 0.6% idle;
  on driver 13.04 SWITCH bodies take cooperative and cluster kernels but any memcpy node segfaults); W1's split-KV
  attention at <= 8 rows (-9% at C1: the bitwise-parallel kernel is already spread over the GPU); the mHC mix in
  MegaMoE's path (steps/s +0.6%, noise: the side-stream mix is not stealing the step); prefetching next-layer cold
  experts (ceiling ~1.3% at C1, the misses are rare). The dense GEMMs prefetch their weights under PDL: on the critical
  path q_a+kv_a takes ~3 us, q_b ~6.4 (6.5 TB/s), wo_b ~8.8; the headroom is ~4-5 us per layer.
- **Embeddings in Grace memory + 3 more static experts per layer in 20-39 (default in 1.10):** the target's input
  embedding table (1.23 GiB; the drafter shares it) moves to pinned host memory and is read in place (bit for bit the
  same rows; an 8K-token prompt chunk gathers its rows in 0.5 ms), and the room holds 3 more static experts in each of
  layers 20-39 (rowmap `exact-splitA-varied-sf6-s171-mm.json`, 171 + 96). Same session against 1.9: unrelated streams
  8 / 16 682 / 780 against 642 / 753 tok/s (+6% / +4%), C1 335 against 333, KV pool 2.76M tokens (2.68M).
  (+6 per layer gave 703 / 803 but cost 14% of the KV pool.)

- **After the release (11-23), the last three consult ideas, measured on a second dump** (`acc3t-exact`: 83K steps
  with every verify row's layer-20 state and the target's final hidden state):
  - *Mid-network row pruning* (pre-drawn acceptance tests, a probe at layer 20 drops later draft rows from layers
    20-39; exact by K3b's rule). The raw logit lens at layer 20 carries no signal through the hyper-connection mean; a
    ridge map to the final hidden state reaches a correlation of 0.21-0.41 with the final log p of the drafted token.
    Simulated with real p, q and uniforms (`tools/draft_shape/prune_sim.py`, 0.08-0.12 ms saved per dropped row): an
    oracle probe gains only +1.7-4% (lil 345 -> 352-355), the ridge probe loses 10-24%. Closed.
  - *Carbon's DrafterRerank* (a SwiGLU adapter 10,240 -> 1,024 -> 5,120 on [the drafter's state; the target's final
    normed hidden at the previous step's bonus row], scored against the candidates' full head rows, plus a per-position
    temperature; trained on acceptance with dropout, weight decay and early stopping; `train_rerank.py`; the linear
    version is `train_lora2.py`): training text +16%, held-out training requests at best +0.5%, held-out kinds of text
    0.0..+0.2%. Data scaling (25 / 50 / 100% of the 52K training steps): +0.15 / +0.30 / +0.52% on the held-out
    requests, ~+0.2% per doubling: ~+1-1.5% expected at Carbon's ~1M steps, not +10-14%. Closed.
  - *Prefetching CPU-memory experts (Carbon's pre-router DMA)*, measured as the cut kernel dropping every host pick in
    all 40 layers against the same kernel with a zero budget (timing only, the dropped picks change the text): step
    periods 7.08-7.10 against 7.20 ms at width 3, 7.31 against 7.41 at width 4, 7.60 against 7.74 at width 5: the whole
    C2C stall at C1 is ~1.2-1.8% of the step, the ceiling of any prefetch before its own traffic. Closed.
  - *Carbon's fused register-Sinkhorn mHC mix*: a redundant second mix behind every real one (`DERPSEEK_TIMING_MIX2=1`,
    outputs and routing unchanged) adds +0.05 ms at width 3, +0.12 at width 4, +0.18 at width 1 (~+0.08 ms, ~1.1%,
    weighted by K3b's widths): the mix's whole cost under contention is ~1-1.5%, the ceiling of any fusion, not the
    0.3-0.4 ms projected. (Skipping the mix instead, `DERPSEEK_TIMING_NOMIX=1`, corrupts the hidden states and with them
    the routing, so its step times are not comparable.)
  - *The third consult's and the remaining second-round ideas* (ab8, one session, C1 cells and per-width periods):
    L2 pre-staging of wo_b during attention, and of q_b as well (dsv41's `DSV41_WOB_PREFETCH` / `DSV41_WQB_PREFETCH`,
    rerun on Derpseek): steps/s -1.0% / -1.3% (dsv41 found the same). The draft reshape's extra barrier, measured with
    zero coefficients (bit-identical drafts): ~0.9% of the step (138.2 off, 137.0 zero, 137.0 on, steps/s), so a
    barrier-free top-32 variant could add at most that to the reshape's +1.4% acceptance. An entropy-gated two-way
    branch at position 1 at widths <= 3 (theta chosen on training steps; `sibgate.py`): 0..+0.7% held out. Grace
    CPU pre-drafting can only pay on steps where every verified draft is accepted: 9-20% of lil's steps at widths 3-4,
    so replacing the ~0.45 ms drafter tail there is worth <= 0.6-1.3%, with drafts as good as DSpark's (the CPU-side
    copy drafter measured 0%). The width rule asked for (argmax over W of expected tokens per step time from a per-width
    table) is K3b's rule; dsv41 replayed it with the confidence head as estimator (no gain).
  - *Built and measured live (ab9, ab10, alternating pairs):* K3b with per-position acceptance factors calibrated on
    the dump (0.86 .. 0.76 instead of the scalar 0.93; `derpseek_engine/k3cal.py`, `DERPSEEK_K3B_SHRINK_VEC`): widths
    3.64 -> 3.21 drafts, C1 337.6 against 335.3 (+0.7%, noise). The W3 stage split (`overlay/w3split`,
    `DERPSEEK_W3_SPLIT=1`: the fused qkv stage launched as its q path on the main stream, q_b's input, and its kv path +
    key table on a side stream joined after q_b; bit-identical to the fused launch at T = 1-8 on q, scales, kv and the
    cache): steps/s 137.1 / 136.9 against 137.4 / 136.7, width-3 steps -0.4%, width-4 +0.3%: the fork and join cost what
    the shorter q launch saves. Both stay off.
  - *Not built, with measured ceilings* (`chain_gaps.py` on the served profile): fusing the router into the wo_b /
    seam chain (dsv41's open P2-A; the router ends 17.5 us after wo_b, MegaMoE launches after 2.6 us and starts on the
    router's ready flag: ~4-7 us per layer, ~2-4%), folding all of W3 stage into the q_a+kv_a GEMM's epilogue (stage
    ends 4.2 us after it, on q_b's path: <= ~2.5-3.5%; the split above took the kv half off q_b's path for nothing), a pre-attention shadow router for prefetching CPU-memory experts (bounded by the whole C2C
    stall, 1.2-1.8%), pinning the Markov head in L2 (the Markov loop is 87 us, ~1.2%, and keeps its rows in shared
    memory after the first position already).

**Future work for `exact`**, ranked (1.10):
1. The router fused into the wo_b / seam chain (dsv41 P2-A): the largest measured single-stream ceiling left
   (~2-4%), multi-day kernel work in dsv41's W8 stage framework.
2. More HBM for static rows: the whole-step graph pool reserve (2.0 GiB, its use not yet measured under load), the
   vision tower (the user's call).
3. Splitting W3 stage: its KV half (RoPE, SWA insert) could run beside q_b (~2-3 us per layer); the q half feeds q_b.
4. The 6-bit codes' last ~0.7 us per call at T <= 16: the all-token head still copies the token scales with +96. The
   shared expert's scales (114..121) can carry -96 as well; the input quantization (vLLM's prepare kernel, a mounted
   overlay) would then write every token scale with +96 and the copy goes away (C1 ~+0.4%).
5. Prompt reading: a freeze rule that keeps lil's promotions (e.g. freeze a staged layer only once its set churns), the
   idle time between chunks (enqueue the next chunk's host work before the current chunk completes), BLOCK_M 240 for
   prompt chunks.
6. The lossy profiles with 6-bit codes (+160 rows each; not measured, so off there in 1.9).
7. Verify width under link pressure, with a low-noise mixed test.
8. The split tick (promotions a step sooner), predicted prefetch on the SMs (copy-engine prefetch competes with the
   lane), and a serving-time audit of the two storage paths before any kernel change ships.
Lossy, deprioritized: the late lane as a bench approach (`DERPSEEK_MISS=wave`), CUT_NOW 2,2,3,5 at batch 1, the 2% cut
cap, and the rare-experts causal skip runs.
