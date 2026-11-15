# Derpseek

Very fast serving of DeepSeek-V4.1-Flash-1017 on a single DGX Station GB300, built around the MegaMoE kernel.

## What is Derpseek

Derpseek serves DeepSeek-V4.1-Flash-1017 from one DGX Station GB300 with no second GPU: 337 tokens per second for a
single conversation and 1,661 for sixteen at once, with every expert computed exactly as DeepSeek ships it.

It is built around MegaMoE, DeepGEMM's mixture-of-experts kernel, which does a whole expert layer in one launch on all
of the GPU's cores. The model does not fit in the GB300's memory (about 285 GiB of weights against 250.7 GiB), so
Derpseek extends MegaMoE to read the experts that are not in GPU memory straight from the CPU's memory, inside the same
launch, and keeps the experts that matter in GPU memory so that this rarely costs time.

It runs on the [dsv41](https://github.com/chronarion/dsv41) engine. dsv41 serves the same model by putting the experts
that do not fit on a second card, an RTX PRO 6000; most DGX Stations do not have one.

## How fast is it

Local Inference Lab's [standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench`
1.7.1, `standard/1`), GB300 only: decode tokens per second with no prompt context, prompt reading in tokens per second
for a 32K / 128K-token prompt, and how far each profile's next-token probabilities are from the official model's (KLD,
lower is closer; the lossy profiles' figures are provisional, [see below](#about-the-kld-figures)). The profiles are
explained [below](#what-the-profiles-do).

| profile | 1 request | 8 requests | 16 requests | prompt reading 32K / 128K | KLD vs the official model |
|---|---|---|---|---|---|
| `exact` (default) | 336.7 | 1,140.7 | 1,661.1 | 66,434 / 70,634 | none added |
| `lossy-fast` | 365.7 | 1,333.5 | 1,906.9 | 66,240 / 70,307 | 0.028 |
| `wavefront-fast` | 358.3 | 1,323.1 | 1,875.5 | 65,980 / 70,160 | 0.030 |
| dsv41 `default` (lossless), GB300 + RTX PRO 6000, for comparison | 376 | 1,337 | – | 75K | none added |

1.7 makes `exact` faster with several requests at once, still exact: an expert copied into GPU memory is used from
the next step instead of a step or two later. In the same server with it off and on (lil-bench cells, two runs each):
8 requests 1,141 -> 1,182 (+3.6%), 16 requests 1,670 -> 1,745 (+4.5%), 1 request unchanged. The table above is 1.5's
full matrix; 1.7's follows.

`exact` with a long prompt already in context, decode tokens per second:

| prompt context | 1 request | 8 requests | 16 requests |
|---|---|---|---|
| none | 336.7 | 1,140.7 | 1,661.1 |
| 64K tokens | 331.9 | 1,071.3 | 1,633.5 |
| 128K tokens | 326.5 | 1,086.7 | 1,616.1 |

lil-bench's prompt repeats 20 sentences; prompts of varied text (exam questions, source code) read at 50.7-58.9K tokens
per second on `exact` in our own test.

**Many unrelated conversations at once.** lil-bench's conversations are alike, so they mostly need the same experts.
Conversations on unrelated text (code, a wiki article, a chat, math, Chinese) together need far more experts than the
GPU holds, and reading the rest from CPU memory becomes the limit. Our own test (each request continues a different
slice of varied text, 768 new tokens, temperature 1; total decode tokens per second over 90 s):

| profile | 8 conversations | 16 conversations |
|---|---|---|
| `exact` | 641 | 702-767 (three runs) |
| `lossy-fast` | 813 | 951 |
| `wavefront-fast` | 836 | 1,038-1,040 (two runs) |

Context: up to 1,048,576 tokens per request. The KV cache holds 2,602,934 tokens in all (about 2.4 KB per token), shared
by the requests running at once: two 1M-token conversations, or sixteen at 128K, with room to spare.

## What makes it fast

- **One kernel per expert layer.** In each of the model's 40 expert layers, a single MegaMoE launch sends every token
  to its 6 chosen experts and the shared one, runs each expert's two matrix products (8-bit activations against the
  experts' 4-bit weights) and adds the results back, keeping every core of the GPU busy instead of running a chain of
  separate kernels.
- **The experts that matter stay in GPU memory.** Layers 0-19 keep 320 of their 384 experts in GPU memory and layers
  20-39 keep 256: the experts every kind of text uses, plus 96 slots per layer that follow what the current
  conversations need. The first half gets more because it reads every token of a prompt; the second half reads only a
  prompt's last 128 tokens. Between steps, an expert that was just read from CPU memory is copied into the least
  recently used slot, and (since 1.7) switched over the moment its copy lands, so usually the very next step finds it
  in GPU memory. On lil-bench about 1-3% of the picks outside the fixed
  set still go to CPU memory; with 16 unrelated conversations, 12-15%.
- **The rest is read in place, exactly.** Every other expert stays, at full precision, in the CPU's memory. When a
  decode step picks one, MegaMoE reads it straight over the NVLink-C2C link between the CPU and the GPU (about 350 GB/s)
  inside the same launch. A small group of the GPU's cores handles these reads while the rest keep working from GPU
  memory, so one to three such reads per layer cost no time.
- **Prompts are staged ahead.** A chunk of 8K prompt tokens needs nearly every expert of layers 0-19, 30-60 of them
  from CPU memory on varied text. While the GPU works on one layer, the CPU-memory experts of the layer two ahead are
  copied into a 2.4 GB staging area in GPU memory (one batched copy at about 370 GB/s), and that layer reads them from
  there: the same bits, copied while the GPU is busy with other work. With staging switched off and on in the same
  server: MMLU-Pro questions 38.8K -> 50.7K tokens per second at 32K and 43.4K -> 58.9K at 128K, Python source
  40.7K -> 53.1K, lil-bench's repeating text 61.0K -> 62.6K.
- **dsv41's engine.** Speculative decoding: the model's own small drafter proposes several tokens, and the main model
  checks them in one step and keeps the ones it agrees with, without changing what the model would say. Each decode
  step (checking the drafts, choosing tokens, drafting the next ones) runs as one prebuilt CUDA graph instead of
  hundreds of separate GPU calls, for up to 16 requests, and a one-launch router picks every token's experts (bit for
  bit the stock router's choice).

What the experts Derpseek keeps in CPU memory actually do (mostly Chinese, formatting, code symbols and other long-tail
tokens, usually as a token's weakest picks): [docs/rare-experts.md](docs/rare-experts.md).

## What the profiles do

The profiles differ mainly in one decision: what a decode step does when it needs an expert that is not in GPU memory.

| profile | an expert that is not in GPU memory | other changes | KLD |
|---|---|---|---|
| `exact` (default) | read from CPU memory, exactly; the step waits for it when it has to | none | none added |
| `lossy-fast` | as `exact` | two of dsv41's lossy speed options | 0.028 |
| `wavefront-fast` | read while that costs no waiting; beyond that, the weakest are skipped for that step | as `lossy-fast` | 0.030 |
| `lossy-fast-prefill` | as `lossy-fast` | more of the prompt-side experts in GPU memory | 0.028 |

**`exact`** is the default and the main profile. It adds no error: every expert is computed at official precision, and
reading an expert from CPU memory, or from the prompt staging area, gives the same bits as reading it from GPU memory
(checked kernel by kernel against the all-GPU result, `tests/`). So there is no KLD to measure for it beyond the
engine's own (dsv41's lossless `default`). Its price is time: with many unrelated conversations, waiting for
CPU-memory reads becomes the limit.

**`lossy-fast`** adds two of dsv41's lossy speed options: a request's two weakest expert picks are skipped when no other
request in the step needs that expert, and the final word scores come from a compact 8-bit copy of the output layer.
With fewer experts to read and compute it is 9-17% faster than `exact` on lil-bench and 24-35% faster with many
unrelated conversations. Called `exact-fast` before 1.4.

**`wavefront-fast`** is `lossy-fast` with a time limit on CPU-memory reads while generating. In layers 20-38, each layer
call reads up to 1-5 CPU-memory experts (more with more requests), which the separate group of cores fetches while the
rest of the layer computes; any beyond that are skipped for that step, weakest first, never one of a token's two
strongest picks, and at most 4% of those layers' picks per step (past that, the step waits as in `exact`). Skipped
experts are still copied into GPU memory for the next steps, and prompt reading and layers 0-19 work as in
`lossy-fast`. On lil-bench, whose conversations are alike, it runs within 2% of `lossy-fast`; with 8 / 16 unrelated
conversations it is 3% / 9% faster. Its output depends a little on which experts earlier requests left in GPU memory,
so two identical requests can come out differently: dsv41's API check that compares a request on the fast path with
the same request on the stock path, token for token, passed 24 of 25 (every other release check passed). `skip-fast`,
its cruder form before 1.6, skipped every such pick (KLD 0.041).

**`lossy-fast-prefill`** keeps more of the prompt-side experts in GPU memory (layers 0-19: 352 of 384, layers 20-39:
224), for prompt-heavy work at `lossy-fast`'s KLD.

### About the KLD figures

Treat the lossy profiles' KLD figures as provisional: we think the way we measure them is not right for these
profiles, and we are revising it. They come from dsv41's measurement, which feeds one conversation at a time through
the decode path. But what the lossy profiles skip depends on everything else running in the same step and on what is
in GPU memory at that moment: `lossy-fast` skips a pick only when no other request in the step needs that expert (with
a single conversation, that is always true), and `wavefront-fast` skips more when many unrelated conversations miss at
once (with a single conversation, it rarely has to). So a busy server can be off from these figures in either
direction. The revised measurement will run many unrelated conversations through the decode path together. `exact` is
not affected: it adds no error, which is checked bit for bit rather than measured. One check under load so far: with
30 requests at once, both lossy profiles answered every `estonia` and `hotel-lights` sample correctly (above); those
30 requests share one prompt, so they need similar experts and do not test the unrelated-traffic case.

### Answer quality on `exact`

On the GB300 alone (GPQA and tool-eval-bench through the lil-bench client on 1.4, GSM8K in the 1.5 release checks,
lil-bench's `estonia` and `hotel-lights` task profiles on 1.6):

| test | result |
|---|---|
| GPQA Diamond (198 graduate-level science questions, temperature 0, reasoning effort high, up to 262K output tokens) | 90.4% (179/198; 95% interval 85.5-93.8%): physics 96.5%, chemistry 86.0%, biology 84.2%. 3 answers still ran out of tokens while reasoning and count as wrong. `lossy-fast` scored 92.4% (183/198) in 1.1: four questions apart is within run-to-run noise |
| [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (commit 570951a, community leaderboard settings, Hard Mode on) | 89/100 (164/184 points; 78 of 92 scenarios pass): Hard Mode 76%, safety and boundaries 85%, deployability 88 (median turn 0.9 s); 3 safety warnings (TC-74, TC-89, TC-92). `lossy-fast` scored 91/100 in 1.0: the engine is not bit-reproducible run to run, so a couple of points either way are noise |
| GSM8K (200 questions, dsv41's checker) | 196/200 at temperature 1, 194/200 greedy |
| lil-bench `estonia` (a 134,187-token packet of documents with planted decoys; name the country a vendor is in; 30 samples, 30 at once, temperature 1) | 30/30, none fell for a decoy; median answer 1,234 tokens |
| lil-bench `hotel-lights` (a short numeric reasoning puzzle; 30 samples, 30 at once, temperature 1) | 30/30; median answer 14,882 tokens |

`lossy-fast` and `wavefront-fast` also answered all 30 `estonia` and all 30 `hotel-lights` samples correctly. These two
tests were run once, on 1.6; they are not repeated for every release.

Release checks on `exact` 1.5 (dsv41's set, GB300 alone): question answering 6/6, tool calls 2/2, response ordering under
concurrency (96/96, 192/192, and 24/24 right after a 975K-token prompt), API behaviour 25/25, 16 simultaneous 30K-token
prompts, a 5-minute 6-agent stress run, GSM8K (above), and needle retrieval at 129K, 511K and 975K tokens (6/6, read at
60-84K tokens per second end to end).

## Running it

```
cd engine
MODEL=/path/to/DeepSeek-V4.1-Flash-1017 DSV41=/path/to/dsv41/engine ./derpseek.sh up [exact|lossy-fast|lossy-fast-prefill|wavefront-fast]
./derpseek.sh status | logs | down
```

With no profile named, `up` starts `exact`. The server listens on port 8112 with an OpenAI-compatible API (model names
`deepseek-v4.1-flash`, `derpseek`):

```
curl -s http://127.0.0.1:8112/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "derpseek", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Needs: a [dsv41](https://github.com/chronarion/dsv41) checkout with its runtime built (`./build.sh` once there), Docker
with the GB300's NVIDIA CDI device, the `vllm/vllm-openai:v0.32.0-aarch64` image, and about 350 GB of free CPU memory
(the Engram tables and the CPU copies of the experts are pinned).

Profiles live in `engine/profiles/`; `DERPSEEK_*` and `DSV41_*` variables set in the environment override them
(`DERPSEEK_STAGE=0` turns prompt staging off and gives its 2.4 GB back to the KV cache). Old profile names still work:
`exact-fast` and `exact-fast-kv` (before 1.4) start `lossy-fast`, `skip-fast` (before 1.6) starts `wavefront-fast`.

<details>
<summary>What is in this repository</summary>

| path | what it is |
|---|---|
| `engine/derpseek.sh` | the launcher |
| `engine/profiles/` | the profiles above |
| `engine/rowmaps/` | which experts each layer keeps in GPU memory (`exact-splitA-mm.json`, the default: 224 fixed + 96 cache slots in layers 0-19, 160 + 96 in layers 20-39) |
| `engine/derpseek_engine/exact.py` | the CPU-memory expert rows and their control block |
| `engine/derpseek_engine/residency.py` | the between-steps copier (least recently used slot out) |
| `engine/derpseek_engine/prefill.py` | prompt staging (and per-layer prompt timing for diagnosis) |
| `engine/derpseek_engine/cutk.py` | `wavefront-fast`'s cut: one small kernel between the router and the MoE kernel |
| `engine/overlay/megamoe_fork/` | the MoE kernel with CPU-memory rows and the separate group of cores for them |
| `tests/` | kernel tests against the all-GPU result (bit for bit, including staged prompts), the miss-cost benchmark, the copier test |
| `tools/make_rowmap.py` | builds a rowmap from routing traces |
| `docs/design.md` | the design notes and measurements |
</details>

## Credits

Derpseek is a new design, separate from [dsv41](https://github.com/chronarion/dsv41). It runs on top of the dsv41 engine
(its vLLM integration, whole-step CUDA graphs, kernels and speed options), and this repository carries modified copies of
three of its files:

- `engine/overlay/megamoe_fork/`: the MegaMoE kernel fork that dsv41 vendors from ds41 (Garner McCloud's
  [dgx-station-recipes](https://github.com/garnermccloud/dgx-station-recipes) DeepSeek-V4.1-Flash recipe, MIT), itself
  two headers of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT, `LICENSE.DeepGEMM`). Derpseek adds expert rows
  read from CPU memory and a separate scheduling lane for them.
- `engine/overlay/fastpath_boundary.py`: ds41's router glue (MIT), with a path that has no second GPU.
- `engine/hook/mega_peer_hook.py`: ds41's MoE hook (MIT), which comes from original-el8's
  [dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research) (Apache-2.0), with the
  CPU-memory expert rows and prompt staging.

`NOTICE` lists the licences; each carried file says at its top what was changed.

Keeping the recently used experts in GPU memory as a least-recently-used cache, with the others in CPU memory, follows
Eliseev and Mazur, [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238)
(2023). Copying the next layers' weights while the current layer computes (prompt staging) is the overlapped weight
loading of Sheng et al., [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single
GPU](https://arxiv.org/abs/2303.06865) (2023). Letting a CPU-memory expert of little weight cost precision instead of a
wait (`wavefront-fast`) follows Tang et al., [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE
Inference](https://arxiv.org/abs/2411.01433) (2024), which replaces such cache-miss experts with low-precision copies;
`wavefront-fast` skips them for that step. From dsv41: speculative decoding follows Leviathan, Kalman and Matias,
[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022), and Chen et al.,
[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023);
the lossy profiles' dropped expert picks, chosen per batch, follow Gupta et al., [Lynx: Enabling Efficient MoE Inference
through Dynamic Batch-Aware Expert Selection](https://arxiv.org/abs/2411.08982) (2024). dsv41 itself is built on top of
original-el8's work and on [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
