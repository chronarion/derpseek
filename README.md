# Derpseek

DeepSeek-V4.1-Flash-1017 on one DGX Station GB300, with no second GPU.

[dsv41](https://github.com/chronarion/dsv41) serves this model fast by putting the experts that do not fit in the GB300's
memory on a second card, an RTX PRO 6000. Most DGX Stations do not have one. Derpseek runs the same model on the GB300
alone, with every expert computed exactly as DeepSeek ships it.

## How fast

The default profile, `exact`: every expert at official precision, none skipped or approximated. Local Inference Lab's
[standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench` 1.7.1, `standard/1`),
GB300 only. Decode tokens per second:

| prompt context | 1 request | 8 requests | 16 requests |
|---|---|---|---|
| none | 336.7 | 1,140.7 | 1,661.1 |
| 64K tokens | 331.9 | 1,071.3 | 1,633.5 |
| 128K tokens | 326.5 | 1,086.7 | 1,616.1 |

Prompt reading (lil-bench): 66,434 tokens per second for a 32K-token prompt, 70,634 for 128K. lil-bench's prompt repeats
20 sentences; prompts of varied text (exam questions, source code) read at 50.7-58.9K tokens per second in our own test,
1.3-1.4x what they read without prompt staging (see How it works).

Context: up to 1,048,576 tokens per request. The KV cache holds 2,602,934 tokens in all (about 2.4 KB per token), shared by
the requests running at once: two 1M-token conversations, or sixteen at 128K, with room to spare.

## How close

`exact` adds no error: every expert is computed at official precision, and reading an expert from CPU memory, or from
the prompt staging area, gives the same bits as reading it from GPU memory (checked kernel by kernel against the all-GPU
result, `tests/`). So there is no KLD to measure for it beyond the engine's own (dsv41's lossless `default`).

| profile | 1 request | 8 requests | 16 requests | prompt reading (lil) | KLD vs the official model |
|---|---|---|---|---|---|
| `exact` (default) | 336.7 | 1,140.7 | 1,661.1 | 66,434 / 70,634 | none added |
| `lossy-fast` (bonus: lossy speed options) | 365.7 | 1,333.5 | 1,906.9 | 66,240 / 70,307 | 0.028 |
| dsv41 `default` (lossless), GB300 + RTX PRO 6000, for comparison | 376 | 1,337 | – | 75K | none added |

Decode columns: no prompt context.

Release checks on `exact` 1.5 (dsv41's set, GB300 alone): question answering 6/6, tool calls 2/2, response ordering under
concurrency (96/96, 192/192, and 24/24 right after a 975K-token prompt), API behaviour 25/25, 16 simultaneous 30K-token
prompts, a 5-minute 6-agent stress run, GSM8K (below), and needle retrieval at 129K, 511K and 975K tokens (6/6, read at
60-84K tokens per second end to end; 1.3: 44-64K).

## Answer quality

`exact` on the GB300 alone (GPQA and tool-eval-bench through the lil-bench client on 1.4, GSM8K in the 1.5 release checks):

| test | result |
|---|---|
| GPQA Diamond (198 graduate-level science questions, temperature 0, reasoning effort high, up to 262K output tokens) | 90.4% (179/198; 95% interval 85.5-93.8%): physics 96.5%, chemistry 86.0%, biology 84.2%. 3 answers still ran out of tokens while reasoning and count as wrong. `lossy-fast` scored 92.4% (183/198) in 1.1: four questions apart is within run-to-run noise |
| [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (commit 570951a, community leaderboard settings, Hard Mode on) | 89/100 (164/184 points; 78 of 92 scenarios pass): Hard Mode 76%, safety and boundaries 85%, deployability 88 (median turn 0.9 s); 3 safety warnings (TC-74, TC-89, TC-92). `lossy-fast` scored 91/100 in 1.0: the engine is not bit-reproducible run to run, so a couple of points either way are noise |
| GSM8K (200 questions, dsv41's checker) | 196/200 at temperature 1, 194/200 greedy |

## How it works

- The model needs about 285 GiB of weights on the GPU; the GB300 has 250.7 GiB. Layers 0-19 keep 320 of their 384
  experts in GPU memory and layers 20-39 keep 256: the experts every kind of text uses, plus 96 slots per layer that
  follow what the current conversation needs. The first half gets more because it reads every token of a prompt; the
  second half only reads a prompt's last 128 tokens.
- Every other expert stays, exact, in the CPU's memory. When a decode step picks one, the GPU reads it straight from CPU
  memory over the NVLink-C2C link inside the same computation. A small group of the GPU's cores handles these reads
  while the rest keep working from GPU memory, so one to three such reads per layer cost no time at all.
- Between steps, an expert that was just read from CPU memory is copied into a free GPU slot (the one used least
  recently), so the next steps find it there. About 1-3% of the picks outside the fixed set still go to CPU memory.
- Prompts are different: a chunk of 8K prompt tokens needs nearly every expert of layers 0-19 at once, 30-60 of them
  from CPU memory on varied text. So while the GPU works on one layer, the CPU-memory experts of the layer two ahead are
  copied into a 2.4 GB staging area in GPU memory (one batched copy at about 370 GB/s), and that layer reads them from
  there. The bits are the same; the copy just happens while the GPU is busy with other work. In the same server with
  staging switched off and on: MMLU-Pro questions 38.8K -> 50.7K tokens per second at 32K and 43.4K -> 58.9K at 128K,
  Python source 40.7K -> 53.1K, lil-bench's repeating text 61.0K -> 62.6K.

<details>
<summary>Profiles</summary>

| profile | what changes | KLD |
|---|---|---|
| `exact` (default) | every expert exact; a pick that is not in GPU memory is read from CPU memory (costs time), prompts stage those experts in GPU memory ahead; dsv41's lossless speed options, including (since 1.5) its one-launch router up to 64 rows and step graphs up to 16 requests | no added loss |
| `lossy-fast` | `exact` + dsv41's lossy speed options: a request's two weakest expert picks are skipped when no other request in the step needs that expert, and the final word scores use a compact 8-bit copy of the output layer. Called `exact-fast` before 1.4 | 0.028 |
| `lossy-fast-prefill` | `lossy-fast` with more of the prompt-side experts in GPU memory (layers 0-19: 352 of 384, layers 20-39: 224) | 0.028 |
| `skip-fast` (experimental) | `lossy-fast`, but in layers 20-38 a pick that is not in GPU memory is skipped for that step instead of read (costs precision, never time) | 0.041 |

Profiles live in `engine/profiles/`; `DERPSEEK_*` and `DSV41_*` variables set in the environment override them
(`DERPSEEK_STAGE=0` turns prompt staging off and gives its 2.4 GB back to the KV cache).
</details>

<details>
<summary>Running it</summary>

```
cd engine
MODEL=/path/to/DeepSeek-V4.1-Flash-1017 DSV41=/path/to/dsv41/engine ./derpseek.sh up [exact|lossy-fast|lossy-fast-prefill|skip-fast]
./derpseek.sh status | logs | down
```

Needs: a [dsv41](https://github.com/chronarion/dsv41) checkout with its runtime built (`./build.sh` once there), Docker
with the GB300's NVIDIA CDI device, the `vllm/vllm-openai:v0.32.0-aarch64` image, and about 350 GB of free CPU memory
(the Engram tables and the CPU copies of the experts are pinned). The server listens on port 8112 with an
OpenAI-compatible API (model names `deepseek-v4.1-flash`, `derpseek`). `exact-fast` and `exact-fast-kv` (before 1.4)
still start `lossy-fast`.
</details>

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
| `engine/overlay/megamoe_fork/` | the MoE kernel with CPU-memory rows and the separate group of cores for them |
| `tests/` | kernel tests against the all-GPU result (bit for bit, including staged prompts), the miss-cost benchmark, the copier test |
| `tools/make_rowmap.py` | builds a rowmap from routing traces |
| `docs/design.md` | the design notes and measurements |
</details>

## Credits

Derpseek is a new design, separate from [dsv41](https://github.com/chronarion/dsv41). It runs on top of the dsv41 engine
(its vLLM integration, whole-step CUDA graphs, kernels and speed options), and this repository carries modified copies of
three of its files:

- `engine/overlay/megamoe_fork/`: the MegaMoE kernel fork that dsv41 vendors from ds41
  ([original-el8's DeepSeek-V4.1-Flash work](https://github.com/original-el8/dgx-station-gb300-research/tree/main/deepseek-v4.1-flash),
  MIT), itself two headers of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT, `LICENSE.DeepGEMM`). Derpseek
  adds expert rows read from CPU memory and a separate scheduling lane for them.
- `engine/overlay/fastpath_boundary.py`: ds41's router glue (MIT), with a path that has no second GPU.
- `engine/hook/mega_peer_hook.py`: ds41's MoE hook (MIT), which comes from original-el8's
  [dgx-station-gb300-research](https://github.com/original-el8/dgx-station-gb300-research) (Apache-2.0), with the
  CPU-memory expert rows and prompt staging.

`NOTICE` lists the licences; each carried file says at its top what was changed.

Keeping the recently used experts in GPU memory as a least-recently-used cache, with the others in CPU memory, follows
Eliseev and Mazur, [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238)
(2023). Copying the next layers' weights while the current layer computes (prompt staging) is the overlapped weight
loading of Sheng et al., [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single
GPU](https://arxiv.org/abs/2303.06865) (2023). dsv41 itself is built on top of original-el8's work and on
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
