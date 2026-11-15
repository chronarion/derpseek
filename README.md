# Derpseek

DeepSeek-V4.1-Flash-1017 on one DGX Station GB300, with no second GPU.

[dsv41](https://github.com/chronarion/dsv41) serves this model fast by putting the experts that do not fit in the GB300's
memory on a second card, an RTX PRO 6000. Most DGX Stations do not have one. Derpseek runs the same model on the GB300
alone, at nearly the same decode speed, while every expert stays exactly as DeepSeek ships it.

## How fast, how close

Local Inference Lab's [standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench`
1.7.1, `standard/1`), profile `exact-fast`, GB300 only. Decode tokens per second:

| prompt context | 1 request | 8 requests | 16 requests |
|---|---|---|---|
| none | 374.6 | 1,333.9 | 1,888.0 |
| 64K tokens | 361.6 | 1,279.7 | 1,881.2 |
| 128K tokens | 349.6 | 1,280.3 | not measured (needs 2.13M KV tokens, this profile has 1.44M; see `exact-fast-kv`) |

Prompt reading: 56,198 tokens per second for a 32K-token prompt, 63,028 for 128K.

Context: up to 1,048,576 tokens per request. The KV cache holds 1,439,667 tokens in all (3.27 GiB of GPU memory, about
2.4 KB per token), shared by the requests running at once: one 1M-token conversation, or eight at 128K, or sixteen at
64K. Profile `exact-fast-kv` gives the KV cache 3,641,827 tokens (8.26 GiB) and runs every cell of the matrix; in that
run decode was 1-5% lower:

| `exact-fast-kv`, decode tok/s | 1 request | 8 requests | 16 requests |
|---|---|---|---|
| no prompt context | 364.5 | 1,313.8 | 1,834.6 |
| 64K tokens | 355.2 | 1,272.5 | 1,771.3 |
| 128K tokens | 344.0 | 1,268.8 | 1,759.3 |

| profile | 1 request | 8 requests | KLD vs the official model |
|---|---|---|---|
| `exact-fast` (default) | 374.6 | 1,333.9 | 0.028 |
| `exact` | 340 | 1,075 | no added loss (see below) |
| dsv41 `default`, GB300 + RTX PRO 6000, for comparison | 376 | 1,337 | 0.021 |

The `exact` row comes from 30-second lil-bench cells (same client and settings, no prompt context), not the full run.

KLD is measured on the decode path against the official-precision model (8 windows, 8,160 scored positions, the same
harness as dsv41); lower is closer, 0 is identical. `exact` adds nothing to the engine's own floor (dsv41's lossless
`default`, 0.021): reading an expert from CPU memory gives the same bits as reading it from GPU memory, checked kernel
by kernel against the all-GPU result.

Before this release the `exact-fast` server passed dsv41's release checks on the GB300 alone: question answering 6/6,
tool calls 2/2, response ordering under concurrency (96/96, 192/192, and 24/24 right after a 975K-token prompt), API
behaviour 25/25, 16 simultaneous 30K-token prompts, a 5-minute 6-agent stress run, GSM8K 192/200 (temperature 1) and
193/200 (greedy), and needle retrieval at 129K, 511K and 975K tokens.

## Answer quality

Measured on the `exact-fast` server, GB300 only (GPQA and tool-eval-bench through the lil-bench client):

| test | result |
|---|---|
| GPQA Diamond (198 graduate-level science questions, temperature 0, reasoning effort high, up to 131K output tokens) | 89.4% (177/198; 95% interval 84.3-93.0%): physics 93.0%, chemistry 88.2%, biology 78.9%. 5 answers ran out of tokens while still reasoning and count as wrong |
| [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (commit 570951a, community leaderboard settings, Hard Mode on) | 91/100 (168/184 points): Hard Mode 85%, safety and boundaries 88%, deployability 90 (median turn 0.8 s); 2 safety warnings (TC-89, TC-92) |
| GSM8K (200 questions, dsv41's checker) | 192/200 at temperature 1, 193/200 greedy (1.0: 195 / 194) |

## How it works

- The model needs about 285 GiB of weights on the GPU; the GB300 has 250.7 GiB. Each layer keeps 288 of its 384 experts
  in GPU memory: 192 that every kind of text uses, plus 96 slots that follow what the current conversation needs.
- Every other expert stays, exact, in the CPU's memory. When the model picks one, the GPU reads it straight from CPU
  memory over the NVLink-C2C link inside the same computation, so nothing waits for a copy. A small group of the GPU's
  cores handles these reads while the rest keep working from GPU memory, so one to three such reads per layer cost no
  time at all.
- Between steps, an expert that was just read from CPU memory is copied into a free GPU slot (the one used least
  recently), so the next steps find it there. About 1-2% of the picks that are not in the fixed 192 still go to CPU
  memory.
- `exact-fast` adds dsv41's speed options: a request's two weakest expert picks are skipped when no other request in
  the step needs that expert, and the final word scores use a compact 8-bit copy of the output layer.

<details>
<summary>Profiles</summary>

| profile | what changes | KLD |
|---|---|---|
| `exact` | every expert exact; a pick that is not in GPU memory is read from CPU memory (costs time) | lossless |
| `exact-fast` | `exact` + dsv41's singleton drop-2, MXFP8 verify head, router to 64 rows, step graphs to batch 16 | 0.028 |
| `exact-fast-kv` | `exact-fast` with a 3.6M-token KV cache (the GPU memory left unused at 0.96 utilization) | 0.028 |
| `skip-fast` (experimental) | `exact-fast`, but in layers 20-38 a pick that is not in GPU memory is skipped for that step instead of read (costs precision, never time); it is still copied in for the next steps. With today's miss rate it is slower than `exact-fast` (C1 339, C8 1,297: the skip runs as extra GPU launches, and the changed answers lower draft acceptance) | 0.041 |

Profiles live in `engine/profiles/`; `DERPSEEK_*` and `DSV41_*` variables set in the environment override them.
</details>

<details>
<summary>Running it</summary>

```
cd engine
MODEL=/path/to/DeepSeek-V4.1-Flash-1017 DSV41=/path/to/dsv41/engine ./derpseek.sh up [exact-fast|exact|skip-fast]
./derpseek.sh status | logs | down
```

Needs: a [dsv41](https://github.com/chronarion/dsv41) checkout with its runtime built (`./build.sh` once there), Docker
with the GB300's NVIDIA CDI device, the `vllm/vllm-openai:v0.32.0-aarch64` image, and about 350 GB of free CPU memory
(the Engram tables and the CPU copies of the experts are pinned). The server listens on port 8112 with an
OpenAI-compatible API (model names `deepseek-v4.1-flash`, `derpseek`).

Long prompts are read more slowly than with dsv41 (lil-bench: 57-63K tokens per second against dsv41's 75K): a prompt
chunk touches nearly every expert, and the ones in CPU memory come over the link.
</details>

<details>
<summary>What is in this repository</summary>

| path | what it is |
|---|---|
| `engine/derpseek.sh` | the launcher |
| `engine/profiles/` | the profiles above |
| `engine/rowmaps/` | which experts each layer keeps in GPU memory (`exact-s192-c96-mm.json`: 192 fixed + 96 cache slots) |
| `engine/derpseek_engine/exact.py` | the CPU-memory expert rows and their control block |
| `engine/derpseek_engine/residency.py` | the between-steps copier (least recently used slot out) |
| `engine/overlay/megamoe_fork/` | the MoE kernel with CPU-memory rows and the separate group of cores for them |
| `tests/` | kernel tests against the all-GPU result (bit for bit), the miss-cost benchmark, the copier test |
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
  CPU-memory expert rows.

`NOTICE` lists the licences; each carried file says at its top what was changed.

Keeping the recently used experts in GPU memory as a least-recently-used cache, with the others in CPU memory, follows
Eliseev and Mazur, [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238)
(2023). dsv41 itself is built on top of original-el8's work and on [vLLM](https://github.com/vllm-project/vllm)
(Apache-2.0).
