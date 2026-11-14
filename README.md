# Derpseek

DeepSeek-V4.1-Flash-1017 on one DGX Station GB300, with no second GPU.

[dsv41](https://github.com/chronarion/dsv41) serves this model fast by putting the experts that do not fit in the GB300's
memory on a second card, an RTX PRO 6000. Most DGX Stations do not have one. Derpseek asks how fast the model can run
on the GB300 alone, while keeping its output close to DeepSeek's own: at least one profile must stay below a KLD of
0.08 against the official-precision model.

**Status: design stage.** Nothing here serves yet. The plan, the numbers behind it and the open questions are in
[docs/design.md](docs/design.md).

## The idea in short

- The model needs about 285 GiB of weights on the GPU; the GB300 has 250.7 GiB. The missing room is found in the top
  half of the model, which in practice only runs while generating, not while reading the prompt.
- The bottom half stays exactly as DeepSeek ships it, so reading prompts keeps full speed.
- The top half's rarely used experts either stay exact in the CPU's memory and are streamed in over the fast
  CPU-GPU link when needed, or are stored in a more compact form that unpacks into the GPU's native 4-bit format.
- The CPU-GPU link sits idle most of the time, so it can keep streaming the extra detail of experts just before the
  model needs them.

<details>
<summary>Measurement tools in this repository</summary>

| path | what it measures (CPU only) |
|---|---|
| `tools/analysis/fp4stats.py` | entropy of the experts' FP4 codes and scales |
| `tools/analysis/prefetch_stats.py` | which signals predict a rarely used expert's next use |
| `tools/analysis/lru_sim.py` | a cache of rarely used experts: hit rate against size |
| `tools/analysis/memo_sim.py` | how much of a reply's expert use its prompt already shows |
| `tools/analysis/conc_sim.py` | the streamed-refinement cache under 1-16 concurrent requests |

Each takes the checkpoint directory or a directory of routing traces as its argument.
</details>

## Credits

Derpseek is a new design, separate from [dsv41](https://github.com/chronarion/dsv41). It runs on top of the dsv41 engine
(its vLLM integration, whole-step CUDA graphs and kernels), and this repository carries modified copies of three of
its files:

- `engine/overlay/megamoe_fork/`: the MegaMoE kernel fork that dsv41 vendors from ds41
  ([original-el8's DeepSeek-V4.1-Flash work](https://github.com/original-el8/dgx-station-gb300-research/tree/main/deepseek-v4.1-flash),
  MIT), itself two headers of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT, `LICENSE.DeepGEMM`). Derpseek
  adds expert rows read from host memory.
- `engine/hook/mega_peer_hook.py` and `engine/overlay/fastpath_boundary.py`: ds41's MoE hook and router glue (MIT),
  with a mode that has no second GPU.

dsv41 itself is built on top of original-el8's work and on [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
