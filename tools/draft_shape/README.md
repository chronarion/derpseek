# Draft reshape and acceptance tools (1.10)

Measurement and fitting scripts behind `engine/draft_shape/shape-v1.json` and the 1.10 notes in `docs/design.md`.

1. Serve the measurement profile `acc3-exact` (`derpseek.sh up acc3-exact`): every batch-1 step verifies all 7 drafts
   and `derpseek_engine/acc3.py` writes, per draft position, the drafter's and the target's top-64 tokens with their
   probabilities and the drafter's hidden state (`DERPSEEK_ACC3=<path>.jsonl`, plus `.bin` and `.h16`).
2. Send the training prompts (`train_prompts.py OUT 160`), then the evaluation prompts (`eval_prompts.py OUT`), then
   lil-bench C1 cells, writing a phase marker file (`train <unix time>`, `eval ...`, `lil ...`): steps are labelled by
   time. The training prompts share no prompt with the evaluation set.
3. Fit and score (CPU, in the vLLM image; DIR holds the dump, the phase file, the evaluation log and `periods.json`):
   - `fit_shape.py DIR`: the reshape shipped as `shape-v1.json` (fitted on the training prompts only; scored per kind
     of text on the held-out prompts and lil);
   - `fit_acc3.py DIR`: context-free variants and an online per-request token bias;
   - `train_lora.py DIR MODEL_DIR RANK EPOCHS` (GPU): a low-rank correction of the drafter's hidden state;
   - `conftilt_eval.py DIR MODEL_DIR`: a confidence-head tilted proposal;
   - `shortlist.py DIR`, `sibsim.py DIR`: a drafter vocabulary shortlist, sibling candidates at position 1.
   - with the `acc3t-exact` dump (`DERPSEEK_ACC3_TARGET=1`: each verify row's layer-20 state and the target's final
     hidden state): `prune_sim.py DIR MODEL_DIR` (mid-network row pruning, probe and oracle) and
     `train_lora2.py DIR MODEL_DIR RANK EPOCHS` (a drafter correction that also sees the target's last state) and
     `train_rerank.py DIR MODEL_DIR 16 EPOCHS` (the SwiGLU reranker; LORA_LR, FRAC = share of the training requests).
