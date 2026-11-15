# Expert-pinning bench

Derpseek keeps some experts in GPU memory and leaves the rest in CPU memory. Which experts stay, how fast a newly needed
one moves in, and what a decode step does when an expert is not in GPU memory are all choices: profiles, rowmaps,
cache and promotion settings, skip and defer policies. This bench compares such choices ("approaches") on the same
inputs, across different kinds of text, with an emphasis on text that needs the rarely used experts, and reports
three things for each approach and each kind of text:

- **KLD against `exact`**: how far the approach's next-token probabilities move from `exact`'s on exactly the same
  tokens, through the served decode path, 16 conversations at a time (the shape of 16 requests with 3 drafts each),
  with the expert cache running. A second `exact` run gives the floor (the engine is not bit-reproducible run to run).
- **Prompt reading speed**: long prompts made of that kind of text, tokens per second end to end.
- **Decode speed**: 8 conversations at once, each continuing a different piece of that kind of text, tokens per second,
  and the share of expert picks that had to come from CPU memory meanwhile.

## The text

One suite of 2,048-token windows (`make_suites.py`), labelled by group:

| group | what |
|---|---|
| ordinary | the KLD corpus: Wikipedia, code, chat, math chats, Chinese |
| degenerate | random tokens, shuffled text, text that switches topic every 64 tokens (`make_corpus.py`) |
| latin-script, other-scripts, cjk | Wikipedia and web text in 30 languages (`tools/varied_corpus.py`) |
| code-main, code-rare | Rosetta Code in 30 programming languages, from Python and C++ to COBOL, Forth and J |
| structured | JSON, CSV, YAML, HTML, server logs, Markdown tables |
| chat, math-science | conversations in many languages; LaTeX math and biomedical articles |
| cold-heavy | the windows whose official routing puts the most picks on rarely used experts (`cold_rates.py`) |

With `--holdout`, each varied category's first window stays out of the suite, so approaches calibrated on text (a
rowmap, for example) are judged on text they did not see.

## Running it

```
python3 -I make_suites.py suite.json --kld KLD_CORPUS.json --deb deb.json --varied varied.json --holdout --cold rates.json
# per approach (a profile plus any DERPSEEK_* / DSV41_* / ROWMAP settings in the environment):
./derpseek.sh up PROFILE; python3 -I serve_bench.py suite.json serve-NAME.json --docker derpseek; ./derpseek.sh down
CORPUS=suite.json OUT=kld-NAME PROFILE=PROFILE MODEL=... DSV41=... GPU=nvidia.com/gpu=... \
    ./run_kld.sh --batch 16 --rows 240 --routes full --ws-decode
# one report: KLD, prompt reading, decode and misses per group, one row per approach
python3 -I report.py suite.json --base kld-exact/full exact-again:kld=kld-exact2/full NAME:kld=kld-NAME/full:serve=serve-NAME.json
```

`run_kld.sh` drives dsv41's `engine/harness/decode_kld.py` (teacher-forced verify rows through the served decode path)
and needs the GPU to itself; `serve_bench.py` needs a running server and only the Python standard library.
