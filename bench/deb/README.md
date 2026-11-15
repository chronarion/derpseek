# Degenerate expert bench (DEB)

The KLD figures in the main README come from a measurement that feeds one conversation at a time. The lossy profiles
make decisions that depend on everything else in the step: `lossy-fast` drops a weak pick only when no other request
needs that expert, and `wavefront-fast` skips more when many requests need experts that are not in GPU memory. This
bench puts them in the hard cases and measures how far each profile's next-token probabilities move from `exact`'s on
exactly the same inputs.

The corpus has 16 windows of 2,048 tokens (`make_corpus.py`): 4 ordinary ones (wiki, code, chat, Chinese), 4 of random
tokens (the widest spread of experts), 4 shuffled (an ordinary token mix with no structure) and 4 that switch topic
every 64 tokens (the expert set keeps changing). They are decoded through the served decode path, teacher-forced, with
dsv41's `decode_kld.py`, 16 at a time (64 rows a step, the shape of 16 requests with 3 drafts each) and one at a time
(as the main README's KLD), with the expert cache running.

`report.py` then prints, per window group, the KLD of each profile against an `exact` run at the same batch, its 99th
percentile and the top-1 agreement. A second `exact` run gives the floor: the engine is not bit-reproducible from run
to run, so `exact` against `exact` is not zero.

```
python3 -I make_corpus.py KLD_CORPUS.json deb.json
DSV41_KLD_CORPUS=deb.json python3 decode_kld.py OUT --batch 16 --rows 240 --routes full --ws-decode   # per profile
python3 -I report.py EXACT_OUT/full exact-again=EXACT2_OUT/full lossy-fast=LOSSY_OUT/full ...
```
