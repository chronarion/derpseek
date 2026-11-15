# What the rarely used experts do

Every layer of DeepSeek-V4.1-Flash has 384 experts, and each token is sent to 6 of them. Derpseek keeps the experts
that most text uses in GPU memory and leaves the rest in CPU memory. That raises an obvious question: what are the
rarely used ones for?

## How we looked

We used the official model's own routing on the 18 text windows behind our KLD measurement (49,152 tokens: Wikipedia,
source code, chat transcripts, math word problems in chat form, Chinese prose and long documents). For each layer we
took the experts Derpseek leaves in CPU memory and outside its starting cache (64 per layer in layers 0-19, 128 in
layers 20-39: the "cold" experts), and asked which tokens, in which texts, are sent to them.

## What we found

**They are rarely picked, and weakly.** The cold experts take 2.7% of all picks in layers 0-19 and 8.7% in layers
20-39. When a token does pick one, it is usually one of its weakest picks: in layers 0-19, 52% of cold picks are a
token's 5th or 6th choice and 6% its first. That is why skipping a few of them now and then costs so little accuracy.
Only 19 of the 15,360 experts were never picked at all.

**They serve the long tail of text.** Ordinary English words mostly go to experts that stay in GPU memory. Cold picks
lean towards everything else. Cold picks per token, by kind of token (layers 20-39):

| kind of token | share of the text | cold picks per token |
|---|---|---|
| whole English words | 51% | 0.40 |
| English word pieces | 10% | 0.53 |
| punctuation and symbols | 18% | 0.53 |
| digits | 3% | 0.52 |
| spaces and indentation | 5% | 0.59 |
| Chinese characters | 7% | 1.19 |
| Chinese punctuation | 1% | 1.14 |

**Some look like specialists, but these labels are untested hypotheses.** Of the 3,612 cold experts picked at least
20 times, 1,366 take 60% or more of their picks from one kind of text: 422 from Chinese, 347 from code, 245 from chat,
191 from long documents, 94 from math chats and 67 from Wikipedia. We have not yet compared that with random "experts"
(groups of picks that mean nothing), checked the labels on text they were not drawn from, or switched the experts off
to see what breaks, and a few documents dominate some of them. Some examples (layer, expert):

| expert | fires on | example (the token in brackets) |
|---|---|---|
| L21 e94, L22 e37, L23 e31 and about ten more in layers 20-38 | Chinese clause punctuation: ，。、 | 受湖岸线变化 [，] 湮没和暴露交替 |
| L28 e114, L29 e49, L34 e165, L35 e226 | the hand-over from question to answer in math chats: the final ?, the assistant marker, "The answer is" | how many years ago was Raymond's son born [?] <Assistant>The answer is |
| L23 e222, L33 e181 | punctuation and speaker turns in chat transcripts | I'm surprised they went with the fake laughing [.] So fake. |
| L38 e252, L38 e227, L39 e106 | C++ / CUDA comments, pointer syntax and indentation | half [*] A_had_ptr = |
| L14 e269, L38 e240 | ASCII tables drawn in code comments | # +=======+=======+ [======] =+ |
| L38 e143 | 年 ("year") right after a number in Chinese | 2017 [年] 发现的一件瓷质 |
| L29 e332 | names from one Korean War article (Naktong, NK, Korea) | a topic specialist |

## A stricter second look

We repeated the question on a far more varied text set (`tools/varied_corpus.py`: Wikipedia and web text in 30
languages, Rosetta Code in 30 programming languages, LaTeX math, chats in many languages, biomedical articles,
generated JSON, CSV, YAML, HTML, logs and tables; 436,224 tokens) plus the original windows, with the methods a claim
needs (`tools/rare_experts_stats.py`, following the interpretability literature):

- labels are found on one window per kind of text and checked on the other windows (text they were not drawn from);
- "this expert prefers this kind of text" is a log-odds ratio with an informative prior, tested by shuffling the
  kinds of text across whole windows, with a false-discovery limit of 1% over all ~138,000 expert x kind tests, and
  only counted when the preference is at least three to one and spread over at least three windows;
- the identical test on fake experts (random positions, the same number of picks) finds **0** such preferences, so
  the ones below are not artifacts of the method.

What came out:

- **Specialists are everywhere, not only among the cold experts.** 8,126 expert x kind preferences pass, on 7,315
  experts (about half of all of them), and 91% of them hold up on the held-out windows. 45% of the cold experts have
  one, and 49% of the others.
- **What sets the cold experts apart is which text they specialize in.** Their replicated specialties: Chinese,
  Japanese and Korean 361, structured data 332, mainstream code 306, math and science 198, chat 156, rare
  programming languages 147, other scripts 107, Latin-script languages 54, and the ordinary text the cache was
  calibrated on 7. "Cold" mostly means "a specialist for text the calibration did not contain".
- **Routing depends on context more than on the token.** A token's six most frequent experts on the discovery
  windows overlap its experts on the held-out windows by 0.42 (Jaccard) in layer 0, 0.14-0.23 in the middle layers
  and 0.28 in layer 39, against 0.09 by chance: the token matters, but most of the choice comes from the context.

## What this means for Derpseek

"Cold" is relative to the text the cache was calibrated on, and that text is mostly English: Chinese makes up 8% of it,
and Chinese tokens use cold experts three times as often as English words. A workload heavy in Chinese, or in code,
would keep a different set in GPU memory. The expert cache adapts while serving (an expert read from CPU memory moves
into GPU memory for the next steps), so the fixed set mainly decides how fast a new kind of text starts.

Still to do: a test that switching a group of specialists off hurts the text it is supposed to serve and little
else (against a random group of the same size), and a rowmap calibrated on varied text, measured on text it did not
see. Both are running in Derpseek's expert-pinning bench (`bench/deb`).

Limits: 49,152 tokens in six kinds of text is a small sample. The Chinese text is one stretch of Chinese Wikipedia (a
few articles, one about old kilns) and the Wikipedia text includes a long Korean War article, so some "specialists"
are partly specialists in those articles.
