"""Training prompts for drafter-correction fits (C1 push): disjoint from prompts.py (the evaluation set) and from lil.
Topics x task templates over many kinds of text, a fixed seed; sent one at a time (batch 1), server-default sampling.
usage: python3 -I train_prompts.py OUT.jsonl [N] [PORT]"""
import json
import random
import sys
import time
import urllib.request

TOPICS = ["volcanoes", "the history of the printing press", "coral reefs", "public transit in large cities",
          "the immune system", "black holes", "medieval castles", "beekeeping", "electric grids", "the Silk Road",
          "photosynthesis", "jazz", "the French Revolution", "machine translation", "ocean tides", "urban gardening",
          "earthquakes", "the Roman aqueducts", "chess openings", "climate adaptation", "sleep", "glaciers",
          "the stock market", "antibiotic resistance", "the moon landing", "wind turbines", "ancient Egypt",
          "coffee", "rainforests", "remote work", "the human eye", "bridges", "migratory birds", "board games",
          "solar panels", "the Renaissance", "fermentation", "cryptography", "deserts", "marathon training"]
TEMPLATES = {
    "explain": ["Explain {t} to a curious adult with no background in it.",
                "What are the most common misconceptions about {t}?",
                "Give me a detailed overview of {t}, with sections and headings."],
    "essay": ["Write a 400-word essay about {t}.", "Write a blog post about {t} for a general audience.",
              "Write a short speech for a school assembly about {t}."],
    "story": ["Write a short story in which {t} plays a central role.",
              "Write a dialogue between two friends arguing about {t}.",
              "Write a diary entry from someone whose life changed because of {t}."],
    "list": ["List 10 surprising facts about {t}, each with one sentence of explanation.",
             "Make a study guide about {t} with key terms and definitions.",
             "Write 8 quiz questions about {t} with answers."],
    "table": ["Make a markdown table summarizing key facts about {t}.",
              "Return a JSON object describing {t} with fields summary, key_dates, key_people, open_questions."],
    "lang": ["Explique {t} en français, en quelques paragraphes.", "Erkläre {t} auf Deutsch für Schüler.",
             "Explica {t} en español de forma sencilla.", "{t}에 대해 한국어로 설명해 주세요.",
             "请用中文写一段关于{t}的介绍。", "{t}について日本語で簡単に説明してください。"],
    "math": ["Write a math word problem involving {t}, then solve it step by step.",
             "Estimate, with a Fermi calculation, a quantity related to {t}. Show your reasoning."],
}
CODE = ["Write a Go program that reads a CSV file and prints the average of each numeric column.",
        "Implement a binary search tree in Java with insert, delete and in-order traversal.",
        "Write a Python class for a bank account with deposits, withdrawals and a transaction log, plus unit tests.",
        "Write a JavaScript function that debounces another function, and explain how it works.",
        "Write a C++ program that computes the shortest path in a weighted graph with Dijkstra's algorithm.",
        "Write a shell script that backs up a directory into dated tar.gz archives and keeps the last 7.",
        "Write a Python script that parses Apache access logs and reports the top 10 IPs and URLs.",
        "Implement a trie in Rust with insert and prefix search.",
        "Write a SQL schema for a library system (books, members, loans) and three useful queries.",
        "Write a Python function that validates a Sudoku board, with tests.",
        "Write a Kotlin data class for a todo item and functions to filter and sort a list of them.",
        "Write a Haskell function that computes the run-length encoding of a list.",
        "Write a small Flask app with endpoints to create, list and delete notes stored in SQLite.",
        "Write a Python generator that yields prime numbers, and benchmark two implementations.",
        "Explain what this code does and how to improve it:\n\ndef f(a):\n    r = {}\n    for x in a:\n        if x in r:\n"
        "            r[x] = r[x] + 1\n        else:\n            r[x] = 1\n    return sorted(r.items(), key=lambda t: -t[1])",
        "Write a TypeScript type-safe event emitter.",
        "Write a Python script using regex to extract all email-like strings and dates from a text file.",
        "Write a C program that implements a ring buffer with push and pop.",
        "Write a React component for a searchable dropdown list.",
        "Write a Python implementation of k-means clustering with numpy, and explain each step."]


def build(n, seed=20261122):
    rng = random.Random(seed)
    out = []
    kinds = list(TEMPLATES) + ["code"]
    while len(out) < n:
        k = kinds[len(out) % len(kinds)]
        if k == "code":
            p = CODE[rng.randrange(len(CODE))]
        else:
            p = rng.choice(TEMPLATES[k]).format(t=rng.choice(TOPICS))
        if all(p != q for _, q in out):
            out.append((k, p))
    return out


def main():
    out = open(sys.argv[1], "a")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    port = sys.argv[3] if len(sys.argv) > 3 else "8112"
    for i, (kind, p) in enumerate(build(n)):
        body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": p}], "max_tokens": 2048,
                "stream": False}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1200))
        u = r.get("usage", {})
        rec = {"kind": kind, "i": i, "half": "train", "t0": t0, "s": round(time.time() - t0, 2),
               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "finish": r["choices"][0].get("finish_reason")}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{kind} {i}: {u.get('prompt_tokens')} + {u.get('completion_tokens')} tokens in {rec['s']} s", flush=True)


if __name__ == "__main__":
    main()
