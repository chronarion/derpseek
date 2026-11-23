"""Real-text decode prompts for the C1 acceptance study: six kinds x 8, sent one at a time (batch 1), server-default
sampling, natural end of answer. Even indices = tuning half, odd = held-out half.
usage: python3 -I prompts.py OUT.jsonl [PORT]"""
import json
import sys
import time
import urllib.request

P = {
    "chat": [
        "What are the main differences between a heat pump and a furnace for heating a house? Which is cheaper to run in a mild climate?",
        "My sourdough starter smells like nail polish remover. What does that mean and how do I fix it?",
        "Explain how vaccines train the immune system, for a curious 12-year-old.",
        "I have a job offer with a higher salary but a longer commute. What questions should I ask myself before deciding?",
        "Why do airplanes leave white trails in the sky, and why do some disappear quickly while others spread out?",
        "Give me a 3-day itinerary for a first visit to Lisbon on a moderate budget.",
        "How does compound interest work? Show an example with a monthly deposit.",
        "What should I check before buying a used electric car?",
    ],
    "reasoning": [
        "A train leaves city A at 9:00 at 80 km/h. Another leaves city B, 300 km away, at 10:00 toward A at 100 km/h. When and where do they meet?",
        "A shop sells pens in packs of 6 for $4 and packs of 10 for $6. What is the cheapest way to buy exactly 58 pens, and what does it cost?",
        "If the probability of rain on each of 3 days is 0.3, independently, what is the probability it rains on at least one day? Then on exactly two days?",
        "Prove that the sum of the first n odd numbers is n squared.",
        "A rectangle has a perimeter of 46 cm and a diagonal of 17 cm. Find its sides.",
        "Three friends split a bill. Ana pays twice what Ben pays, and Cleo pays $12 more than Ben. The bill is $112. How much does each pay?",
        "How many integers between 1 and 1000 are divisible by 3 or 7 but not by both?",
        "A tank fills in 6 hours with pipe A and 9 hours with pipe B; a drain empties it in 12 hours. Starting empty with all three open, how long to fill it?",
    ],
    "code": [
        "Write a Python function that merges overlapping intervals, with type hints and a few tests.",
        "Write a bash script that finds the 10 largest files under a directory and prints their sizes in human-readable form.",
        "Implement an LRU cache in TypeScript with get and put in O(1).",
        "Write a SQL query that returns, for each customer, their most recent order and the total they spent in 2025. Explain the schema you assume.",
        "Write a Rust function that parses a line like 'key=value; key2=value2' into a HashMap, handling whitespace and errors.",
        "Write a Python asyncio script that downloads a list of URLs with at most 5 concurrent requests and retries failures twice.",
        "Write a C function that reverses a singly linked list in place, and a small main to test it.",
        "Refactor this into idiomatic Python and explain the changes:\n\nresult = []\nfor i in range(len(items)):\n    if items[i] != None:\n        if items[i] % 2 == 0:\n            result.append(items[i] * items[i])\nprint(result)",
    ],
    "writing": [
        "Write a short story (about 500 words) about a lighthouse keeper who receives a letter addressed to someone who died a century ago.",
        "Write a persuasive op-ed arguing that cities should plant more trees, about 400 words.",
        "Write a product description for a handmade ceramic coffee mug, warm but not over the top.",
        "Write a cover letter for a junior data analyst position, for someone switching careers from teaching.",
        "Write a poem about the first snow of winter in a quiet town.",
        "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.",
        "Write a friendly email to a neighbor asking them to keep their dog from barking late at night.",
        "Write the opening scene of a mystery novel set on a night train across Europe.",
    ],
    "cjk": [
        "请用中文解释什么是通货膨胀，以及它对普通家庭的影响。",
        "用中文写一篇关于秋天的短文，大约300字。",
        "请用中文介绍一下如何开始学习编程，给初学者一些建议。",
        "日本語で、日本の四季の特徴を説明してください。",
        "日本語で、初めて東京を訪れる人へのアドバイスを書いてください。",
        "请用中文比较一下高铁和飞机出行的优缺点。",
        "用中文写一封感谢老师的信。",
        "日本語で、健康的な朝ごはんのアイデアを5つ紹介してください。",
    ],
    "structured": [
        "Return a JSON array of 8 fictional employees with fields name, department, start_date (ISO), salary, and skills (list). Only the JSON.",
        "Make a markdown table comparing 6 programming languages by typing, memory management, typical use, and package manager.",
        "Convert this to YAML: a web service named shop with 3 replicas, image shop:1.4, env DB_HOST=db and LOG_LEVEL=info, ports 80->8080, and a health check on /healthz every 10s.",
        "Write a JSON schema for a recipe with title, servings, ingredients (name, quantity, unit), and steps.",
        "List the planets of the solar system as CSV with columns name, mean_radius_km, moons, has_rings.",
        "Produce an OpenAPI 3 YAML snippet for a GET /books/{id} endpoint returning a Book object with id, title, author, year.",
        "Give me a weekly workout plan as a markdown table: day, focus, exercises, sets x reps.",
        "Return a JSON object mapping 10 common HTTP status codes to a short description.",
    ],
}


def main():
    out = open(sys.argv[1], "a")
    port = sys.argv[2] if len(sys.argv) > 2 else "8112"
    for kind, prompts in P.items():
        for i, p in enumerate(prompts):
            body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": p}], "max_tokens": 2048,
                    "stream": False}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=1200))
            u = r.get("usage", {})
            m = r["choices"][0]["message"]
            rec = {"kind": kind, "i": i, "half": "tune" if i % 2 == 0 else "held", "t0": t0, "s": round(time.time() - t0, 2),
                   "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                   "finish": r["choices"][0].get("finish_reason"), "content": m.get("content"),
                   "reasoning": m.get("reasoning_content") or m.get("reasoning")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{kind} {i}: {u.get('prompt_tokens')} + {u.get('completion_tokens')} tokens in {rec['s']} s "
                  f"({rec['finish']})", flush=True)


if __name__ == "__main__":
    main()
