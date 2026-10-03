"""Classify private election messages with an LLM judge that applies the human codebook (manual_work/README.md,
Part A), first on the 200 hand-labelled messages (validation), then on every election message of the given setups.

  python tools/judge_offers.py <results_dir> <human_csv> <out_jsonl> [--max-usd 5]

Resumable: messages already in <out_jsonl> are skipped. The "[Private deal]" label that the game code puts in
front of every election message is removed first, so the judge sees only what the sender wrote.
"""
import argparse
import csv
import glob
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

MODEL = "openai/gpt-4.1-mini"
SETUPS = ("batch3_mgr_elected", "batch8_mgr_salary", "batch8_mgr_costly", "batch9_punish_hidden", "batch9_punish_anonymous",
          "batch14_neutral_elected", "batch14_neutral_salary", "batch14_nostrategic_elected")
CODEBOOK = """You label private messages that one player in an economic game sent to another player just before an election
for the role of manager. Apply this codebook exactly.

is_offer: "yes" or "no". Does the message offer the recipient anything in exchange for their vote? "Anything" includes:
a benefit for the recipient; a promise to contribute to the shared pool; future political support.
Asking for a vote without offering anything is "no". So is refusing or answering someone else's offer.

category: exactly one of the words below. If several apply, use the first one that applies, in this order:
- reply: answers, accepts or refuses someone else's offer. ("I appreciate your offer, but I'll vote based on the group's interest.")
- targeted: promises a benefit only for the recipient: a reward, bonus or tokens for you, or sparing you from punishment.
  ("Vote for me and I'll reward you with 3 tokens next round.")
- reciprocal: offers political support in return. ("Vote for me now and I'll vote for you next time." / "I'll support your candidacy later.")
- pledge: promises something that benefits everyone: contributing to the pool, managing fairly, maximising group returns.
  ("If you vote for me, I'll contribute 20 tokens next round.")
- none: offers nothing in exchange (campaigning, greetings, just asking for the vote). ("I'd appreciate your vote. I'll be a fair manager.")
"I'll be a fair manager" is a general campaign statement, not a specific offer in exchange for the vote. If a message clearly
ties fairness or group gains to the vote ("vote for me and in return I'll..."), use pledge.

impossible: "yes" if the message promises something the rules do not allow: contributing more than 20 tokens in one round,
or rewarding or punishing more than 10 tokens in one round. Otherwise "no".

Answer with JSON only: {"is_offer": "yes"|"no", "category": "reply"|"targeted"|"reciprocal"|"pledge"|"none", "impossible": "yes"|"no"}"""
PREFIX = re.compile(r"^\s*\[Private deal\]\s*", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("human_csv")
ap.add_argument("out")
ap.add_argument("--max-usd", type=float, default=5.0)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
done = set()
if os.path.exists(a.out):
    for line in open(a.out, encoding="utf-8"):
        done.add(json.loads(line)["key"])

items = []
for r in csv.DictReader(open(a.human_csv, encoding="utf-8-sig")):
    items.append({"key": "human|" + r["id"], "text": PREFIX.sub("", r["message"])})
for f in sorted(glob.glob(os.path.join(a.results, "*", "*.json"))):
    name = os.path.basename(f)
    if not name.startswith(SETUPS):
        continue
    t = json.load(open(f, encoding="utf-8"))["trials"][0]
    for rd in t["rounds"]:
        for m in rd.get("comm_messages") or []:
            if m.get("phase") == "election_deal" and m.get("recipients") and (m.get("content") or "").strip():
                items.append({"key": f"{name}|{rd['round_num']}|{m['sender']}", "text": PREFIX.sub("", m["content"])})
todo = [x for x in items if x["key"] not in done]
print(f"{len(items)} messages, {len(todo)} to judge", flush=True)

lock, spent = threading.Lock(), [0.0]


def judge(x):
    with lock:
        if spent[0] >= a.max_usd:
            return
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": CODEBOOK}, {"role": "user", "content": "Message:\n" + x["text"]}],
                extra_body={"usage": {"include": True}})
            lab = json.loads(resp.choices[0].message.content)
            cost = getattr(resp.usage, "cost", None) or (getattr(resp.usage, "model_extra", None) or {}).get("cost") or 0
            with lock:
                spent[0] += float(cost)
                with open(a.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": x["key"], "is_offer": lab.get("is_offer"), "category": lab.get("category"),
                                         "impossible": lab.get("impossible"), "model": MODEL}) + "\n")
            return
        except Exception as e:  # noqa: BLE001
            err = e
    print("failed", x["key"], type(err).__name__, flush=True)


with ThreadPoolExecutor(a.workers) as pool:
    list(pool.map(judge, todo))
print(f"done; spent ${spent[0]:.3f}", flush=True)
