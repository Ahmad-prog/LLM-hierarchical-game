"""Clean-label neutral prompt (Claude), GPT-5 chat-only endgame, placebo worker contribution, fair-ballot manager effects.

  python analysis/round8.py <results> <offers_llm_judge.jsonl> <out.json>
"""
import glob, json, os, statistics as st, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.deception import _classify_with_regex  # noqa: E402

R, J, OUT = sys.argv[1:4]
JL = {d["key"]: d for d in map(json.loads, open(J, encoding="utf-8"))}
yes = lambda d: str((d or {}).get("is_offer", "")).lower().startswith("y")
load = lambda s, m: [(os.path.basename(f), json.load(open(f, encoding="utf-8"))["trials"][0])
                     for f in sorted(glob.glob(os.path.join(R, "*", f"{s}_{m}__t*.json")))]


def perm(a, b, two=True, reps=100000, seed=20261004):
    rng, pool, na = np.random.default_rng(seed), np.array(a + b, float), len(a)
    obs, h = np.mean(b) - np.mean(a), 0
    for _ in range(reps):
        x = rng.permutation(pool)
        d = x[na:].mean() - x[:na].mean()
        h += (abs(d) >= abs(obs) - 1e-12) if two else (d >= obs - 1e-12)
    return round((h + 1) / (reps + 1), 4)


def holm(ps):
    o, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(ps) - k) * ps[i]))
        adj[i] = round(run, 4)
    return adj


def offers(name, t, first=False):
    n = 5 * (1 if first else len(t["election_history"]))
    k = sum(1 for rd in t["rounds"] if not first or rd["round_num"] == 1 for m in (rd.get("comm_messages") or [])
            if m.get("phase") == "election_deal" and m.get("recipients") and yes(JL.get(f"{name}|{rd['round_num']}|{m['sender']}")))
    return 100 * k / n


res = {}
for first in (False, True):
    a = [offers(n, t, first) for n, t in load("batch17_neutral_clean_elected", "claude")]
    b = [offers(n, t, first) for n, t in load("batch17_neutral_clean_salary", "claude")]
    res["clean_claude_" + ("first" if first else "all")] = {"none": round(np.mean(a), 1), "salary": round(np.mean(b), 1), "n": [len(a), len(b)],
                                                            "p_one_sided": perm(a, b, two=False)}

b20 = s20 = bb = sb = 0
g = load("batch2_comm_full", "gpt5")
for _, t in g:
    T = len(t["rounds"])
    for rd in t["rounds"]:
        con = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
        pub = {m["sender"]: m["content"] for m in (rd.get("comm_messages") or [])
               if m.get("phase") == "pre_action" and m.get("recipients") is None and m.get("content")}
        for aid, text in pub.items():
            cl = _classify_with_regex(text)
            if cl["promise_type"] == "explicit":
                br = con.get(aid, 0) < cl["extracted_value"] - 5
                if rd["round_num"] == T:
                    s20, b20 = s20 + 1, b20 + br
                else:
                    sb, bb = sb + 1, bb + br
last = [st.fmean(a["contribution"] for a in t["rounds"][-1]["agents"]) for _, t in g]
res["gpt5_chat"] = {"games": len(g), "round20_broken": f"{b20}/{s20}", "before_broken": f"{bb}/{sb}", "round20_mean_contribution": round(np.mean(last), 1)}


def workers(t, skip):
    return st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"] if not a.get("is_manager") and a["agent_id"] not in skip)


res["placebo"] = {}
for m in ("gemini", "qwen"):
    p10 = [workers(t, {"agent_1"}) for _, t in load("batch17_mgr_fixed_peer10", m)]
    p20 = [workers(t, {"agent_1"}) for _, t in load("batch17_mgr_fixed_peer20", m)]
    m10 = [workers(t, set()) for _, t in load("batch16_mgr_fixed_contrib10", m)]
    m20 = [workers(t, set()) for _, t in load("batch16_mgr_fixed_contrib20", m)]
    res["placebo"][m] = {"peer10": round(np.mean(p10), 2), "peer20": round(np.mean(p20), 2), "peer_diff": round(np.mean(p20) - np.mean(p10), 2),
                         "p_peer": perm(p10, p20), "mgr_diff": round(np.mean(m20) - np.mean(m10), 2), "n": [len(p10), len(p20)]}

rows = []
for m in ["gpt4o", "gemini", "qwen", "deepseek", "gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]:
    c = [st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"]) for _, t in load("batch2_comm_full", m)]
    f = [st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"]) for _, t in load("batch13_ballot_random", m)]
    if len(f) > 1:
        rows.append((m, round(np.mean(c), 1), round(np.mean(f), 1), len(f), round(np.mean(f) - np.mean(c), 1), perm(c, f)))
adj = holm([r[-1] for r in rows])
res["fair_ballot_manager"] = {r[0]: {"chat": r[1], "fair_elected": r[2], "n": r[3], "diff": r[4], "p": r[5], "holm": h} for r, h in zip(rows, adj)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "fair_ballot_manager"}, indent=1))
