"""Extra analyses for the paper, computed from the result files.

  python analysis/revision_extras.py <results_dir> <out_json>

  A. manager spending split into punishment and reward (requested, capped at the per-round budget)
  B. elections: winner vote counts, majorities, ties, self-votes, turnover
  C. promise-breaking re-scored deterministically from the public messages (regex only), with and
     without conditional promises, next to the logged (GPT-4o-mini first, regex fallback) events
  D. welfare split into the contribution part (0.6 x contributions) and transfers (sanctions, salary)
  E. Welch tests with 95% CIs for the key contrasts (game = unit)
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.deception import _classify_with_regex  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
FRONTIER = ["gpt4o", "claude", "gemini", "deepseek", "grok", "qwen"]
SAMEGEN = ["gpt5", "deepseek31"]
OPEN = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]
SETUPS = {
    "baseline": "batch1_baseline_{m}", "chat_only": "batch2_comm_full_{m}",
    "fixed": "batch3_mgr_fixed_{m}", "elected": "batch3_mgr_elected_{m}", "rotating": "batch3_mgr_rotating_{m}",
    "salary": "batch8_mgr_salary_{m}", "costly": "batch8_mgr_costly_{m}",
    "hidden": "batch9_punish_hidden_{m}", "anonymous": "batch9_punish_anonymous_{m}",
}
MANAGED = ["fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"]
ELECTED = ["elected", "salary", "costly", "hidden", "anonymous"]
COND = re.compile(r"\b(if|as long as|provided|unless|only when|in return)\b", re.I)
STRICT_DEAL = re.compile(r"\bvot\w*\b.{0,120}\b(reward|bonus|token|pay|benefit)|\b(reward|bonus|token|pay|benefit)\w*\b.{0,120}\bvot", re.I | re.S)


def load(sub, prefix):
    return [json.load(open(f, encoding="utf-8"))["trials"][0]
            for f in sorted(glob.glob(os.path.join(R, sub, prefix + "__t*.json")))]


def where(m):
    return "oss" if m in OPEN else "api"


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(st.fmean(xs), 3), "sd": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0, "n": len(xs)}


# ---------------------------------------------------------------- A. spending split
def spend(t):
    p = r = 0.0
    rounds_p = 0
    for rd in t["rounds"]:
        a = rd.get("manager_action") or {}
        pp = min(10.0, sum(float(v) for v in (a.get("punish") or {}).values() if isinstance(v, (int, float))))
        rr = min(10.0, sum(float(v) for v in (a.get("reward") or {}).values() if isinstance(v, (int, float))))
        p, r = p + pp, r + rr
        rounds_p += pp > 0
    n = len(t["rounds"])
    return p / n, r / n, 100 * rounds_p / n


def spend_block(models, setups):
    out = {}
    for s in setups:
        out[s] = {}
        for m in models:
            g = [spend(t) for t in load(where(m), SETUPS[s].format(m=m))]
            if g:
                out[s][m] = {"punish": ms([x[0] for x in g]), "reward": ms([x[1] for x in g]),
                             "pct_rounds_punish": ms([x[2] for x in g])}
    return out


# ---------------------------------------------------------------- B. elections
def elections(games):
    c = Counter()
    for t in games:
        prev = None
        for e in t["election_history"]:
            votes = e.get("votes") or {}
            cnt = Counter(votes.values())
            top = cnt.most_common()
            wv = cnt.get(e["winner"], 0)
            c["elections"] += 1
            c["winner_votes"] += wv
            c["majority"] += wv >= 3
            c["unanimous"] += wv == 5
            c["tie_at_top"] += len(top) > 1 and top[0][1] == top[1][1]
            c["self_votes"] += sum(1 for v, w in votes.items() if v == w)
            c["ballots"] += len(votes)
            c["candidates_with_votes"] += len(cnt)
            c["first_index_wins"] += e["winner"] == "agent_0"
            if prev is not None:
                c["contested"] += 1
                c["turnovers"] += e["winner"] != prev
                c["turnovers_by_tie"] += (e["winner"] != prev) and len(top) > 1 and top[0][1] == top[1][1]
                c["incumbent_votes"] += cnt.get(prev, 0)
            prev = e["winner"]
    if not c["elections"]:
        return None
    return {"elections": c["elections"], "contested": c["contested"], "turnovers": c["turnovers"],
            "turnovers_by_tie": c["turnovers_by_tie"],
            "pct_agent0_wins": round(100 * c["first_index_wins"] / c["elections"], 1),
            "mean_winner_votes": round(c["winner_votes"] / c["elections"], 2),
            "pct_majority": round(100 * c["majority"] / c["elections"], 1),
            "pct_unanimous": round(100 * c["unanimous"] / c["elections"], 1),
            "pct_tie_at_top": round(100 * c["tie_at_top"] / c["elections"], 1),
            "pct_self_votes": round(100 * c["self_votes"] / max(1, c["ballots"]), 1),
            "mean_candidates_with_votes": round(c["candidates_with_votes"] / c["elections"], 2)}


# ---------------------------------------------------------------- C. promise re-score
def promises(games):
    c = Counter()
    for t in games:
        for rd in t["rounds"]:
            contrib = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
            pub = {}
            for msg in rd.get("comm_messages") or []:
                if msg.get("phase") == "pre_action" and msg.get("recipients") is None and msg.get("content"):
                    pub[msg["sender"]] = msg["content"]
            for aid, text in pub.items():
                cl = _classify_with_regex(text)
                if cl["promise_type"] != "explicit":
                    continue
                broken = contrib.get(aid, 0.0) < cl["extracted_value"] - 5
                cond = bool(COND.search(text))
                c["explicit"] += 1
                c["broken"] += broken
                c["conditional"] += cond
                if not cond:
                    c["explicit_uncond"] += 1
                    c["broken_uncond"] += broken
        for e in t["deception"]["events"]:
            if e["promise_type"] == "explicit" and e.get("stated_intention") is not None:
                c["logged_explicit"] += 1
                c["logged_broken"] += e["actual_contribution"] < e["stated_intention"] - 5
    return dict(c)


# ---------------------------------------------------------------- D. welfare split
def welfare_split(t):
    recs = [a for r in t["rounds"] for a in r["agents"]]
    total = sum(a["net_payoff"] for a in recs) - 5 * 20 * len(t["rounds"])
    contrib_part = 0.6 * sum(a["contribution"] for a in recs)
    return total, contrib_part, total - contrib_part


def welfare_block(games):
    w = [welfare_split(t) for t in games]
    return {"total": ms([x[0] for x in w]), "from_contributions": ms([x[1] for x in w]),
            "transfers_and_salary": ms([x[2] for x in w])}


# ---------------------------------------------------------------- E. tests
def per_game_metric(t, k):
    recs = [a for r in t["rounds"] for a in r["agents"]]
    if k == "contrib":
        return st.fmean(a["contribution"] for a in recs)
    if k == "coop":
        return 100 * sum(a["contribution"] > 0 for a in recs) / len(recs)
    if k == "deals":
        n = sum(1 for r in t["rounds"] for m in (r.get("comm_messages") or [])
                if m.get("recipients") and STRICT_DEAL.search(m.get("content") or ""))
        return 100 * n / (5 * len(t["rounds"]))
    if k == "broken_rate":
        ev = [e for e in t["deception"]["events"] if e["promise_type"] == "explicit" and e.get("stated_intention") is not None]
        return 100 * sum(e["actual_contribution"] < e["stated_intention"] - 5 for e in ev) / len(ev) if ev else None
    raise KeyError(k)


def welch(m, k, a, b):
    xa = [x for x in (per_game_metric(t, k) for t in load(where(m), SETUPS[a].format(m=m))) if x is not None]
    xb = [x for x in (per_game_metric(t, k) for t in load(where(m), SETUPS[b].format(m=m))) if x is not None]
    if len(xa) < 2 or len(xb) < 2:
        return None
    d = st.fmean(xb) - st.fmean(xa)
    va, vb = st.variance(xa) / len(xa), st.variance(xb) / len(xb)
    if va + vb == 0:
        return {"model": m, "metric": k, "from": a, "to": b, "diff": round(d, 2), "p": None, "ci": None, "note": "no variance"}
    se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / ((va ** 2 / (len(xa) - 1) if va else 0) + (vb ** 2 / (len(xb) - 1) if vb else 0))
    tq = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(d / se), df)
    return {"model": m, "metric": k, "from": a, "to": b, "diff": round(d, 2), "p": round(p, 4),
            "ci": [round(d - tq * se, 2), round(d + tq * se, 2)], "n": [len(xa), len(xb)]}


res = {"spend": {"frontier": spend_block(FRONTIER, MANAGED), "samegen": spend_block(SAMEGEN, MANAGED),
                 "open": spend_block(OPEN, MANAGED)},
       "crossrule_spend": {}, "elections": {}, "promises": {}, "welfare": {}, "tests": []}

for f in sorted({os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(R, "api", "batch11_*__t*.json"))}):
    g = [spend(t) for t in load("api", f)]
    res["crossrule_spend"][f.replace("batch11_mgr_", "")] = {"punish": ms([x[0] for x in g]), "reward": ms([x[1] for x in g])}
    res["welfare"]["crossrule_" + f.replace("batch11_mgr_", "")] = welfare_block(load("api", f))

for grp, models in (("frontier", FRONTIER), ("samegen", SAMEGEN), ("open", OPEN)):
    res["elections"][grp] = {m: elections(sum((load(where(m), SETUPS[s].format(m=m)) for s in ELECTED), [])) for m in models}
    res["elections"][grp + "_elected_only"] = {m: elections(load(where(m), SETUPS["elected"].format(m=m))) for m in models}
res["elections"]["mixed"] = elections(sum((load("api", f"batch4_hetero_all_drop_{m}") for m in FRONTIER), []))

for grp, models in (("frontier", FRONTIER), ("samegen", SAMEGEN), ("open", OPEN)):
    res["promises"][grp] = {s: {m: promises(load(where(m), SETUPS[s].format(m=m))) for m in models}
                            for s in ("chat_only", "elected", "hidden", "anonymous")}

for grp, models in (("frontier", FRONTIER), ("samegen", SAMEGEN)):
    res["welfare"][grp] = {s: {m: welfare_block(load(where(m), SETUPS[s].format(m=m))) for m in models}
                           for s in ("fixed", "elected", "rotating", "salary", "costly")}

T = res["tests"]
for m in FRONTIER:
    for s in ("fixed", "elected", "rotating"):
        T.append(welch(m, "contrib", "chat_only", s))
for m in FRONTIER + SAMEGEN + ["qwen3.8-27b", "gpt-oss-120b"]:
    T.append(welch(m, "deals", "elected", "salary"))
    T.append(welch(m, "deals", "salary", "costly"))
for m in OPEN:
    T.append(welch(m, "coop", "baseline", "chat_only"))
    for s in ("fixed", "elected", "rotating"):
        T.append(welch(m, "coop", "chat_only", s))
    T.append(welch(m, "broken_rate", "chat_only", "elected"))
T[:0] = [welch(m, "contrib", "baseline", "chat_only") for m in FRONTIER]
res["tests"] = [x for x in T if x]

json.dump(res, open(OUT, "w"), indent=1)

# ---------------------------------------------------------------- readable print
for grp in ("frontier", "samegen", "open"):
    print(f"\n== SPEND punish/reward per round, {grp}")
    for s, d in res["spend"][grp].items():
        print("  ", s, " | ".join(f"{m} P{v['punish']['mean']:.1f} R{v['reward']['mean']:.1f} pr%{v['pct_rounds_punish']['mean']:.0f}" for m, v in d.items()))
print("\n== CROSSRULE spend", {k: (v["punish"]["mean"], v["reward"]["mean"]) for k, v in res["crossrule_spend"].items()})
print("\n== ELECTIONS")
for k, v in res["elections"].items():
    print("  ", k, json.dumps(v))
print("\n== PROMISES (regex re-score) explicit/broken | uncond explicit/broken | logged explicit/broken")
for grp, d in res["promises"].items():
    for s, mm in d.items():
        print(f"   {grp:8s} {s:9s} " + " | ".join(
            f"{m}: {c.get('broken', 0)}/{c.get('explicit', 0)} u{c.get('broken_uncond', 0)}/{c.get('explicit_uncond', 0)} L{c.get('logged_broken', 0)}/{c.get('logged_explicit', 0)}"
            for m, c in mm.items()))
print("\n== WELFARE split total = contributions + transfers")
for grp, d in res["welfare"].items():
    if grp.startswith("crossrule"):
        print("  ", grp, {k: v["mean"] for k, v in d.items()})
        continue
    for s, mm in d.items():
        print(f"   {grp} {s}: " + " | ".join(f"{m} {v['total']['mean']:.0f}={v['from_contributions']['mean']:.0f}+{v['transfers_and_salary']['mean']:.0f}" for m, v in mm.items()))
print("\n== TESTS (Welch, game = unit)")
for x in res["tests"]:
    print("  ", x)
