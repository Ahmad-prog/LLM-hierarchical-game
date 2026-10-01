"""Second-round analyses for the paper, computed from the game logs.

  python analysis/revision2.py <results_dir> <out_json>

  A. stated intentions re-scored with ONE rule for every model (regular expression), split into
     first-person commitments and other statements, at three thresholds (3, 5, 8 tokens), plus the
     share of messages that state a number
  B. vote-contingent offers split by an automatic rule into: reply/refusal, targeted private benefit,
     reciprocal political support, public contribution pledge, other; plus offers naming an amount
     the rules do not allow (contributions above 20, sanctions above the budget of 10)
  C. does the incumbent's record in the block before an election predict re-election?
  D. two-way analysis of variance over games: model x institution
  E. worker-only contributions in homogeneous fixed-manager games (for the cross-rule comparison)
  F. Nemotron outcomes over valid replies only
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np
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
ELECTED = ["elected", "salary", "costly", "hidden", "anonymous"]
STRICT_DEAL = re.compile(r"\bvot\w*\b.{0,120}\b(reward|bonus|token|pay|benefit)|\b(reward|bonus|token|pay|benefit)\w*\b.{0,120}\bvot", re.I | re.S)
FIRST_PERSON = re.compile(r"\bI(?:'ll| will| am going to|'m going to| plan to| intend to| commit| am contributing|'m contributing| shall)\b", re.I)
CONDITIONAL = re.compile(r"\b(if|as long as|provided|unless|only when|in return)\b", re.I)

# offer categories (automatic, priority order)
REPLY = re.compile(r"\b(appreciate your|thanks? (you )?for (your|the) (offer|deal|message)|your (private )?(deal|offer|proposal)|I (can(?:no|')t|won't|will not) (commit|promise|accept)|declin\w+|not (going|able) to vote)\b", re.I)
TARGETED = re.compile(r"\b(reward you|rewarding you|give you|pay you|bonus (to|for) you|you(?:'ll| will) (get|receive|earn)|(extra|additional) (tokens?|rewards?|bonus) (to|for) you|reward (your|for your) (vote|support|loyalty)|send you)\b", re.I)
RECIPROCAL = re.compile(r"\b(vote for you|support (you|your candidacy|your bid)|back you|your turn (as|to be) manager|make you manager)\b", re.I)
PLEDGE = re.compile(r"\b(I(?:'ll| will| commit to| promise to| pledge to)\s+(contribute|put|give|invest|match|keep)|contribut\w* (my )?(full|all|\d+)|maximi[sz]e (our|the|group|collective))", re.I)
OVER_CONTRIB = re.compile(r"\bcontribut\w*\s+(?:my\s+)?(?:full\s+|all\s+)?(\d+)", re.I)
OVER_SANCTION = re.compile(r"\b(?:reward|bonus|give)\w*\s+(?:you\s+)?(?:with\s+)?(\d+)\s*(?:tokens?)?", re.I)


def where(m):
    return "oss" if m in OPEN else "api"


def load(sub, prefix):
    return [json.load(open(f, encoding="utf-8"))["trials"][0]
            for f in sorted(glob.glob(os.path.join(R, sub, prefix + "__t*.json")))]


def games(m, s):
    return load(where(m), SETUPS[s].format(m=m))


# ---------------------------------------------------------------- A. stated intentions
def intentions(ts):
    c = Counter()
    for t in ts:
        for rd in t["rounds"]:
            con = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
            pub = {}
            for msg in rd.get("comm_messages") or []:
                if msg.get("phase") == "pre_action" and msg.get("recipients") is None and msg.get("content"):
                    pub[msg["sender"]] = msg["content"]
            c["messages"] += len(pub)
            for aid, text in pub.items():
                cl = _classify_with_regex(text)
                if cl["promise_type"] != "explicit":
                    continue
                v, actual = cl["extracted_value"], con.get(aid, 0.0)
                fp = bool(FIRST_PERSON.search(text)) and not CONDITIONAL.search(text)
                c["stated"] += 1
                c["first_person"] += fp
                for thr in (3, 5, 8):
                    b = actual < v - thr
                    c[f"broken_{thr}"] += b
                    if fp:
                        c[f"fp_broken_{thr}"] += b
    return dict(c)


# ---------------------------------------------------------------- B. offers
def offer_category(text):
    if REPLY.search(text):
        return "reply"
    if TARGETED.search(text):
        return "targeted"
    if RECIPROCAL.search(text):
        return "reciprocal"
    if PLEDGE.search(text):
        return "pledge"
    return "other"


def impossible(text):
    over_c = any(int(x) > 20 for x in OVER_CONTRIB.findall(text) if x.isdigit())
    over_s = any(10 < int(x) <= 100 for x in OVER_SANCTION.findall(text) if x.isdigit())
    return over_c or over_s


def offers(ts):
    c = Counter()
    for t in ts:
        for rd in t["rounds"]:
            for msg in rd.get("comm_messages") or []:
                text = msg.get("content") or ""
                if msg.get("recipients") and STRICT_DEAL.search(text):
                    c["offers"] += 1
                    c[offer_category(text)] += 1
                    c["impossible"] += impossible(text)
    return dict(c)


def over_twenty_mentions(ts):
    """Messages (public or private) that mention contributing more than 20 tokens."""
    n = k = 0
    for t in ts:
        for rd in t["rounds"]:
            for msg in rd.get("comm_messages") or []:
                text = msg.get("content") or ""
                if not text:
                    continue
                n += 1
                k += any(int(x) > 20 for x in OVER_CONTRIB.findall(text) if x.isdigit())
    return {"messages": n, "over_20": k}


# ---------------------------------------------------------------- C. re-election
def reelection_records(ts):
    out = []
    for t in ts:
        rounds = {r["round_num"]: r for r in t["rounds"]}
        hist = t["election_history"]
        for i in range(1, len(hist)):
            inc, e = hist[i - 1]["winner"], hist[i]
            r0 = e["round_num"]
            block = [rounds[k] for k in range(r0 - 5, r0) if k in rounds]
            if not block:
                continue
            group = st.fmean(a["contribution"] for r in block for a in r["agents"])
            own = st.fmean(a["contribution"] for r in block for a in r["agents"] if a["agent_id"] == inc)
            rew = st.fmean(sum(float(v) for v in ((r.get("manager_action") or {}).get("reward") or {}).values()
                               if isinstance(v, (int, float))) for r in block)
            pun = st.fmean(sum(float(v) for v in ((r.get("manager_action") or {}).get("punish") or {}).values()
                               if isinstance(v, (int, float))) for r in block)
            votes = Counter((e.get("votes") or {}).values()).most_common()
            tie = len(votes) > 1 and votes[0][1] == votes[1][1]
            out.append({"reelected": e["winner"] == inc, "tie": tie, "group_contrib": group,
                        "own_contrib": own, "reward": rew, "punish": pun})
    return out


def reelection_summary(models_sets):
    """Pooled over models after centring each predictor within model (so model differences do not drive it)."""
    rows = []
    for m, sets in models_sets:
        recs = [r for s in sets for r in reelection_records(games(m, s))]
        if not recs:
            continue
        for k in ("group_contrib", "own_contrib", "reward", "punish"):
            mu = st.fmean(r[k] for r in recs)
            for r in recs:
                r[k + "_c"] = r[k] - mu
        rows.extend(recs)
    res = {"contested": len(rows), "reelected": sum(r["reelected"] for r in rows)}
    for k in ("group_contrib", "own_contrib", "reward", "punish"):
        a = [r[k + "_c"] for r in rows if r["reelected"]]
        b = [r[k + "_c"] for r in rows if not r["reelected"]]
        pb = stats.pointbiserialr([int(r["reelected"]) for r in rows], [r[k + "_c"] for r in rows])
        res[k] = {"reelected_mean_c": round(st.fmean(a), 3) if a else None,
                  "replaced_mean_c": round(st.fmean(b), 3) if b else None,
                  "r": round(float(pb.statistic), 3), "p": round(float(pb.pvalue), 4)}
    return res


# ---------------------------------------------------------------- D. two-way ANOVA (type II, nested OLS)
def anova2(cells):
    """cells: list of (model, condition, value). Returns partial eta^2 and p for model, condition, interaction."""
    ms = sorted({c[0] for c in cells})
    cs = sorted({c[1] for c in cells})
    y = np.array([c[2] for c in cells], float)

    def design(main_m, main_c, inter):
        cols = [np.ones(len(cells))]
        if main_m:
            cols += [np.array([c[0] == m for c in cells], float) for m in ms[1:]]
        if main_c:
            cols += [np.array([c[1] == k for c in cells], float) for k in cs[1:]]
        if inter:
            cols += [np.array([c[0] == m and c[1] == k for c in cells], float) for m in ms[1:] for k in cs[1:]]
        return np.column_stack(cols)

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), np.linalg.matrix_rank(X)

    full, rk_full = rss(design(1, 1, 1))
    df_err = len(y) - rk_full
    out = {"n_games": len(y), "df_error": int(df_err)}
    for name, reduced, ref in (("model", design(0, 1, 0), design(1, 1, 0)),
                               ("institution", design(1, 0, 0), design(1, 1, 0)),
                               ("interaction", design(1, 1, 0), design(1, 1, 1))):
        r_red, k_red = rss(reduced)
        r_ref, k_ref = rss(ref)
        ss, df = r_red - r_ref, k_ref - k_red
        f = (ss / df) / (full / df_err)
        out[name] = {"ss": round(ss, 2), "df": int(df), "F": round(f, 2), "p": float(f"{stats.f.sf(f, df, df_err):.2e}"),
                     "partial_eta2": round(ss / (ss + full), 3)}
    return out


def game_value(t, metric):
    recs = [a for r in t["rounds"] for a in r["agents"]]
    if metric == "contrib":
        return st.fmean(a["contribution"] for a in recs)
    return 100 * sum(a["contribution"] > 0 for a in recs) / len(recs)


def anova_block(models, conds, metric):
    cells = [(m, s, game_value(t, metric)) for m in models for s in conds for t in games(m, s)]
    return anova2(cells)


# ---------------------------------------------------------------- E/F
def worker_contrib(ts):
    vals = [st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"] if not a["is_manager"]) for t in ts]
    return {"mean": round(st.fmean(vals), 2), "sd": round(st.stdev(vals), 2) if len(vals) > 1 else 0.0, "n": len(vals)}


def valid_only(ts):
    coop, contrib, valid = [], [], []
    for t in ts:
        recs = [a for r in t["rounds"] for a in r["agents"]]
        ok = [a for a in recs if a.get("parse_ok", True)]
        valid.append(100 * len(ok) / len(recs))
        coop.append(100 * sum(a["contribution"] > 0 for a in ok) / len(ok))
        contrib.append(st.fmean(a["contribution"] for a in ok))
    agg = lambda v: {"mean": round(st.fmean(v), 1), "sd": round(st.stdev(v), 1) if len(v) > 1 else 0.0}
    return {"valid_pct": agg(valid), "coop_valid": agg(coop), "contrib_valid": agg(contrib)}


res = {"intentions": {}, "offers": {}, "over20": {}, "reelection": {}, "anova": {}, "workers_fixed": {}, "nemotron_valid": {}}
for grp, models in (("frontier", FRONTIER), ("samegen", SAMEGEN), ("open", OPEN)):
    res["intentions"][grp] = {s: {m: intentions(games(m, s)) for m in models}
                              for s in ("chat_only", "elected", "hidden", "anonymous")}
    res["offers"][grp] = {s: {m: offers(games(m, s)) for m in models} for s in ("elected", "salary", "costly")}
    res["over20"][grp] = {m: over_twenty_mentions(sum((games(m, s) for s in SETUPS), [])) for m in models}
res["reelection"]["api_main"] = reelection_summary([(m, ELECTED) for m in FRONTIER])
res["reelection"]["newer"] = reelection_summary([(m, ELECTED) for m in SAMEGEN])
res["reelection"]["open"] = reelection_summary([(m, ELECTED) for m in OPEN])
res["reelection"]["all"] = reelection_summary([(m, ELECTED) for m in FRONTIER + SAMEGEN + OPEN])
res["anova"]["api_contrib_mgr"] = anova_block(FRONTIER, ["chat_only", "fixed", "elected", "rotating"], "contrib")
res["anova"]["api_contrib_all"] = anova_block(FRONTIER, list(SETUPS), "contrib")
res["anova"]["open_coop_mgr"] = anova_block(OPEN, ["chat_only", "fixed", "elected", "rotating"], "coop")
res["anova"]["open_coop_all"] = anova_block(OPEN, list(SETUPS), "coop")
for m in ("qwen", "claude", "grok", "gpt4o"):
    res["workers_fixed"][m] = worker_contrib(games(m, "fixed"))
res["nemotron_valid"] = {s: valid_only(games("nemotron-3-super-120b", s)) for s in SETUPS}


def cross_worker_means(name):
    return [st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"] if not a["is_manager"])
            for t in load("api", f"batch11_mgr_{name}")]


res["crossrule_tests"] = {}
for a, b in (("claude_qwen", "gpt4o_qwen"), ("claude_qwen", "grok_qwen"), ("grok_claude", "qwen_claude")):
    xa, xb = cross_worker_means(a), cross_worker_means(b)
    res["crossrule_tests"][f"{a}_vs_{b}"] = {"diff": round(st.fmean(xa) - st.fmean(xb), 2),
                                             "p": round(float(stats.ttest_ind(xa, xb, equal_var=False).pvalue), 4)}

json.dump(res, open(OUT, "w"), indent=1)

# ---------------------------------------------------------------- readable print
print("== A. stated intentions (regex for all): stated/messages | broken at 3/5/8 | first-person: n, broken at 5")
for grp, d in res["intentions"].items():
    for s, mm in d.items():
        print(f"   {grp:8s} {s:9s} " + " | ".join(
            f"{m}: {c.get('stated', 0)}/{c.get('messages', 0)} b{c.get('broken_3', 0)}/{c.get('broken_5', 0)}/{c.get('broken_8', 0)} "
            f"fp{c.get('first_person', 0)} b{c.get('fp_broken_5', 0)}" for m, c in mm.items()))
print("\n== B. offers by category (reply, targeted, reciprocal, pledge, other; impossible)")
for grp, d in res["offers"].items():
    for s, mm in d.items():
        print(f"   {grp:8s} {s:7s} " + " | ".join(
            f"{m}: {c.get('offers', 0)} [{c.get('reply', 0)},{c.get('targeted', 0)},{c.get('reciprocal', 0)},{c.get('pledge', 0)},{c.get('other', 0)}; imp {c.get('impossible', 0)}]"
            for m, c in mm.items()))
print("\n== over-20 contribution mentions", json.dumps(res["over20"]))
print("\n== C. re-election", json.dumps(res["reelection"], indent=1))
print("\n== D. ANOVA", json.dumps(res["anova"], indent=1))
print("\n== E. worker-only fixed", res["workers_fixed"])
print("\n== F. nemotron valid only", json.dumps(res["nemotron_valid"]))
