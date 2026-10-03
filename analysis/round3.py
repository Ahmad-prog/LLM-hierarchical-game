"""Round-3 analyses: one section per remaining review concern (ChatGPT review and an independent second review).

  python analysis/round3.py <results_dir> <out_json> [<human_offers_csv>]

  A  primary contrasts, fixed in advance: communication (no chat -> chat only) and manager (chat only -> elected),
     contribution, every model; Welch and exact/Monte Carlo permutation p; Holm over the whole family
  A2 pooled manager effect for the main API models (OLS with model effects, game as unit, HC3), share of headroom
  A3 two-way ANOVA (model x institution) with partial eta^2 and omega^2
  B  neutral election prompt vs the deal prompt; pay effect under the neutral prompt
  C  mechanism: manager without sanctions, scripted reward rule, automatic reward without a manager
  D  accountability: bad vs good incumbent (replaced at round 6), random ballot vs fixed ballot
  E  visibility of contributions: aggregate-only vs individual, with and without a manager
  F  version vs reasoning: GPT-4o, GPT-4.1, GPT-5, GPT-5 minimal reasoning
  G  prompt robustness: without "Be strategic."
  H  quantisation: Qwen3.8 27B full precision vs NVFP4
  I  endgame: contributions and broken intentions in the last round
  J  offer measure: population-weighted recall of the strict rule; keyword classifier vs hand labels
  K  re-election on the incumbent's record: logistic regression, standard errors clustered by game
  L  manipulation check for sanction visibility: how often agents mention sanctions
  M  outcomes conditional on valid replies
  N  LLM judge (human codebook): validation against the hand labels, and offers by prompt and pay
Games are the unit throughout.
"""
import csv
import glob
import itertools
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.manipulation import _classify_message  # noqa: E402
from analysis.deception import _classify_with_regex  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
HUMAN = sys.argv[3] if len(sys.argv) > 3 else None
JUDGE = sys.argv[4] if len(sys.argv) > 4 else None     # LLM-judge labels (tools/judge_offers.py)
STRICT = re.compile(r"\bvot\w*\b.{0,120}\b(reward|bonus|token|pay|benefit)|\b(reward|bonus|token|pay|benefit)\w*\b.{0,120}\bvot", re.I | re.S)
API_MAIN = ["gpt4o", "claude", "gemini", "deepseek", "grok", "qwen"]
NEWER = ["gpt5", "deepseek31"]
OSS = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]
BELOW = ["gpt4o", "gemini", "qwen"]       # main API models whose chat-only level leaves room for a manager
S = {"baseline": "batch1_baseline", "chat": "batch2_comm_full", "fixed": "batch3_mgr_fixed",
     "elected": "batch3_mgr_elected", "rotating": "batch3_mgr_rotating", "salary": "batch8_mgr_salary",
     "costly": "batch8_mgr_costly", "hidden": "batch9_punish_hidden", "anonymous": "batch9_punish_anonymous",
     "agg_elected": "batch13_info_aggregate", "nosanction": "batch13_mgr_nosanction", "autoreward": "batch13_mgr_autoreward",
     "bad": "batch13_mgr_badincumbent", "ballot": "batch13_ballot_random",
     "neutral": "batch14_neutral_elected", "neutral_salary": "batch14_neutral_salary",
     "nostrat_chat": "batch14_nostrategic_chat", "nostrat_elected": "batch14_nostrategic_elected",
     "agg_chat": "batch14_aggregate_chat", "good": "batch14_mgr_goodincumbent", "sysreward": "batch14_system_reward",
     "elected_nobudget": "batch15_mgr_elected_nobudget", "bad_rb": "batch15_mgr_badincumbent_rb", "good_rb": "batch15_mgr_goodincumbent_rb"}
_cache = {}


def games(m, s):
    key = (m, s)
    if key not in _cache:
        fs = []
        for sub in ("api", "oss"):
            fs += sorted(glob.glob(os.path.join(R, sub, f"{S[s]}_{m}__t*.json")))
        _cache[key] = [json.load(open(f, encoding="utf-8"))["trials"][0] for f in fs]
    return _cache[key]


# ---------------------------------------------------------------- per-game metrics
def recs(t):
    return [a for r in t["rounds"] for a in r["agents"]]


def contrib(t):
    return st.fmean(a["contribution"] for a in recs(t))


def coop(t):
    rs = recs(t)
    return 100 * sum(a["contribution"] > 0 for a in rs) / len(rs)


def private(t, phase=None):
    return [m for r in t["rounds"] for m in (r.get("comm_messages") or [])
            if m.get("recipients") and (phase is None or m.get("phase") == phase)]


def offers(t):            # the paper's measure: strict-rule private messages per 100 agent-rounds
    return 100 * sum(bool(STRICT.search(m.get("content") or "")) for m in private(t)) / (5 * len(t["rounds"]))


def offers_kw(t):
    return 100 * sum(_classify_message(m.get("content") or "") == "deal_offer" for m in private(t)) / (5 * len(t["rounds"]))


def dm_share(t):          # share of election-message opportunities used (any private election message)
    n = 5 * len(t["election_history"])
    return 100 * len(private(t, "election_deal")) / n if n else None


def offer_share(t):       # strict-rule offers as a share of election-message opportunities
    n = 5 * len(t["election_history"])
    return 100 * sum(bool(STRICT.search(m.get("content") or "")) for m in private(t, "election_deal")) / n if n else None


def stated_events(t):
    return [e for e in t["deception"]["events"] if e["promise_type"] == "explicit" and e.get("stated_intention") is not None]


FIRST_PERSON = re.compile(r"\bI(?:'ll| will| am going to|'m going to| plan to| intend to| commit| am contributing|'m contributing| shall)\b", re.I)
CONDITIONAL = re.compile(r"\b(if|as long as|provided|unless|only when|in return)\b", re.I)


def statements(t):
    """Stated intentions scored exactly as in revision2.py (Table 2): per round, each agent's last public message of
    the communication phase before the contribution decision; one regular expression for every model.
    Yields (round, speaker, stated amount, given, previous-round contribution or None, first-person commitment)."""
    prev = {}
    for rd in t["rounds"]:
        con = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
        pub = {}
        for msg in rd.get("comm_messages") or []:
            if msg.get("phase") == "pre_action" and msg.get("recipients") is None and msg.get("content"):
                pub[msg["sender"]] = msg["content"]
        for aid, text in pub.items():
            cl = _classify_with_regex(text)
            if cl["promise_type"] == "explicit":
                fp = bool(FIRST_PERSON.search(text)) and not CONDITIONAL.search(text)
                yield rd["round_num"], aid, cl["extracted_value"], con.get(aid, 0.0), prev.get(aid), fp
        prev = con


def broken_counts(ts, last=None, first_person=False):
    """broken (gave > 5 below the stated amount) and stated; last=True only round 20, False only rounds 1-19"""
    b = n = 0
    for t in ts:
        T = len(t["rounds"])
        for rnd, _, v, given, _, fp in statements(t):
            if last is True and rnd != T or last is False and rnd == T or first_person and not fp:
                continue
            n += 1
            b += given < v - 5
    return {"broken": b, "stated": n, "share": round(100 * b / n, 1) if n else None}


def welfare_parts(t):
    rs = recs(t)
    total = sum(a["net_payoff"] for a in rs) - 20 * len(rs)
    return total, 0.6 * sum(a["contribution"] for a in rs)


# ---------------------------------------------------------------- statistics
def ms(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": round(st.fmean(xs), 2), "sd": round(st.stdev(xs), 2) if len(xs) > 1 else 0.0, "n": len(xs)} if xs else None


def perm_p(a, b, reps=20000, seed=7):
    """two-sided permutation p for the difference in means (exact when the split count is small)"""
    a, b = list(a), list(b)
    pool, na = a + b, len(a)
    obs = abs(st.fmean(b) - st.fmean(a))
    total = math.comb(len(pool), na)
    if total <= 30000:
        hits = cnt = 0
        for idx in itertools.combinations(range(len(pool)), na):
            s = set(idx)
            xa = [pool[i] for i in idx]
            xb = [pool[i] for i in range(len(pool)) if i not in s]
            hits += abs(st.fmean(xb) - st.fmean(xa)) >= obs - 1e-12
            cnt += 1
        return hits / cnt
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pool)
        hits += abs(st.fmean(pool[na:]) - st.fmean(pool[:na])) >= obs - 1e-12
    return (hits + 1) / (reps + 1)


def test(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    d = st.fmean(b) - st.fmean(a)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    out = {"a": ms(a), "b": ms(b), "diff": round(d, 2)}
    if va + vb == 0:
        out.update(p=None, ci=None, perm_p=None, note="no variance")
        return out
    se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va else 0) + (vb ** 2 / (len(b) - 1) if vb else 0))
    q = stats.t.ppf(0.975, df)
    out.update(p=round(float(2 * stats.t.sf(abs(d / se), df)), 5), ci=[round(d - q * se, 2), round(d + q * se, 2)],
               perm_p=round(perm_p(a, b), 5))
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = round(run, 5)
    return adj


def metric_test(m, s1, s2, f):
    return test([f(t) for t in games(m, s1)], [f(t) for t in games(m, s2)])


res = {}

# ---------------------------------------------------------------- A. primary contrasts
prim = []
for m in API_MAIN + NEWER + OSS:
    for name, s1, s2 in (("communication", "baseline", "chat"), ("manager", "chat", "elected")):
        t = metric_test(m, s1, s2, contrib)
        if t and t.get("p") is not None:
            prim.append(dict(model=m, contrast=name, **t))
for row, h, hp in zip(prim, holm([r["p"] for r in prim]), holm([r["perm_p"] for r in prim])):
    row["holm"], row["holm_perm"] = h, hp
res["A_primary"] = {"family_size": len(prim), "tests": prim}
# cooperation versions for the self-hosted models (secondary)
res["A_oss_cooperation"] = {m: {n: metric_test(m, a, b, coop) for n, a, b in
                                (("communication", "baseline", "chat"), ("manager", "chat", "elected"))} for m in OSS}

# ---------------------------------------------------------------- A2. pooled manager effect, main API models
import statsmodels.formula.api as smf  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402


def frame(models, setups, f=contrib):
    return pd.DataFrame([{"model": m, "setup": s, "y": f(t), "managed": int(s != "chat")}
                         for m in models for s in setups for t in games(m, s)])


a2 = {}
for label, models in (("main_api", API_MAIN), ("below_ceiling", BELOW), ("at_ceiling", ["claude", "deepseek", "grok"]),
                      ("self_hosted", OSS)):
    df = frame(models, ["chat", "fixed", "elected", "rotating"])
    fit = smf.ols("y ~ C(model) + managed", df).fit(cov_type="HC3")
    full = smf.ols("y ~ C(model) * managed", df).fit()
    add = smf.ols("y ~ C(model) + managed", df).fit()
    f_int = anova_lm(add, full)
    chat = {m: st.fmean(contrib(t) for t in games(m, "chat")) for m in models}
    mgd = {m: st.fmean(contrib(t) for s in ("fixed", "elected", "rotating") for t in games(m, s)) for m in models}
    a2[label] = {"games": len(df), "managed_coef": round(float(fit.params["managed"]), 2),
                 "ci": [round(float(x), 2) for x in fit.conf_int().loc["managed"]], "p": float(fit.pvalues["managed"]),
                 "interaction_F_p": float(f_int["Pr(>F)"].iloc[1]),
                 "per_model": {m: {"chat": round(chat[m], 2), "managed": round(mgd[m], 2), "diff": round(mgd[m] - chat[m], 2),
                                   "share_of_headroom": round((mgd[m] - chat[m]) / (20 - chat[m]), 2) if chat[m] < 19.9 else None}
                               for m in models}}
res["A2_pooled_manager"] = a2


# ---------------------------------------------------------------- A3. ANOVA with omega^2
def anova_omega(models, setups, f):
    df = frame(models, setups, f)
    fit = smf.ols("y ~ C(model) * C(setup)", df).fit()
    tab = anova_lm(fit, typ=2)
    mse, sst = tab.loc["Residual", "sum_sq"] / tab.loc["Residual", "df"], float(((df.y - df.y.mean()) ** 2).sum())
    out = {"games": len(df)}
    for k, name in (("C(model)", "model"), ("C(setup)", "institution"), ("C(model):C(setup)", "interaction")):
        ss, dfe = tab.loc[k, "sum_sq"], tab.loc[k, "df"]
        out[name] = {"partial_eta2": round(ss / (ss + tab.loc["Residual", "sum_sq"]), 3),
                     "omega2": round((ss - dfe * mse) / (sst + mse), 3), "p": float(tab.loc[k, "PR(>F)"])}
    return out


res["A3_anova"] = {"main_api_contrib": anova_omega(API_MAIN, ["chat", "fixed", "elected", "rotating"], contrib),
                   "self_hosted_coop": anova_omega(OSS, ["chat", "fixed", "elected", "rotating"], coop),
                   "self_hosted_contrib": anova_omega(OSS, ["chat", "fixed", "elected", "rotating"], contrib)}

# ---------------------------------------------------------------- B. neutral election prompt
b = {}
for m in ["claude", "grok", "gemini", "deepseek31"] + OSS:
    if not games(m, "neutral"):
        continue
    row = {}
    for f, nm in ((offers, "offers"), (offers_kw, "offers_keyword"), (dm_share, "election_dm_share"), (offer_share, "offer_share")):
        row[nm] = {"deal_vs_neutral": metric_test(m, "elected", "neutral", f),
                   "deal_salary_vs_neutral_salary": metric_test(m, "salary", "neutral_salary", f),
                   "pay_effect_deal": metric_test(m, "elected", "salary", f),
                   "pay_effect_neutral": metric_test(m, "neutral", "neutral_salary", f)}
    b[m] = row
res["B_neutral_prompt"] = b

# ---------------------------------------------------------------- C. mechanism
c = {}
for m in BELOW + OSS:
    if not games(m, "sysreward"):
        continue
    row = {s: {"contrib": ms([contrib(t) for t in games(m, s)]), "coop": ms([coop(t) for t in games(m, s)]),
               "welfare": ms([welfare_parts(t)[0] for t in games(m, s)]),
               "welfare_from_contrib": ms([welfare_parts(t)[1] for t in games(m, s)])}
           for s in ("chat", "fixed", "nosanction", "autoreward", "sysreward")}
    row["tests_contrib"] = {f"{x}_vs_{y}": metric_test(m, x, y, contrib) for x, y in
                            (("chat", "fixed"), ("chat", "nosanction"), ("fixed", "nosanction"), ("chat", "autoreward"),
                             ("fixed", "autoreward"), ("chat", "sysreward"), ("autoreward", "sysreward"))}
    c[m] = row
res["C_mechanism"] = c


# ---------------------------------------------------------------- D. accountability
def replaced_at_6(ts):
    k = n = 0
    for t in ts:
        e6 = [e for e in t["election_history"] if e["round_num"] == 6]
        if e6:
            n += 1
            k += e6[0]["winner"] != "agent_0"
    return k, n


def elect_stats(ts):
    cnt = Counter()
    for t in ts:
        prev = None
        for e in t["election_history"]:
            v = Counter((e.get("votes") or {}).values()).most_common()
            cnt["elections"] += 1
            cnt["agent0_wins"] += e["winner"] == "agent_0"
            cnt["tie_top"] += len(v) > 1 and v[0][1] == v[1][1]
            if prev is not None:
                cnt["contested"] += 1
                cnt["turnovers"] += e["winner"] != prev
            prev = e["winner"]
    return dict(cnt)


d, pool_bad, pool_good, pool_fix, pool_rand = {}, [0, 0], [0, 0], [0, 0], [0, 0]
for m in ["gpt4o", "gemini", "qwen", "deepseek"] + OSS:
    if not games(m, "good"):
        continue
    kb, nb = replaced_at_6(games(m, "bad"))
    kg, ng = replaced_at_6(games(m, "good"))
    ef, er = elect_stats(games(m, "elected")), elect_stats(games(m, "ballot"))
    d[m] = {"bad_replaced_at_6": [kb, nb], "good_replaced_at_6": [kg, ng],
            "fisher_p": float(stats.fisher_exact([[kb, nb - kb], [kg, ng - kg]])[1]),
            "agent0_rounds_in_office": {"bad": ms([sum(r["manager_id"] == "agent_0" for r in t["rounds"]) for t in games(m, "bad")]),
                                        "good": ms([sum(r["manager_id"] == "agent_0" for r in t["rounds"]) for t in games(m, "good")])},
            "fixed_ballot": ef, "random_ballot": er,
            "agent0_wins_fisher_p": float(stats.fisher_exact([[ef.get("agent0_wins", 0), ef["elections"] - ef.get("agent0_wins", 0)],
                                                               [er.get("agent0_wins", 0), er["elections"] - er.get("agent0_wins", 0)]])[1])}
    pool_bad = [pool_bad[0] + kb, pool_bad[1] + nb]
    pool_good = [pool_good[0] + kg, pool_good[1] + ng]
res["D_accountability"] = {"per_model": d, "pooled": {
    "bad_replaced": pool_bad, "good_replaced": pool_good,
    "fisher_p": float(stats.fisher_exact([[pool_bad[0], pool_bad[1] - pool_bad[0]], [pool_good[0], pool_good[1] - pool_good[0]]])[1])}}

# ---------------------------------------------------------------- E. visibility of contributions
e_ = {}
for m in BELOW + OSS:
    if not games(m, "agg_chat"):
        continue
    e_[m] = {"chat_vs_aggregate": {"contrib": metric_test(m, "chat", "agg_chat", contrib), "coop": metric_test(m, "chat", "agg_chat", coop),
                                   "broken": {"individual": broken_counts(games(m, "chat")), "aggregate": broken_counts(games(m, "agg_chat"))}},
             "elected_vs_aggregate": {"contrib": metric_test(m, "elected", "agg_elected", contrib),
                                      "coop": metric_test(m, "elected", "agg_elected", coop),
                                      "broken": {"individual": broken_counts(games(m, "elected")),
                                                 "aggregate": broken_counts(games(m, "agg_elected"))}}}
res["E_visibility_contributions"] = e_

# ---------------------------------------------------------------- F. version vs reasoning
res["F_version"] = {m: {"chat_contrib": ms([contrib(t) for t in games(m, "chat")]),
                        "elected_contrib": ms([contrib(t) for t in games(m, "elected")]),
                        "offers": {s: ms([offers(t) for t in games(m, s)]) for s in ("elected", "salary", "costly")},
                        "election_dm_share": {s: ms([dm_share(t) for t in games(m, s)]) for s in ("elected", "salary", "costly")},
                        "broken_chat": broken_counts(games(m, "chat"))}
                    for m in ("gpt4o", "gpt41", "gpt5", "gpt5min")}

# ---------------------------------------------------------------- G. prompt robustness
g = {}
for m in ["gpt4o", "claude", "gemini"] + OSS:
    if not games(m, "nostrat_chat"):
        continue
    g[m] = {"chat": {"contrib": metric_test(m, "chat", "nostrat_chat", contrib), "coop": metric_test(m, "chat", "nostrat_chat", coop),
                     "broken": [broken_counts(games(m, "chat")), broken_counts(games(m, "nostrat_chat"))]},
            "elected": {"contrib": metric_test(m, "elected", "nostrat_elected", contrib),
                        "coop": metric_test(m, "elected", "nostrat_elected", coop),
                        "offers": metric_test(m, "elected", "nostrat_elected", offers),
                        "broken": [broken_counts(games(m, "elected")), broken_counts(games(m, "nostrat_elected"))]}}
res["G_no_strategic"] = g

# ---------------------------------------------------------------- H. quantisation
h = {}
for s in ("baseline", "chat", "elected", "salary"):
    a, q = games("qwen3.8-27b", s), games("qwen3.8-27b-nvfp4", s)
    h[s] = {"coop": test([coop(t) for t in a], [coop(t) for t in q]),
            "contrib": test([contrib(t) for t in a], [contrib(t) for t in q]),
            "offers": test([offers(t) for t in a], [offers(t) for t in q]) if s != "baseline" else None,
            "broken": {"bf16": broken_counts(a), "nvfp4": broken_counts(q)}}
res["H_quantisation"] = h

# ---------------------------------------------------------------- I. endgame
endg = {}
for m in API_MAIN + NEWER + OSS + ["gpt41", "gpt5min"]:
    row = {}
    for s in ("chat", "elected"):
        ts = games(m, s)
        if not ts:
            continue
        early = st.fmean(a["contribution"] for t in ts for r in t["rounds"][:-1] for a in r["agents"])
        last = st.fmean(a["contribution"] for t in ts for a in t["rounds"][-1]["agents"])
        bl, be = broken_counts(ts, last=True), broken_counts(ts, last=False)
        row[s] = {"contrib_r1_19": round(early, 2), "contrib_r20": round(last, 2), "broken_r20": bl, "broken_r1_19": be,
                  "share_of_broken_in_r20": round(100 * bl["broken"] / (bl["broken"] + be["broken"]), 1) if bl["broken"] + be["broken"] else None}
    endg[m] = row
res["I_endgame"] = endg

# ---------------------------------------------------------------- J. offer measure
j = {}
ELECTED_SETUPS = ("batch3_mgr_elected", "batch8_mgr_salary", "batch8_mgr_costly", "batch9_punish_hidden", "batch9_punish_anonymous")
npos = nneg = 0
for f in glob.glob(os.path.join(R, "*", "*.json")):
    name = os.path.basename(f).split("__")[0]
    if not name.startswith(ELECTED_SETUPS):
        continue
    o = json.load(open(f, encoding="utf-8"))
    if o.get("finished", "9999") >= "2026-10-01 16:30":     # the labelled sample was drawn from round-1 games
        continue
    for msg in private(o["trials"][0], "election_deal"):
        text = (msg.get("content") or "").strip()
        if text:
            if STRICT.search(text):
                npos += 1
            else:
                nneg += 1
j["population_round1"] = {"rule_matches": npos, "non_matches": nneg}
if HUMAN:
    rows = list(csv.DictReader(open(HUMAN, encoding="utf-8-sig")))
    lab = [(bool(STRICT.search(r["message"])), _classify_message(r["message"]) == "deal_offer", r["is_offer"].strip().lower() == "yes")
           for r in rows]
    pos = [x for x in lab if x[0]]
    neg = [x for x in lab if not x[0]]
    w = {True: npos / len(pos), False: nneg / len(neg)}       # stratum weights: population / sample

    def weighted(pred):
        tp = sum(w[s] for s, k, hmn in lab if pred(s, k) and hmn)
        fp = sum(w[s] for s, k, hmn in lab if pred(s, k) and not hmn)
        fn = sum(w[s] for s, k, hmn in lab if not pred(s, k) and hmn)
        return {"precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}

    j["sample"] = {"rule_matches": len(pos), "non_matches": len(neg),
                   "offers_among_non_matches": sum(x[2] for x in neg), "offers_among_matches": sum(x[2] for x in pos)}
    j["strict_rule_population_weighted"] = weighted(lambda s, k: s)
    j["keyword_population_weighted"] = weighted(lambda s, k: k)
    j["strict_or_keyword_population_weighted"] = weighted(lambda s, k: s or k)
    j["population_share_of_election_messages_that_are_offers"] = round(
        (npos * sum(x[2] for x in pos) / len(pos) + nneg * sum(x[2] for x in neg) / len(neg)) / (npos + nneg), 3)
# per model: election-message use, strict and keyword offers per 100 agent-rounds under elected / salary / costly
j["per_model"] = {m: {s: {"election_dm_share": ms([dm_share(t) for t in games(m, s)]), "offers": ms([offers(t) for t in games(m, s)]),
                          "offers_keyword": ms([offers_kw(t) for t in games(m, s)])} for s in ("elected", "salary", "costly")}
                  for m in API_MAIN + NEWER + OSS}
j["pay_effect_keyword"] = {m: metric_test(m, "elected", "salary", offers_kw) for m in API_MAIN + NEWER + OSS}
res["J_offer_measure"] = j

# ---------------------------------------------------------------- K. re-election, clustered by game
krows = []
for group, models in (("main_api", API_MAIN), ("newer", NEWER), ("self_hosted", OSS)):
    for m in models:
        for s in ("elected", "salary", "costly", "hidden", "anonymous"):
            for gi, t in enumerate(games(m, s)):
                rounds = {r["round_num"]: r for r in t["rounds"]}
                hist = t["election_history"]
                for i in range(1, len(hist)):
                    inc, e = hist[i - 1]["winner"], hist[i]
                    block = [rounds[k] for k in range(e["round_num"] - 5, e["round_num"]) if k in rounds]
                    if not block:
                        continue
                    krows.append({"group": group, "model": m, "game": f"{m}|{s}|{gi}", "reelected": int(e["winner"] == inc),
                                  "group_contrib": st.fmean(a["contribution"] for r in block for a in r["agents"]),
                                  "own_contrib": st.fmean(a["contribution"] for r in block for a in r["agents"] if a["agent_id"] == inc),
                                  "reward": st.fmean(sum(float(v) for v in ((r.get("manager_action") or {}).get("reward") or {}).values()
                                                         if isinstance(v, (int, float))) for r in block),
                                  "punish": st.fmean(sum(float(v) for v in ((r.get("manager_action") or {}).get("punish") or {}).values()
                                                         if isinstance(v, (int, float))) for r in block)})
kd = pd.DataFrame(krows)
k = {}
for group in ("main_api", "newer", "self_hosted"):
    sub = kd[kd.group == group].copy()
    for col in ("group_contrib", "own_contrib", "reward", "punish"):
        sd = sub[col].std()
        sub[col + "_z"] = (sub[col] - sub.groupby("model")[col].transform("mean")) / (sd if sd > 0 else 1)
    out = {"elections": int(len(sub)), "games": int(sub.game.nunique()), "reelected": int(sub.reelected.sum())}
    for col in ("group_contrib", "own_contrib", "reward", "punish"):
        try:
            fit = smf.glm(f"reelected ~ {col}_z + C(model)", sub, family=__import__("statsmodels.api").api.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.game)[0]})
            out[col] = {"log_odds_per_sd": round(float(fit.params[col + "_z"]), 3), "p": round(float(fit.pvalues[col + "_z"]), 4),
                        "ci": [round(float(x), 3) for x in fit.conf_int().loc[col + "_z"]]}
        except Exception as ex:     # e.g. a perfectly separated or constant predictor
            out[col] = {"error": type(ex).__name__}
    out["record_spread_sd"] = {col: round(float(sub[col].std()), 2) for col in ("group_contrib", "own_contrib", "reward", "punish")}
    k[group] = out
res["K_reelection_clustered"] = k

# ---------------------------------------------------------------- L. manipulation check (sanction visibility)
SANC = re.compile(r"\b(punish\w*|penal\w*|sanction\w*|fined?|lost \d+ tokens|bonus|rewarded)\b", re.I)
lrow = {}
for m in API_MAIN + NEWER + OSS:
    row = {}
    for s in ("elected", "hidden", "anonymous"):
        msgs = [x for t in games(m, s) for r in t["rounds"] for x in (r.get("comm_messages") or [])
                if x.get("phase") == "pre_action" and x.get("content")]
        row[s] = round(100 * sum(bool(SANC.search(x["content"])) for x in msgs) / len(msgs), 1) if msgs else None
    lrow[m] = row
res["L_sanction_mentions_pct_of_public_messages"] = lrow

# ---------------------------------------------------------------- M. valid replies
mrow = {}
for m in OSS:
    row = {}
    for s in ("baseline", "chat", "elected"):
        rs = [a for t in games(m, s) for a in recs(t)]
        ok = [a for a in rs if a.get("parse_ok", True)]
        row[s] = {"valid_share": round(100 * len(ok) / len(rs), 1) if rs else None,
                  "contrib_all": round(st.fmean(a["contribution"] for a in rs), 2) if rs else None,
                  "contrib_valid": round(st.fmean(a["contribution"] for a in ok), 2) if ok else None,
                  "coop_valid": round(100 * sum(a["contribution"] > 0 for a in ok) / len(ok), 1) if ok else None}
    mrow[m] = row
res["M_valid_replies"] = mrow

# ---------------------------------------------------------------- N. LLM judge
if JUDGE and HUMAN:
    JL = {}
    for line in open(JUDGE, encoding="utf-8"):
        o = json.loads(line)
        JL[o["key"]] = o
    yes = lambda v: (v or "").strip().lower() == "yes"
    rows = list(csv.DictReader(open(HUMAN, encoding="utf-8-sig")))
    pairs = [(r, JL["human|" + r["id"]]) for r in rows if "human|" + r["id"] in JL]

    def kappa(pairs_):
        po = sum(a == b for a, b in pairs_) / len(pairs_)
        ca, cb = Counter(a for a, _ in pairs_), Counter(b for _, b in pairs_)
        pe = sum(ca[x] * cb[x] for x in ca) / len(pairs_) ** 2
        return {"n": len(pairs_), "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3)}

    tp = sum(yes(j["is_offer"]) and yes(r["is_offer"]) for r, j in pairs)
    fp = sum(yes(j["is_offer"]) and not yes(r["is_offer"]) for r, j in pairs)
    fn = sum(not yes(j["is_offer"]) and yes(r["is_offer"]) for r, j in pairs)
    strat = {True: j_["population_round1"]["rule_matches"] / j_["sample"]["rule_matches"],
             False: j_["population_round1"]["non_matches"] / j_["sample"]["non_matches"]} if (j_ := res["J_offer_measure"]).get("sample") else None
    wt = lambda r: strat[bool(STRICT.search(r["message"]))] if strat else 1
    wtp = sum(wt(r) for r, j in pairs if yes(j["is_offer"]) and yes(r["is_offer"]))
    wfp = sum(wt(r) for r, j in pairs if yes(j["is_offer"]) and not yes(r["is_offer"]))
    wfn = sum(wt(r) for r, j in pairs if not yes(j["is_offer"]) and yes(r["is_offer"]))
    nres = {"model": next(iter(JL.values()))["model"],
            "validation": {"n": len(pairs), "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3),
                           "is_offer": kappa([(yes(r["is_offer"]), yes(j["is_offer"])) for r, j in pairs]),
                           "category": kappa([(r["category"].strip().lower(), (j["category"] or "").strip().lower()) for r, j in pairs]),
                           "population_weighted": {"precision": round(wtp / (wtp + wfp), 3), "recall": round(wtp / (wtp + wfn), 3)}}}

    def judged_share(t, name):
        n = 5 * len(t["election_history"])
        k = sum(1 for rd in t["rounds"] for x in (rd.get("comm_messages") or [])
                if x.get("phase") == "election_deal" and x.get("recipients")
                and yes((JL.get(f"{name}|{rd['round_num']}|{x['sender']}") or {}).get("is_offer")))
        return 100 * k / n if n else None

    def jgames(m, s):
        out = []
        for sub in ("api", "oss"):
            for f in sorted(glob.glob(os.path.join(R, sub, f"{S[s]}_{m}__t*.json"))):
                out.append(judged_share(json.load(open(f, encoding="utf-8"))["trials"][0], os.path.basename(f)))
        return out

    by = {}
    for m in API_MAIN + NEWER + ["gpt41", "gpt5min"] + OSS:
        cell = {s: jgames(m, s) for s in ("elected", "salary", "costly", "neutral", "neutral_salary")}
        by[m] = {"share": {s: ms(v) for s, v in cell.items() if v},
                 "pay_effect_deal": test(cell["elected"], cell["salary"]),
                 "cost_vs_salary_deal": test(cell["salary"], cell["costly"])}
        if cell["neutral"]:
            by[m].update(deal_vs_neutral=test(cell["elected"], cell["neutral"]),
                         deal_vs_neutral_salary=test(cell["salary"], cell["neutral_salary"]),
                         pay_effect_neutral=test(cell["neutral"], cell["neutral_salary"]))
    nres["offer_share_of_election_opportunities"] = by
    res["N_llm_judge"] = nres

# ---------------------------------------------------------------- O. broken intentions given the speaker's previous contribution
def bin_prev(x):
    return "none" if x is None else "0" if x == 0 else "1-9" if x < 10 else "10-14" if x < 15 else "15-20"


o = {}
for grp, models in (("main_api", API_MAIN), ("newer", NEWER), ("self_hosted", OSS)):
    for s in ("chat", "elected"):
        cnt = Counter()
        for m in models:
            for t in games(m, s):
                for _, _, v, given, prv, _ in statements(t):
                    k = bin_prev(prv)
                    cnt[(k, "n")] += 1
                    cnt[(k, "b")] += given < v - 5
        o[f"{grp}_{s}"] = {k: {"stated": cnt[(k, "n")], "broken": cnt[(k, "b")],
                                "share": round(100 * cnt[(k, "b")] / cnt[(k, "n")], 1) if cnt[(k, "n")] else None}
                            for k in ("none", "0", "1-9", "10-14", "15-20")}
res["O_broken_by_previous_contribution"] = o
res["O_commitment_only_chat"] = {m: broken_counts(games(m, "chat"), first_person=True) for m in API_MAIN + NEWER + OSS}

# ---------------------------------------------------------------- P. incumbents: stratified test, tie-rule survivals
from statsmodels.stats.contingency_tables import StratifiedTable  # noqa: E402

tabs, tie = [], {}
for m, v in res["D_accountability"]["per_model"].items():
    kb, nb = v["bad_replaced_at_6"]
    kg, ng = v["good_replaced_at_6"]
    tabs.append(np.array([[kb, nb - kb], [kg, ng - kg]], float) + 0.5)    # 0.5 added to every cell (empty cells)
    row = {}
    for pol in ("bad", "good"):
        surv = by_tie = 0
        for t in games(m, pol):
            e6 = [e for e in t["election_history"] if e["round_num"] == 6]
            if not e6 or e6[0]["winner"] != "agent_0":
                continue
            surv += 1
            v6 = Counter((e6[0].get("votes") or {}).values()).most_common()
            by_tie += len(v6) > 1 and v6[0][1] == v6[1][1]
        row[pol] = {"survived": surv, "survived_on_a_tie": by_tie}
    tie[m] = row
st_ = StratifiedTable(tabs)
res["P_incumbents"] = {"cmh": {"pooled_odds_ratio": round(float(st_.oddsratio_pooled), 2),
                               "p": round(float(st_.test_null_odds(correction=True).pvalue), 4)},
                       "survivals_decided_by_tie_rule": tie}

# ---------------------------------------------------------------- Q. mixed groups: who sat where on the ballot
q = Counter()
for f in sorted(glob.glob(os.path.join(R, "api", "batch4_hetero_all_drop_*__t*.json"))):
    t = json.load(open(f, encoding="utf-8"))["trials"][0]
    amap = t["agent_model_map"]
    for e in t["election_history"]:
        w = e["winner"]
        q[("wins_by_position", w)] += 1
        q[("wins_by_family", amap[w])] += 1
    for aid, mod in amap.items():
        q[("seats", mod, aid)] += 1
res["Q_mixed_positions"] = {"wins_by_position": {a: q[("wins_by_position", a)] for a in sorted({k[1] for k in q if k[0] == "wins_by_position"})},
                            "wins_by_family": {k[1]: n for k, n in q.items() if k[0] == "wins_by_family"},
                            "seat_counts": {f"{k[1]}@{k[2]}": n for k, n in q.items() if k[0] == "seats"}}

# ---------------------------------------------------------------- R. cross-rule: the manager's own contribution
rr = {}
for f in sorted(glob.glob(os.path.join(R, "api", "batch11_mgr_*__t*.json"))):
    pair = os.path.basename(f).split("__")[0].replace("batch11_mgr_", "")
    t = json.load(open(f, encoding="utf-8"))["trials"][0]
    mg = [a["contribution"] for r in t["rounds"] for a in r["agents"] if a["is_manager"]]
    wk = [a["contribution"] for r in t["rounds"] for a in r["agents"] if not a["is_manager"]]
    rr.setdefault(pair, {"manager": [], "workers": []})
    rr[pair]["manager"].append(st.fmean(mg))
    rr[pair]["workers"].append(st.fmean(wk))
res["R_crossrule_manager_contribution"] = {k: {"manager": ms(v["manager"]), "workers": ms(v["workers"])} for k, v in rr.items()}

# ---------------------------------------------------------------- S. were the model IDs constant over the runs?
ids = {}
for f in glob.glob(os.path.join(R, "*", "*.json")):
    t = json.load(open(f, encoding="utf-8"))["trials"][0]
    for a in recs(t):
        ids.setdefault(a.get("model_name"), set()).add(os.path.basename(f).split("__")[0].rsplit("_", 1)[-1])
res["S_model_ids"] = {str(k): sorted(v)[:6] for k, v in ids.items()}

# ---------------------------------------------------------------- T. Claude under an elected manager
tt = []
for t in games("claude", "elected"):
    hist = t["election_history"]
    turn = sum(hist[i]["winner"] != hist[i - 1]["winner"] for i in range(1, len(hist)))
    tt.append((contrib(t), turn))
res["T_claude_elected"] = {"games": [{"contrib": round(c, 2), "turnovers": n} for c, n in tt],
                           "spearman": [round(float(x), 3) for x in stats.spearmanr([c for c, _ in tt], [n for _, n in tt])]}

# ---------------------------------------------------------------- U. campaign pledges: kept in the following rounds?
u = {}
for m in ["gemini", "claude", "gpt4o", "qwen", "gpt-oss-120b", "qwen3.8-27b"]:
    k_all = k_stated = kept = later = 0
    for t in games(m, "elected"):
        rounds = {r["round_num"]: {a["agent_id"]: a["contribution"] for a in r["agents"]} for r in t["rounds"]}
        for r in t["rounds"]:
            for msg in r.get("comm_messages") or []:
                if msg.get("phase") != "election_speech" or not msg.get("content"):
                    continue
                k_all += 1
                cl = _classify_with_regex(msg["content"])
                if cl["promise_type"] != "explicit":
                    continue
                k_stated += 1
                for rnd in range(r["round_num"], r["round_num"] + 5):
                    if rnd in rounds:
                        later += 1
                        kept += rounds[rnd].get(msg["sender"], 0) >= cl["extracted_value"] - 5
    u[m] = {"speeches": k_all, "with_amount": k_stated, "agent_rounds_after": later,
            "kept_share": round(100 * kept / later, 1) if later else None}
res["U_campaign_pledges"] = u

# ---------------------------------------------------------------- V. third-review controls (batch 15)
v = {"elected_nobudget": {}, "incumbents_random_ballot": {}}
for m in ["gemini", "gpt-oss-120b", "qwen3.8-27b"]:
    if games(m, "elected_nobudget"):
        v["elected_nobudget"][m] = {s_: ms([contrib(t) for t in games(m, s_)]) for s_ in ("chat", "fixed", "elected_nobudget", "elected")}
        v["elected_nobudget"][m]["vs_chat"] = metric_test(m, "chat", "elected_nobudget", contrib)
        v["elected_nobudget"][m]["vs_elected"] = metric_test(m, "elected_nobudget", "elected", contrib)
tabs_rb, pb, pg = [], [0, 0], [0, 0]
for m in ["gpt4o", "gemini", "qwen", "deepseek"] + OSS:
    if not games(m, "bad_rb"):
        continue
    kb, nb = replaced_at_6(games(m, "bad_rb"))
    kg, ng = replaced_at_6(games(m, "good_rb"))
    v["incumbents_random_ballot"][m] = {"bad": [kb, nb], "good": [kg, ng]}
    tabs_rb.append(np.array([[kb, nb - kb], [kg, ng - kg]], float) + 0.5)
    pb = [pb[0] + kb, pb[1] + nb]
    pg = [pg[0] + kg, pg[1] + ng]
if tabs_rb:
    st_rb = StratifiedTable(tabs_rb)
    v["incumbents_random_ballot_pooled"] = {"bad": pb, "good": pg, "cmh_odds_ratio": round(float(st_rb.oddsratio_pooled), 2),
                                            "cmh_p": round(float(st_rb.test_null_odds(correction=True).pvalue), 4)}
res["V_batch15"] = v

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
