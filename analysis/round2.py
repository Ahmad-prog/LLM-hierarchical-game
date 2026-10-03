"""Round-2 mechanism controls (batch 13), compared with their reference setups, per model.

  python analysis/round2.py <results_dir> <out_json>

  aggregate-only visibility   vs elected (contributions visible)         cooperation, contribution, broken intentions
  manager without sanctions   vs fixed manager and chat only             cooperation, contribution
  scripted reward rule        vs fixed manager                           cooperation, contribution, welfare split
  bad incumbent (agent_0)     replaced at its first election (round 6)?  vs incumbent replacement at round 6 when elected
  randomised ballot           vs elected                                 turnover, ties at the top, agent_0 wins
Games are the unit; Welch tests where both sides have spread. Works for any model with batch-13 files.
"""
import glob
import json
import os
import statistics as st
import sys
from collections import Counter

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.deception import _classify_with_regex  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]


def load(prefix):
    out = []
    for sub in ("oss", "api"):
        out += [json.load(open(f, encoding="utf-8"))["trials"][0]
                for f in sorted(glob.glob(os.path.join(R, sub, prefix + "__t*.json")))]
    return out


def coop(t):
    recs = [a for r in t["rounds"] for a in r["agents"]]
    return 100 * sum(a["contribution"] > 0 for a in recs) / len(recs)


def contrib(t):
    return st.fmean(a["contribution"] for r in t["rounds"] for a in r["agents"])


def welfare_parts(t):
    recs = [a for r in t["rounds"] for a in r["agents"]]
    total = sum(a["net_payoff"] for a in recs) - 5 * 20 * len(t["rounds"])
    c = 0.6 * sum(a["contribution"] for a in recs)
    return total, c


def broken_share(ts):
    b = n = 0
    for t in ts:
        for rd in t["rounds"]:
            con = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
            for m in rd.get("comm_messages") or []:
                if m.get("phase") == "pre_action" and m.get("recipients") is None and m.get("content"):
                    cl = _classify_with_regex(m["content"])
                    if cl["promise_type"] == "explicit":
                        n += 1
                        b += con.get(m["sender"], 0) < cl["extracted_value"] - 5
    return {"broken": b, "stated": n}


def ms(xs):
    return {"mean": round(st.fmean(xs), 2), "sd": round(st.stdev(xs), 2) if len(xs) > 1 else 0.0, "n": len(xs)} if xs else None


def welch(a, b):
    if len(a) < 2 or len(b) < 2 or (st.variance(a) + st.variance(b)) == 0:
        return None
    r = stats.ttest_ind(b, a, equal_var=False)
    return {"diff": round(st.fmean(b) - st.fmean(a), 2), "p": round(float(r.pvalue), 4)}


def elections(ts):
    c = Counter()
    for t in ts:
        prev = None
        for e in t["election_history"]:
            votes = Counter((e.get("votes") or {}).values()).most_common()
            c["elections"] += 1
            c["agent0_wins"] += e["winner"] == "agent_0"
            c["tie_top"] += len(votes) > 1 and votes[0][1] == votes[1][1]
            if prev is not None:
                c["contested"] += 1
                c["turnovers"] += e["winner"] != prev
            prev = e["winner"]
    return dict(c)


def replaced_at_6(ts, incumbent_at_5):
    """share of games where the manager in round 5 is replaced by the round-6 election"""
    k = n = 0
    for t in ts:
        rounds = {r["round_num"]: r for r in t["rounds"]}
        e6 = [e for e in t["election_history"] if e["round_num"] == 6]
        if not e6 or 5 not in rounds:
            continue
        inc = rounds[5]["manager_id"] if incumbent_at_5 else "agent_0"
        n += 1
        k += e6[0]["winner"] != inc
    return {"replaced": k, "games": n}


def tenure_agent0(ts):
    return ms([sum(r["manager_id"] == "agent_0" for r in t["rounds"]) for t in ts])


tags = sorted({os.path.basename(f).split("__")[0].replace("batch13_info_aggregate_", "")
               for f in glob.glob(os.path.join(R, "*", "batch13_info_aggregate_*__t*.json"))})
res = {}
for m in tags:
    E, F, C = load(f"batch3_mgr_elected_{m}"), load(f"batch3_mgr_fixed_{m}"), load(f"batch2_comm_full_{m}")
    A, NS, AR = load(f"batch13_info_aggregate_{m}"), load(f"batch13_mgr_nosanction_{m}"), load(f"batch13_mgr_autoreward_{m}")
    BI, BR = load(f"batch13_mgr_badincumbent_{m}"), load(f"batch13_ballot_random_{m}")
    r = {}
    if A:
        r["aggregate_vs_elected"] = {"elected": {"coop": ms([coop(t) for t in E]), "contrib": ms([contrib(t) for t in E]),
                                                 "intentions": broken_share(E)},
                                     "aggregate": {"coop": ms([coop(t) for t in A]), "contrib": ms([contrib(t) for t in A]),
                                                   "intentions": broken_share(A)},
                                     "test_contrib": welch([contrib(t) for t in E], [contrib(t) for t in A]),
                                     "test_coop": welch([coop(t) for t in E], [coop(t) for t in A])}
    if NS:
        r["nosanction"] = {k: {"coop": ms([coop(t) for t in v]), "contrib": ms([contrib(t) for t in v])}
                           for k, v in (("chat_only", C), ("fixed", F), ("nosanction", NS))}
        r["nosanction"]["test_vs_fixed_coop"] = welch([coop(t) for t in F], [coop(t) for t in NS])
        r["nosanction"]["test_vs_chat_coop"] = welch([coop(t) for t in C], [coop(t) for t in NS])
        r["nosanction"]["test_vs_fixed_contrib"] = welch([contrib(t) for t in F], [contrib(t) for t in NS])
    if AR:
        r["autoreward_vs_fixed"] = {k: {"coop": ms([coop(t) for t in v]), "contrib": ms([contrib(t) for t in v]),
                                        "welfare": ms([welfare_parts(t)[0] for t in v]),
                                        "welfare_from_contrib": ms([welfare_parts(t)[1] for t in v])}
                                    for k, v in (("fixed", F), ("autoreward", AR))}
        r["autoreward_vs_fixed"]["test_contrib"] = welch([contrib(t) for t in F], [contrib(t) for t in AR])
    if BI:
        r["bad_incumbent"] = {"bad_replaced_at_6": replaced_at_6(BI, False),
                              "elected_incumbent_replaced_at_6": replaced_at_6(E, True),
                              "agent0_rounds_in_office": tenure_agent0(BI), "elections": elections(BI)}
    if BR:
        r["ballot_random_vs_elected"] = {"elected": elections(E), "ballot_random": elections(BR)}
    res[m] = r

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
