"""First-election offers under the neutral prompt (Claude), and vote-level accountability of the bad incumbent.

  python analysis/round7.py <results> <results_confirm> <offers_llm_judge.jsonl> <confirm_llm_judge.jsonl> <out.json>
"""
import glob, itertools, json, os, statistics as st, sys
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

R, RC, J1, J2, OUT = sys.argv[1:6]
judge = lambda p: {d["key"]: d for d in map(json.loads, open(p, encoding="utf-8"))}
yes = lambda d: str((d or {}).get("is_offer", "")).lower().startswith("y")


def perm_greater(a, b, reps=100000, seed=20261004):
    rng, pool, na, obs = np.random.default_rng(seed), np.array(a + b, float), len(a), st.fmean(b) - st.fmean(a)
    h = 0
    for _ in range(reps):
        x = rng.permutation(pool)
        h += x[na:].mean() - x[:na].mean() >= obs - 1e-12
    return (h + 1) / (reps + 1)


N_MSG = []


def first_share(root, jl, setup):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*", f"{setup}_claude__t*.json"))):
        name, t = os.path.basename(f), json.load(open(f, encoding="utf-8"))["trials"][0]
        r1 = [m for rd in t["rounds"] if rd["round_num"] == 1 for m in (rd.get("comm_messages") or [])
              if m.get("phase") == "election_deal" and m.get("recipients")]
        N_MSG.append(len(r1))
        out.append(100 * sum(yes(jl.get(f"{name}|1|{m['sender']}")) for m in r1) / 5)
    return out


res = {"first_election_claude": {}}
for tag, root, jl in (("exploratory", R, judge(J1)), ("fresh", RC, judge(J2))):
    a, b = first_share(root, jl, "batch14_neutral_elected"), first_share(root, jl, "batch14_neutral_salary")
    res["first_election_claude"][tag] = {"none": a, "salary": b}
a = sum((v["none"] for v in res["first_election_claude"].values()), [])
b = sum((v["salary"] for v in res["first_election_claude"].values()), [])
res["first_election_claude"]["pooled"] = {"none_mean": round(st.fmean(a), 1), "salary_mean": round(st.fmean(b), 1),
                                          "n": [len(a), len(b)], "p_one_sided": round(perm_greater(a, b), 4)}

# vote level: at the round-6 election, do voters the bad incumbent punished vote against it?
for pol, key in (("bad", "punish"), ("good", "reward")):
    tabs, cnt = {"all": [], "no_self": []}, {"all": [0, 0, 0, 0], "no_self": [0, 0, 0, 0]}
    for f in glob.glob(os.path.join(R, "*", f"batch1[345]_mgr_{pol}incumbent*__t*.json")):
        t = json.load(open(f, encoding="utf-8"))["trials"][0]
        hit = {a for rd in t["rounds"] if rd["round_num"] == 5 for a, v in ((rd.get("manager_action") or {}).get(key) or {}).items() if v}
        e6 = [e for e in t["election_history"] if e["round_num"] == 6]
        if not e6:
            continue
        for v in ("all", "no_self"):
            c = np.zeros((2, 2))
            for voter, cand in (e6[0].get("votes") or {}).items():
                if voter == "agent_0" or (v == "no_self" and cand == voter):
                    continue
                c[0 if voter in hit else 1, 0 if cand != "agent_0" else 1] += 1
            cnt[v] = [cnt[v][0] + c[0, 0], cnt[v][1] + c[0].sum(), cnt[v][2] + c[1, 0], cnt[v][3] + c[1].sum()]
            if c[0].sum() and c[1].sum():
                tabs[v].append(c)
    res[f"votes_{pol}_incumbent"] = {}
    for v in ("all", "no_self"):
        stt = StratifiedTable(tabs[v])
        k = cnt[v]
        res[f"votes_{pol}_incumbent"][v] = {f"{key}ed_against": f"{int(k[0])}/{int(k[1])}", "others_against": f"{int(k[2])}/{int(k[3])}",
                                            "games": len(tabs[v]), "mh_odds_ratio": round(float(stt.oddsratio_pooled), 2),
                                            "or_ci95": [round(float(x), 2) for x in stt.oddsratio_pooled_confint()],
                                            "p": round(float(stt.test_null_odds(correction=True).pvalue), 4)}
res["first_election_claude"]["pooled"]["round1_messages"] = sum(N_MSG)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "first_election_claude"} | {"first": res["first_election_claude"]["pooled"]}, indent=1))
