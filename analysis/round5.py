"""Round 5: equivalence test for incumbents (fair ballot), batch 16 controls, and the confirmatory batch.

  python analysis/round5.py <results_dir> <results_confirm_dir> <confirm_judge.jsonl> <out.md>

The confirmatory tests follow review_plans/CONFIRMATORY_PLAN.md exactly (one-sided permutation, Holm over 6).
"""
import glob, itertools, json, math, os, random, re, statistics as st, sys
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.deception import _classify_with_regex  # noqa: E402

R, RC, JUDGE, OUT = sys.argv[1:5]
API = ["gpt4o", "gemini", "qwen", "deepseek"]
OSS = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]
FIRST_PERSON = re.compile(r"\bI(?:'ll| will| am going to|'m going to| plan to| intend to| commit| am contributing|'m contributing| shall)\b", re.I)
CONDITIONAL = re.compile(r"\b(if|as long as|provided|unless|only when|in return)\b", re.I)


def load(root, setup, m):
    fs = sorted(glob.glob(os.path.join(root, "*", f"{setup}_{m}__t*.json")))
    return [(os.path.basename(f), json.load(open(f, encoding="utf-8"))["trials"][0]) for f in fs]


def contrib(t, workers_only=False):
    xs = [a["contribution"] for r in t["rounds"] for a in r["agents"] if not (workers_only and a.get("is_manager"))]
    return st.fmean(xs)


def replaced_at_6(gs):
    k = n = 0
    for _, t in gs:
        e6 = [e for e in t["election_history"] if e["round_num"] == 6]
        if e6:
            n += 1
            k += e6[0]["winner"] != "agent_0"
    return k, n


def perm(a, b, alternative="two", reps=100000, seed=20261003):
    """permutation p for mean(b) - mean(a); alternative 'greater' tests b > a"""
    a, b = list(a), list(b)
    pool, na = a + b, len(a)
    obs = st.fmean(b) - st.fmean(a)
    hit = (lambda d: d >= obs - 1e-12) if alternative == "greater" else (lambda d: abs(d) >= abs(obs) - 1e-12)
    if math.comb(len(pool), na) <= 200000:
        h = c = 0
        for idx in itertools.combinations(range(len(pool)), na):
            s = set(idx)
            h += hit(st.fmean([pool[i] for i in range(len(pool)) if i not in s]) - st.fmean([pool[i] for i in idx]))
            c += 1
        return h / c
    rng, h = random.Random(seed), 0
    for _ in range(reps):
        rng.shuffle(pool)
        h += hit(st.fmean(pool[na:]) - st.fmean(pool[:na]))
    return (h + 1) / (reps + 1)


def boot_ci(a, b, reps=10000, seed=20261003):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(reps)]
    return np.percentile(d, [2.5, 97.5]).round(2).tolist()


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def ms(xs):
    return f"{st.fmean(xs):.2f} ± {st.stdev(xs):.2f} (n={len(xs)})" if len(xs) > 1 else (f"{xs[0]:.2f} (n=1)" if xs else "–")


md = ["# Round 5 results (raw, for the write-up)", ""]

# ---------------------------------------------------------------- 1. incumbents, fair ballot: equivalence
md += ["## 1. Bad vs good incumbent with a fair ballot (removed at the first election, round 6)", "",
       "| Model | Bad removed | Good removed |", "|---|---|---|"]
tabs, num, den, var, tot = [], 0.0, 0.0, 0.0, [0, 0, 0, 0]
for m in API + OSS:
    kb, nb = replaced_at_6(load(R, "batch15_mgr_badincumbent_rb", m))
    kg, ng = replaced_at_6(load(R, "batch15_mgr_goodincumbent_rb", m))
    if not nb or not ng:
        continue
    md.append(f"| {m} | {kb}/{nb} | {kg}/{ng} |")
    tot = [tot[0] + kb, tot[1] + nb, tot[2] + kg, tot[3] + ng]
    tabs.append(np.array([[kb, nb - kb], [kg, ng - kg]], float) + 0.5)
    w, pb, pg = nb * ng / (nb + ng), kb / nb, kg / ng
    num, den, var = num + w * (pb - pg), den + w, var + w * w * (pb * (1 - pb) / nb + pg * (1 - pg) / ng)
rd, se = num / den, var ** 0.5 / den
s_ = StratifiedTable(tabs)
lo, hi = s_.oddsratio_pooled_confint()
md += [f"| **Pooled** | {tot[0]}/{tot[1]} | {tot[2]}/{tot[3]} |", "",
       f"- Stratified risk difference (bad − good): **{100 * rd:.1f} points**, 95% CI {100 * (rd - 1.96 * se):.1f} to {100 * (rd + 1.96 * se):.1f}; "
       f"90% CI {100 * (rd - 1.645 * se):.1f} to {100 * (rd + 1.645 * se):.1f}.",
       f"- CMH odds ratio {s_.oddsratio_pooled:.2f} (95% CI {lo:.2f}–{hi:.2f}), p = {s_.test_null_odds(correction=True).pvalue:.3f}."]
for margin in (10, 15, 20):
    p_lo = 1 - st.NormalDist().cdf((rd + margin / 100) / se)
    p_hi = st.NormalDist().cdf((rd - margin / 100) / se)
    md.append(f"- Equivalence (TOST) at ±{margin} points: p = {max(p_lo, p_hi):.4f} → {'equivalent' if max(p_lo, p_hi) < 0.05 else 'not shown'}.")
md.append("")

# ---------------------------------------------------------------- 2. batch 16
md += ["## 2. Batch 16 controls", "", "### Reward-only fixed manager (mean contribution per agent-round)", "",
       "| Model | Chat only | Fixed, no budget | Fixed, reward only | Fixed, full | p reward-only vs full | p reward-only vs chat |",
       "|---|---|---|---|---|---|---|"]
for m in ["gemini", "gpt-oss-120b", "qwen3.8-27b"]:
    g = {k: [contrib(t) for _, t in load(R, s, m)] for k, s in
         (("chat", "batch2_comm_full"), ("nob", "batch13_mgr_nosanction"), ("ro", "batch16_mgr_fixed_rewardonly"), ("full", "batch3_mgr_fixed"))}
    if len(g["ro"]) < 2:
        continue
    md.append(f"| {m} | {ms(g['chat'])} | {ms(g['nob'])} | {ms(g['ro'])} | {ms(g['full'])} | "
              f"{perm(g['ro'], g['full']):.3f} | {perm(g['chat'], g['ro']):.3f} |")
md += ["", "### Manager contribution fixed by the experimenter (workers' mean contribution, manager excluded)", "",
       "| Model | Manager gives 10 | Manager gives 20 | Diff (20 − 10) | 95% CI | p (two-sided) | Own fixed manager (workers) |",
       "|---|---|---|---|---|---|---|"]
for m in ["qwen", "gemini", "gpt-oss-120b", "qwen3.8-27b"]:
    a = [contrib(t, True) for _, t in load(R, "batch16_mgr_fixed_contrib10", m)]
    b = [contrib(t, True) for _, t in load(R, "batch16_mgr_fixed_contrib20", m)]
    own = [contrib(t, True) for _, t in load(R, "batch3_mgr_fixed", m)]
    if len(a) < 2 or len(b) < 2:
        continue
    md.append(f"| {m} | {ms(a)} | {ms(b)} | {st.fmean(b) - st.fmean(a):+.2f} | {boot_ci(a, b)} | {perm(a, b):.4f} | {ms(own) if own else '–'} |")
md.append("")

# ---------------------------------------------------------------- 3. confirmatory batch
JL = {}
if os.path.exists(JUDGE):
    for line in open(JUDGE, encoding="utf-8"):
        d = json.loads(line)
        JL[d["key"]] = d


def judged_share(name, t):
    n = 5 * len(t["election_history"])
    k = sum(1 for rd in t["rounds"] for x in (rd.get("comm_messages") or [])
            if x.get("phase") == "election_deal" and x.get("recipients")
            and str((JL.get(f"{name}|{rd['round_num']}|{x['sender']}") or {}).get("is_offer", "")).lower().startswith("y"))
    return 100 * k / n if n else None


def commit_broken(t):
    b = n = 0
    for rd in t["rounds"]:
        con = {a["agent_id"]: a["contribution"] for a in rd["agents"]}
        pub = {}
        for msg in rd.get("comm_messages") or []:
            if msg.get("phase") == "pre_action" and msg.get("recipients") is None and msg.get("content"):
                pub[msg["sender"]] = msg["content"]
        for aid, text in pub.items():
            cl = _classify_with_regex(text)
            if cl["promise_type"] == "explicit" and FIRST_PERSON.search(text) and not CONDITIONAL.search(text):
                n += 1
                b += con.get(aid, 0.0) < cl["extracted_value"] - 5
    return 100 * b / n if n else None


C = lambda s, m: load(RC, s, m)
plan = [
    ("H1 Gemini: elected > chat only (contribution)", [contrib(t) for _, t in C("batch2_comm_full", "gemini")], [contrib(t) for _, t in C("batch3_mgr_elected", "gemini")]),
    ("H2 Qwen3.8: elected > chat only (contribution)", [contrib(t) for _, t in C("batch2_comm_full", "qwen3.8-27b")], [contrib(t) for _, t in C("batch3_mgr_elected", "qwen3.8-27b")]),
    ("H3 Claude, neutral prompt: salary > none (judged offers, % of opportunities)", [judged_share(n, t) for n, t in C("batch14_neutral_elected", "claude")], [judged_share(n, t) for n, t in C("batch14_neutral_salary", "claude")]),
    ("H4a Gemini: deal prompt > neutral (judged offers)", [judged_share(n, t) for n, t in C("batch14_neutral_elected", "gemini")], [judged_share(n, t) for n, t in C("batch3_mgr_elected", "gemini")]),
    ("H4b DeepSeek V3.1: deal prompt > neutral (judged offers)", [judged_share(n, t) for n, t in C("batch14_neutral_elected", "deepseek31")], [judged_share(n, t) for n, t in C("batch3_mgr_elected", "deepseek31")]),
    ("H5 broken first-person commitments, chat only: Qwen3.8 > Gemini (% per game)", [commit_broken(t) for _, t in C("batch2_comm_full", "gemini")], [commit_broken(t) for _, t in C("batch2_comm_full", "qwen3.8-27b")]),
]
rows, ps = [], []
for name, a, b in plan:
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        rows.append((name, a, b, None))
        continue
    p = perm(a, b, "greater")
    ps.append(p)
    rows.append((name, a, b, p))
adj = iter(holm(ps)) if ps else iter([])
md += ["## 3. Confirmatory batch (fresh games, plan fixed in review_plans/CONFIRMATORY_PLAN.md)", "",
       f"Judge labels: {len(JL)} messages.", "",
       "| Hypothesis | Control | Treatment | Diff | 95% bootstrap CI | one-sided p | Holm p | Confirmed |", "|---|---|---|---|---|---|---|---|"]
for name, a, b, p in rows:
    if p is None:
        md.append(f"| {name} | {ms(a)} | {ms(b)} | – | – | – | – | not enough games |")
        continue
    h = next(adj)
    md.append(f"| {name} | {ms(a)} | {ms(b)} | {st.fmean(b) - st.fmean(a):+.2f} | {boot_ci(a, b)} | {p:.4f} | {h:.4f} | {'yes' if h < 0.05 else 'no'} |")
open(OUT, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
