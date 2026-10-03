"""LaTeX tables for the paper, generated from paper_numbers.json, extras.json and revision2.json.

  python analysis/appendix_tables.py paper_data/paper_numbers.json paper_data/extras.json \
      paper_data/revision2.json > appendix_tables.tex

The output only defines macros; the paper places each table where it belongs:
  \\hgPromiseTable (Results), \\hgParseTable (Appendix A), \\hgResultTables (Appendix C),
  \\hgTestsTable (Appendix D), \\hgRobustTable and \\hgOffersTable (Appendix E).
Tables are two-column wide (table*) unless marked narrow, for the AAAI two-column layout.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
x = json.load(open(sys.argv[2]))
y = json.load(open(sys.argv[3]))

NAME = {"gpt4o": "GPT-4o", "claude": "Claude", "gemini": "Gemini", "deepseek": "DeepSeek V3", "grok": "Grok",
        "qwen": "Qwen Plus", "gpt5": "GPT-5", "deepseek31": "DeepSeek V3.1", "gpt-oss-120b": "gpt-oss",
        "nemotron-3-super-120b": "Nemotron", "ling-3.0-flash": "Ling", "qwen3.8-27b": "Qwen3.8",
        "gemma-4-31b": "Gemma"}
FRONT = ["gpt4o", "claude", "gemini", "deepseek", "grok", "qwen"]
SAME = ["gpt5", "deepseek31"]
OPEN = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]
GROUPS = (("frontier", FRONT), ("samegen", SAME), ("open", OPEN))
SETNAME = {"baseline": "No chat", "chat_only": "Chat only", "public_chat": "Public only", "private_chat": "Private only",
           "fixed": "Fixed", "elected": "Elected", "rotating": "Rotating", "salary": "Salary", "costly": "Costly",
           "hidden": "Hidden", "anonymous": "Anonymous", "belief_unknown": "Belief: unknown",
           "belief_human": "Belief: all human", "belief_mixed": "Belief: mixed"}


def pm(v, nd=1):
    return "---" if not v else f"{v['mean']:.{nd}f}$\\pm${v['sd']:.{nd}f}"


def num(v, nd=1):
    return "---" if not v else f"{v['mean']:.{nd}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(caption, label, cols, header, rows, size="\\small", sep="3pt", wide=True, place="t"):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[{place}]", "\\centering", size, f"\\setlength{{\\tabcolsep}}{{{sep}}}",
           f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r if r.startswith("\\midrule") else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


def cell(v):
    if not v:
        return "---"
    return f"{v['contrib']['mean']:.1f}$\\pm${v['contrib']['sd']:.1f} ({v['coop']['mean']:.0f})$_{{{v['contrib']['n']}}}$"


def pct(v):
    return int(v + 0.5)  # round half up, as in the text


def frac(b, n):
    return "---" if not n else f"{b}/{n} ({pct(100 * b / n)}\\%)"


def short_frac(b, n):
    """compact cell for one-column tables: 67/240 (28)"""
    return "---" if not n else f"{b}/{n} ({pct(100 * b / n)})"


def pct_n(b, n):
    """compact cell: 30 (651) = 30% of 651 stated intentions"""
    return "---" if not n else f"{pct(100 * b / n)} ({n})"


def group_row(label, ncols):
    return f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\emph{{{label}}}}}"


T = []

# ---------------------------------------------------------------- main-text promise table (one rule for all models)
I = y["intentions"]
rows = []
conds = ("chat_only", "elected", "hidden", "anonymous")
tot = {s: [sum(I["frontier"][s][m].get("broken_5", 0) for m in FRONT), sum(I["frontier"][s][m].get("stated", 0) for m in FRONT)]
       for s in conds}
rows.append(group_row("Main API models", 5))
rows.append("All six & " + " & ".join(f"{tot[s][0]}/{tot[s][1]}" for s in conds))
rows.append(group_row("Newer API models", 5))
for m in SAME:
    rows.append(f"{NAME[m]} & " + " & ".join(
        f"{I['samegen'][s][m].get('broken_5', 0)}/{I['samegen'][s][m].get('stated', 0)}" for s in conds))
rows.append(group_row("Self-hosted models", 5))
for m in OPEN:
    rows.append(f"{NAME[m]} & " + " & ".join(
        pct_n(I["open"][s][m].get("broken_5", 0), I["open"][s][m].get("stated", 0)) for s in conds))
T.append(("hgPromiseTable", table(
    "Broken stated intentions, summed over games: broken / stated for the API models, and for the self-hosted models the "
    "percentage broken with the number stated in parentheses. A stated "
    "intention is a public message that names a contribution; it is broken when the speaker then gives more than 5 tokens "
    "less. One regular expression scores every model. No main API model breaks more than 2 in any condition, and GPT-5 at "
    "most 3 with a manager; per model, first-person commitments and other thresholds are in Table~\\ref{tab:robust}.",
    "tab:promises", "lcccc",
    "& \\textbf{Chat} & \\multicolumn{3}{c}{\\textbf{Manager; sanctions are}} \\\\ \\cmidrule(lr){3-5}\n"
    "\\textbf{Model} & \\textbf{only} & \\textbf{visible} & \\textbf{hidden} & \\textbf{anon.}",
    rows, sep="2.5pt", wide=False, place="!tb")))   # main text: may also go at the foot of a column

# ---------------------------------------------------------------- appendix C: results
R = []
setups = ["baseline", "chat_only", "fixed", "elected", "rotating"]
R.append(table(
    "Main API models by institution: mean contribution $\\pm$ s.d. over games (cooperation rate, \\%). "
    "Subscript: number of games. Manager setups have full communication.",
    "tab:frontier_full", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups),
    [NAME[m] + " & " + " & ".join(cell(d["frontier"][s][m]) for s in setups) for m in FRONT]))

setups = ["elected", "salary", "costly", "hidden", "anonymous"]
R.append(table(
    "Main API models under manager pay and sanction visibility (elected manager, full communication): mean contribution "
    "$\\pm$ s.d. (cooperation rate, \\%). Elected is the no-salary, transparent control. Subscript: number of games.",
    "tab:frontier_pay", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups),
    [NAME[m] + " & " + " & ".join(cell(d["frontier"][s][m]) for s in setups) for m in FRONT]))

setups = ["baseline", "fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"]
R.append(table(
    "Newer API models next to their predecessors: mean contribution $\\pm$ s.d. (cooperation rate, \\%); subscript: number of games.",
    "tab:samegen_full", "l" + "c" * 4,
    "\\textbf{Setup} & \\textbf{GPT-5} & \\textbf{DeepSeek V3.1} & \\textbf{GPT-4o} & \\textbf{DeepSeek V3}",
    [SETNAME[s] + " & " + " & ".join(cell(d[g][s][m]) for g, m in
                                      (("samegen", "gpt5"), ("samegen", "deepseek31"), ("frontier", "gpt4o"), ("frontier", "deepseek")))
     for s in setups]))

setups = ["baseline", "public_chat", "private_chat", "chat_only", "fixed", "elected", "rotating", "salary", "costly",
          "hidden", "anonymous", "belief_unknown", "belief_human", "belief_mixed"]


def ocell(v):
    return "---" if not v else f"{v['coop']['mean']:.0f}$\\pm${v['coop']['sd']:.0f} / {v['contrib']['mean']:.1f}$_{{{v['coop']['n']}}}$"


R.append(table(
    "Self-hosted open-weight models: cooperation rate $\\pm$ s.d. (\\%) / mean contribution; subscript: number of games. \\emph{Chat only} "
    "is full communication without a manager; all setups below it have full communication and (except where stated) an "
    "elected manager; the belief setups vary what agents are told about the other players (default: all AI).",
    "tab:open_full", "l" + "c" * len(OPEN),
    "\\textbf{Setup} & " + " & ".join(f"\\textbf{{{NAME[m]}}}" for m in OPEN),
    [SETNAME[s] + " & " + " & ".join(ocell(d["open"][s][m]) for m in OPEN) for s in setups]))

rows = []
for grp, models in GROUPS:
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(pm(d[grp][s][m]["deals_strict"]) for s in ("elected", "salary", "costly"))
                    + " & " + " & ".join(num(d[grp][s][m]["deals_keyword"]) for s in ("elected", "salary", "costly")))
    rows.append("\\midrule")
rows = rows[:-1]
R.append(table(
    "Vote-contingent offers per 100 agent-rounds by manager pay (elected manager). Strict: private messages that tie a vote "
    "to a material benefit (mean $\\pm$ s.d.). Keyword: the broader deal-offer category of a keyword classifier (mean). "
    "What the offers contain is broken down in Table~\\ref{tab:offers}.",
    "tab:deals_full", "lcccccc",
    "& \\multicolumn{3}{c}{\\textbf{Strict}} & \\multicolumn{3}{c}{\\textbf{Keyword}} \\\\ \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    "\\textbf{Model} & No salary & Salary & Costly & No salary & Salary & Costly", rows))

setups = ["fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"]
rows = []
for grp, models in GROUPS:
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(
            f"{x['spend'][grp][s][m]['punish']['mean']:.1f} / {x['spend'][grp][s][m]['reward']['mean']:.1f} "
            f"({x['spend'][grp][s][m]['pct_rounds_punish']['mean']:.0f})" for s in setups))
    rows.append("\\midrule")
rows = rows[:-1]
R.append(table(
    "Manager spending per round: punishment / reward tokens (mean over games; each budget is 10 tokens per round), and in "
    "parentheses the share of rounds (\\%) in which the manager punished anyone.",
    "tab:spend", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups), rows, sep="2pt"))

setups = ["fixed", "elected", "rotating", "salary", "costly"]
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME)):
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(
            f"{x['welfare'][grp][s][m]['total']['mean']:.0f}$\\pm${x['welfare'][grp][s][m]['total']['sd']:.0f} "
            f"({x['welfare'][grp][s][m]['from_contributions']['mean']:.0f})" for s in setups))
R.append(table(
    "Group welfare over 20 rounds (total payoff minus endowments, including sanctions and salary), mean $\\pm$ s.d.; in "
    "parentheses the part created by contributions alone ($0.6\\sum c$, at most 1,200). The rest is net transfers, which "
    "rewards create mechanically (one token spent adds three).",
    "tab:welfare", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups), rows))

rows = []
for grp, models in GROUPS:
    for m in models:
        e = x["elections"][grp][m]
        rows.append(f"{NAME[m]} & {e['turnovers']}/{e['contested']} & {e['turnovers_by_tie']} & {e['mean_winner_votes']:.1f} & "
                    f"{e['pct_majority']:.0f} & {e['pct_tie_at_top']:.0f} & {e['pct_self_votes']:.0f} & {e['pct_agent0_wins']:.0f}")
    rows.append("\\midrule")
e = x["elections"]["mixed"]
rows.append(f"Mixed groups & {e['turnovers']}/{e['contested']} & {e['turnovers_by_tie']} & {e['mean_winner_votes']:.1f} & "
            f"{e['pct_majority']:.0f} & {e['pct_tie_at_top']:.0f} & {e['pct_self_votes']:.0f} & {e['pct_agent0_wins']:.0f}")
R.append(table(
    "Elections in the elected, salary, costly, hidden and anonymous setups (the belief setups of the self-hosted models are "
    "not included). Turnovers / contested elections (the three after the first in each game); winner's votes "
    "(of 5); share of elections won with a majority ($\\ge$3 votes), with a tie at the top, share of self-votes, and "
    "share won by \\texttt{agent\\_0}, the first agent on the ballot list (20\\% if list position did not matter). "
    "\\emph{By tie}: turnovers decided by the lowest-index tie rule.",
    "tab:elections", "lccccccc",
    "\\textbf{Model} & \\textbf{Turnover} & \\textbf{By tie} & \\textbf{Winner votes} & \\textbf{Majority \\%} & "
    "\\textbf{Tie \\%} & \\textbf{Self-vote \\%} & \\textbf{agent\\_0 \\%}", rows))

LBL = {"group_contrib": "Group contribution", "own_contrib": "Own contribution",
       "reward": "Rewards given", "punish": "Punishments given"}
rows = []
for k in ("group_contrib", "own_contrib", "reward", "punish"):
    rows.append(LBL[k] + " & " + " & ".join(
        f"{y['reelection'][g][k]['r']:+.2f} ({pval(y['reelection'][g][k]['p'])})" for g in ("api_main", "newer", "open")))
rows.append("\\midrule")
rows.append("Contested (re-elected) & " + " & ".join(
    f"{y['reelection'][g]['contested']} ({y['reelection'][g]['reelected']})" for g in ("api_main", "newer", "open")))
R.append(table(
    "Does the incumbent's record in the five rounds before an election predict its re-election? Point-biserial "
    "correlation $r$ (two-sided $p$) between re-election and each measure, centered within model so that differences between "
    "models do not drive it; elected, salary, costly, hidden and anonymous setups.",
    "tab:reelection", "lccc",
    "\\textbf{Incumbent's record} & \\textbf{Main API} & \\textbf{Newer API} & \\textbf{Self-hosted}", rows,
    sep="2.5pt", wide=True))

A = y["anova"]
ALBL = {"api_contrib_mgr": "Main API, contribution: chat only + 3 manager types",
        "api_contrib_all": "Main API, contribution: all 9 setups",
        "open_coop_mgr": "Self-hosted, cooperation: chat only + 3 manager types",
        "open_coop_all": "Self-hosted, cooperation: all 9 setups"}
rows = [f"{ALBL[k]} & {A[k]['n_games']} & " + " & ".join(
    f"{A[k][f]['partial_eta2']:.2f} / {A[k][f]['omega2']:.2f} ({pval(A[k][f]['p'])})" for f in ("model", "institution", "interaction")) for k in ALBL]
R.append(table(
    "Model versus institution: two-way analysis of variance over games (type II sums of squares); partial $\\eta^2$ / $\\omega^2$ with "
    "$p$ in parentheses. Partial $\\eta^2$ values are not shares of one total and depend on the levels chosen; $\\omega^2$ "
    "is the bias-corrected share of total variance. \\emph{All 9 setups}: no chat, chat only, three manager types, salary, costly, hidden, anonymous.",
    "tab:anova", "lcccc",
    "\\textbf{Data} & \\textbf{Games} & \\textbf{Model} & \\textbf{Institution} & \\textbf{Model $\\times$ institution}", rows))

PAIR = {"claude_grok": "Claude$\\to$Grok", "claude_qwen": "Claude$\\to$Qwen Plus", "gpt4o_qwen": "GPT-4o$\\to$Qwen Plus",
        "grok_claude": "Grok$\\to$Claude", "grok_qwen": "Grok$\\to$Qwen Plus", "qwen_claude": "Qwen Plus$\\to$Claude"}
rows = []
for k, v in d["crossrule"].items():
    s = x["crossrule_spend"][k]
    rows.append(f"{PAIR[k]} & {v['worker_coop']['mean']:.1f} & {pm(v['worker_contrib'])} & {s['punish']['mean']:.1f} / "
                f"{s['reward']['mean']:.1f} & {pm(v['welfare'], 0)}")
W = y["workers_fixed"]
rows.append("\\midrule")
for m in ("claude", "grok", "qwen", "gpt4o"):
    rows.append(f"{NAME[m]} (own family) & --- & {W[m]['mean']:.1f}$\\pm${W[m]['sd']:.1f} & --- & ---")
R.append(table(
    "Cross-Rule (B11): a fixed manager of one family over four workers of another (3 games each). Worker cooperation (\\%), "
    "worker contribution, manager spending per round (punishment / reward), and group welfare. Bottom rows: worker-only "
    "contribution (managers excluded) under a fixed manager of the same family (5 games for Claude and Grok, 10 for "
    "Qwen Plus and GPT-4o), the comparison used in the text.",
    "tab:crossrule", "lcccc",
    "\\textbf{Manager$\\to$Workers} & \\textbf{Coop.} & \\textbf{Contrib.} & \\textbf{P / R} & \\textbf{Welfare}", rows,
    sep="2.5pt", wide=True))

rows = []
for k, v in d["mixed"].items():
    w = ", ".join(f"{NAME[a]} {n}" for a, n in sorted(v["election_winners"].items(), key=lambda z: -z[1]))
    rows.append(f"without {NAME[k.replace('drop_', '')]} & {v['coop']['mean']:.1f} & {pm(v['contrib'])} & "
                f"{v['turnovers']}/{v['contested']} & {w}")
R.append(table(
    "Mixed groups (B4): five of the six main API families, elected manager, full communication (3 games each). "
    "Cooperation (\\%), contribution, turnovers / contested elections, and election winners by family (of 12 elections).",
    "tab:mixed", "lcccl",
    "\\textbf{Composition} & \\textbf{Coop.} & \\textbf{Contribution} & \\textbf{Turnover} & \\textbf{Winners}", rows))

# ---------------------------------------------------------------- appendix A: valid replies
rows = []
for grp, models in GROUPS:
    for m in models:
        cells = [(s, v["parse_ok"], v["replies"]) for s, sv in d[grp].items() for v in [sv.get(m)] if v]
        ok, n = sum(c[1] for c in cells), sum(c[2] for c in cells)
        low = min(cells, key=lambda c: c[1] / c[2])
        rows.append(f"{NAME[m]} & {n:,} & {100 * ok / n:.1f} & {100 * low[1] / low[2]:.1f} ({SETNAME[low[0]]})")
NV = y["nemotron_valid"]
nem = "; ".join(f"{SETNAME[s].lower()} {NV[s]['coop_valid']['mean']:.0f}\\%" for s in ("baseline", "chat_only", "elected"))
PARSE = table(
    "Valid (parseable JSON) contribution replies per model over all its homogeneous setups; an invalid reply counts as a "
    "contribution of 0. Lowest: the setup with the lowest rate. Counting Nemotron's valid replies only, its cooperation is "
    f"{nem} (against " + ", ".join(f"{d['open'][s]['nemotron-3-super-120b']['coop']['mean']:.0f}\\%" for s in ("baseline", "chat_only", "elected"))
    + " when invalid replies count as 0).",
    "tab:parse", "lccc",
    "\\textbf{Model} & \\textbf{Replies} & \\textbf{Valid \\%} & \\textbf{Lowest \\% (setup)}", rows, wide=False)

# ---------------------------------------------------------------- appendix D: tests with Holm correction
METRIC = {"contrib": "contribution", "coop": "cooperation (pp)", "deals": "offers /100", "broken_rate": "broken \\% (pp)"}
tests = [t for t in x["tests"] if t.get("p") is not None]


def family(t):
    grp = "api" if t["model"] in FRONT else "newer" if t["model"] in SAME else "open"
    kind = ("speech" if t["from"] == "baseline" else "manager" if t["from"] == "chat_only" and t["metric"] != "broken_rate"
            else "pay" if t["metric"] == "deals" else t["metric"])
    return grp, t["metric"], kind


fams = {}
for i, t in enumerate(tests):
    fams.setdefault(family(t), []).append(i)
holm = {}
for idx in fams.values():
    order = sorted(idx, key=lambda i: tests[i]["p"])
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - rank) * tests[i]["p"]))
        holm[i] = run
def minus(v):
    return f"{v:.1f}".replace("-", "$-$")


def primary(t):
    """a primary contrast (Table tab:primary_full) on contribution"""
    return t["metric"] == "contrib" and (t["from"], t["to"]) in (("baseline", "chat_only"), ("chat_only", "elected"))


def trow(i, t):
    return (f"{NAME[t['model']]} & {METRIC[t['metric']]} & {SETNAME[t['from']]} $\\to$ {SETNAME[t['to']]} & "
            f"{'+' if t['diff'] >= 0 else ''}{minus(t['diff'])} & [{minus(t['ci'][0])}, {minus(t['ci'][1])}] & "
            f"{pval(t['p'])} & {pval(holm[i])}")


THEAD = ("\\textbf{Model} & \\textbf{Metric} & \\textbf{Contrast} & \\textbf{Diff.} & \\textbf{95\\% CI} & \\textbf{$p$} & "
         "\\textbf{Holm $p$}")
# two tables, so that each fits on a page (AAAI does not allow a smaller font)
TESTS = table(
    "Welch's $t$-tests for the secondary contrasts discussed in the text, API models (game = unit; $n$ = 3 to 10 per side). "
    "Difference = second setup minus first; 95\\% confidence interval; two-sided $p$, and Holm-adjusted $p$ within each "
    "family of contrasts (same model set, metric and kind of contrast: speech, manager, pay, or promises). Contrasts where "
    "both setups have zero variance (e.g., GPT-4o's offers, always 0) are omitted. The primary contrasts (no chat $\\to$ chat "
    "only and chat only $\\to$ elected, contribution) are in Table~\\ref{tab:primary_full} with their own correction; the Holm "
    "families here still include them.",
    "tab:tests", "lllcccc", THEAD, [trow(i, t) for i, t in enumerate(tests) if t["model"] not in OPEN and not primary(t)],
    sep="3pt")
TESTS_OSS = table(
    "Welch's $t$-tests for the secondary contrasts discussed in the text, self-hosted models (as in Table~\\ref{tab:tests}).",
    "tab:tests_oss", "lllcccc", THEAD, [trow(i, t) for i, t in enumerate(tests) if t["model"] in OPEN], sep="3pt")

# ---------------------------------------------------------------- appendix E: robustness of stated intentions
rows = []
for grp, models in GROUPS:
    for m in models:
        cells = []
        for s in ("chat_only", "elected"):
            c = I[grp][s][m]
            if not c.get("messages"):
                cells += ["---"] * 4
                continue
            st_, n = c.get("stated", 0), c.get("messages", 0)
            cells += [f"{100 * st_ / n:.0f}\\%",
                      f"{c.get('broken_3', 0)} / {c.get('broken_5', 0)} / {c.get('broken_8', 0)} of {st_}",
                      f"{c.get('first_person', 0)}", f"{c.get('fp_broken_5', 0)}"]
        rows.append(NAME[m] + " & " + " & ".join(cells))
    rows.append("\\midrule")
rows = rows[:-1]
ROBUST = table(
    "Robustness of broken stated intentions (one regular expression for every model). Per condition: share of public "
    "messages that state a contribution; broken at thresholds of 3 / 5 / 8 tokens; number of first-person, unconditional "
    "commitments (``I will contribute 10'', no ``if'') and how many of them are broken at 5 tokens. Statements that are "
    "proposals (``let's each give 10'') or conditional (``10 if you do'') make up the rest.",
    "tab:robust", "lcccccccc",
    "& \\multicolumn{4}{c}{\\textbf{Chat only (no manager)}} & \\multicolumn{4}{c}{\\textbf{Elected manager}} \\\\ "
    "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
    "\\textbf{Model} & \\textbf{Stating} & \\textbf{Broken 3/5/8} & \\textbf{1st pers.} & \\textbf{broken} & "
    "\\textbf{Stating} & \\textbf{Broken 3/5/8} & \\textbf{1st pers.} & \\textbf{broken}", rows, sep="3pt")

O = y["offers"]
rows = []
CATS = ("reply", "targeted", "reciprocal", "pledge", "other")
for grp, models in GROUPS:
    for m in models:
        tot_ = {k: sum(O[grp][s][m].get(k, 0) for s in ("elected", "salary", "costly")) for k in CATS + ("offers", "impossible")}
        if not tot_["offers"]:
            rows.append(f"{NAME[m]} & 0 & " + " & ".join(["---"] * 6))
            continue
        rows.append(f"{NAME[m]} & {tot_['offers']} & " + " & ".join(
            f"{100 * tot_[k] / tot_['offers']:.0f}" for k in CATS) + f" & {tot_['impossible']}")
    rows.append("\\midrule")
rows = rows[:-1]
OV = y["over20"]
over = sum(v["over_20"] for g in OV.values() for v in g.values())
msgs = sum(v["messages"] for g in OV.values() for v in g.values())
OFFERS = table(
    "What vote-contingent offers contain (elected, salary and costly setups pooled; automatic split; its agreement with hand labels is in "
    "Table~\\ref{tab:validation}). Shares (\\%) of offers that reply to or refuse another offer (Reply), promise the voter a targeted "
    "private benefit (Targ.), offer reciprocal political support such as ``I'll vote for you next time'' (Recip.), pledge "
    "a public contribution (Pledge), or none of these (Other). Imp.: number of offers naming an amount the rules do not "
    f"allow (a contribution above 20 or a sanction above the budget of 10). Across all {msgs:,} messages, {over} mention "
    "contributing more than 20 tokens.",
    "tab:offers", "lccccccc",
    "\\textbf{Model} & \\textbf{Offers} & \\textbf{Reply} & \\textbf{Targ.} & \\textbf{Recip.} & \\textbf{Pledge} & "
    "\\textbf{Other} & \\textbf{Imp.}", rows, sep="2.5pt", wide=True)

print("% Generated by hg-rerun/analysis/appendix_tables.py from paper_numbers.json, extras.json and revision2.json.")
print("% Do not edit by hand. Defines table macros that the paper places in their sections.\n")
for name, body in T:
    print(f"\\newcommand{{\\{name}}}{{%\n{body.rstrip()}\n}}\n")
for name, body in (("hgParseTable", PARSE), ("hgTestsTable", TESTS), ("hgTestsOssTable", TESTS_OSS), ("hgRobustTable", ROBUST), ("hgOffersTable", OFFERS)):
    print(f"\\newcommand{{\\{name}}}{{%\n{body.rstrip()}\n}}\n")
# mixed groups before cross-rule, so that the appendix pages pack well
_cr = next(k for k, t in enumerate(R) if 'tab:crossrule' in t)
R.append(R.pop(_cr))
print("\\newcommand{\\hgResultTables}{%\n" + "\n\n".join(t.rstrip() for t in R) + "\n}")
