"""LaTeX tables for the round-3 analyses (paper_data/round3.json). Defines macros only; the paper places them.

  python analysis/round3_tables.py paper_data/round3.json > round3_tables.tex

  \\hgPrimaryTable (Results)    primary contrasts with Holm over the whole family
  \\hgNeutralTable (Results)    offers under the deal and the neutral election prompt (LLM judge)
  \\hgMechanismTable (Results)  manager without sanctions, reward rule, automatic reward without a manager
  \\hgAccountTable, \\hgAggregateTable, \\hgVersionTable, \\hgPromptQuantTable, \\hgEndgameTable,
  \\hgReelectTable (Appendix)
"""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
NAME = {"gpt4o": "GPT-4o", "claude": "Claude", "gemini": "Gemini", "deepseek": "DeepSeek V3", "grok": "Grok",
        "qwen": "Qwen Plus", "gpt5": "GPT-5", "deepseek31": "DeepSeek V3.1", "gpt41": "GPT-4.1", "gpt5min": "GPT-5 minimal",
        "gpt-oss-120b": "gpt-oss", "nemotron-3-super-120b": "Nemotron", "ling-3.0-flash": "Ling",
        "qwen3.8-27b": "Qwen3.8", "gemma-4-31b": "Gemma"}
API_MAIN = ["gpt4o", "claude", "gemini", "deepseek", "grok", "qwen"]
NEWER = ["gpt5", "deepseek31"]
OSS = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]


def num(x, fmt="+.1f"):
    if float(format(x, fmt)) == 0:      # no "-0.0"
        x = 0.0
    return format(x, fmt).replace("-", "$-$")


def p(x):
    return "---" if x is None else ("$<$0.001" if x < 0.001 else f"{x:.3f}")


def table(macro, caption, label, cols, header, rows, wide=False, sep="3pt"):
    env = "table*" if wide else "table"
    out = [f"\\newcommand{{\\{macro}}}{{%", f"\\begin{{{env}}}[t]", "\\centering", "\\small",
           f"\\setlength{{\\tabcolsep}}{{{sep}}}", f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [x if x.startswith("\\midrule") else x + " \\\\" for x in rows]
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", "}", ""]
    return "\n".join(out)


def grp(models):
    return ["\\midrule"] if models else []


T = []

# ---------------------------------------------------------------- primary contrasts
P = {(t["model"], t["contrast"]): t for t in r["A_primary"]["tests"]}


def pc(m, c):
    t = P.get((m, c))
    if not t:
        return "--- & ---"
    return f"{num(t['diff'])} [{num(t['ci'][0], '.1f')}, {num(t['ci'][1], '.1f')}] & {p(t['holm'])}"


def pshort(m, c):
    t = P.get((m, c))
    return "--- & ---" if not t else f"{num(t['diff'])} & {p(t['holm'])}"


def pfull(m, c):
    t = P.get((m, c))
    return "--- & --- & ---" if not t else f"{pc(m, c)} & {p(t['holm_perm'])}"


fam = r["A_primary"]["family_size"]
ok = [f"{NAME[t['model']]}'s {t['contrast']} effect" for t in r["A_primary"]["tests"] if t["holm_perm"] < 0.05]
surv = ", ".join(ok[:-1]) + " and " + ok[-1] if len(ok) > 1 else "".join(ok)
rows, rows_full = [], []
for models in (API_MAIN, NEWER, OSS):
    if rows:
        rows.append("\\midrule")
        rows_full.append("\\midrule")
    rows += [f"{NAME[m]} & {pshort(m, 'communication')} & {pshort(m, 'manager')}" for m in models]
    rows_full += [f"{NAME[m]} & {pfull(m, 'communication')} & {pfull(m, 'manager')}" for m in models]
T.append(table(
    "hgPrimaryTable",
    f"Primary contrasts: communication (no chat $\\to$ chat only) and manager (chat only $\\to$ elected "
    f"manager), difference in mean contribution (tokens), game as the unit, with Holm-adjusted $p$ (Welch) over all {fam} "
    f"tests. Confidence intervals and permutation tests: Table~\\ref{{tab:primary_full}}.",
    "tab:primary", "lcccc",
    "& \\multicolumn{2}{c}{\\textbf{Communication}} & \\multicolumn{2}{c}{\\textbf{Manager}} \\\\ "
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n\\textbf{Model} & $\\Delta$ & Holm $p$ & $\\Delta$ & Holm $p$",
    rows, sep="4pt"))
T.append(table(
    "hgPrimaryFullTable",
    f"Primary contrasts in full (Table~\\ref{{tab:primary}}): difference in mean contribution with 95\\% confidence interval "
    f"(Welch), Holm-adjusted $p$ over all {fam} tests, and Holm-adjusted $p$ from exact or Monte Carlo permutation tests "
    f"(which cannot go below 0.10 for 3 against 3 games). With permutation tests, {surv} stay below 0.05.",
    "tab:primary_full", "lcccccc",
    "& \\multicolumn{3}{c}{\\textbf{Communication}} & \\multicolumn{3}{c}{\\textbf{Manager}} \\\\ "
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n\\textbf{Model} & $\\Delta$ [95\\% CI] & Holm $p$ & Perm.\\ $p$ & "
    "$\\Delta$ [95\\% CI] & Holm $p$ & Perm.\\ $p$",
    rows_full, wide=True))

# ---------------------------------------------------------------- neutral prompt (LLM judge)
N = r["N_llm_judge"]["offer_share_of_election_opportunities"]


def sh(m, s):
    v = N[m]["share"].get(s)
    return "---" if not v else f"{v['mean']:.0f}"


rows = []
for m in ["claude", "gemini", "grok", "deepseek31", "gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b"]:
    rows.append(f"{NAME[m]} & {sh(m, 'elected')} & {sh(m, 'salary')} & {sh(m, 'neutral')} & {sh(m, 'neutral_salary')}")
V = r["N_llm_judge"]["validation"]
T.append(table(
    "hgNeutralTable",
    "Vote-contingent offers under the original election prompt (``send ONE private deal message \\dots\\ to secure their "
    "vote'') and a neutral one (``send ONE private message \\dots\\ before the vote''): share (\\%) of election-message "
    "opportunities (5 agents $\\times$ 4 elections) used for an offer, as labeled by an LLM judge applying the human "
    f"codebook (precision {V['precision']:.2f}, recall {V['recall']:.2f} against the hand labels; Table~\\ref{{tab:validation}}). "
    "Elected manager without and with a salary; 8 games per cell under the deal prompt (10 for Gemini without salary), 5 under the neutral prompt.",
    "tab:neutral", "lcccc",
    "& \\multicolumn{2}{c}{\\textbf{Deal prompt}} & \\multicolumn{2}{c}{\\textbf{Neutral prompt}} \\\\ "
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n\\textbf{Model} & no salary & salary & no salary & salary", rows))

# ---------------------------------------------------------------- mechanism
C = r["C_mechanism"]


def mc(m, s):
    v = C[m][s]["contrib"]
    return f"{v['mean']:.1f}$_{{{v['n']}}}$"


rows = []
for m in ["gpt4o", "gemini", "qwen"]:
    rows.append(f"{NAME[m]} & " + " & ".join(mc(m, s) for s in ("chat", "fixed", "nosanction", "autoreward", "sysreward")))
rows.append("\\midrule")
for m in OSS:
    rows.append(f"{NAME[m]} & " + " & ".join(mc(m, s) for s in ("chat", "fixed", "nosanction", "autoreward", "sysreward")))
T.append(table(
    "hgMechanismTable",
    "Taking the manager apart: mean contribution (subscript: games). \\emph{Fixed}: an LLM manager with a budget. "
    "\\emph{No budget}: the same manager without sanctions (authority and messages only). \\emph{Rule}: the manager's "
    "sanctions follow a fixed rule (10 reward tokens split among those who gave at least 10; no punishment). "
    "\\emph{Auto}: the same rule applied by the game, with no manager and at no cost to any player. All with full communication.",
    "tab:mechanism", "lccccc",
    "\\textbf{Model} & \\textbf{Chat only} & \\textbf{Fixed} & \\textbf{No budget} & \\textbf{Rule} & \\textbf{Auto}", rows))

# ---------------------------------------------------------------- accountability
D = r["D_accountability"]["per_model"]
rows = []
for m in ["gpt4o", "gemini", "qwen", "deepseek"] + OSS:
    v = D[m]
    fb, rb = v["fixed_ballot"], v["random_ballot"]
    rows.append(f"{NAME[m]} & {v['bad_replaced_at_6'][0]}/{v['bad_replaced_at_6'][1]} & {v['good_replaced_at_6'][0]}/{v['good_replaced_at_6'][1]} & "
                f"{fb.get('agent0_wins', 0)}/{fb['elections']} & {rb.get('agent0_wins', 0)}/{rb['elections']} & "
                f"{fb.get('tie_top', 0)}/{fb['elections']} & {rb.get('tie_top', 0)}/{rb['elections']}")
PB = r["D_accountability"]["pooled"]
T.append(table(
    "hgAccountTable",
    "Elections. Left: \\texttt{agent\\_0} starts in office and, while it holds office, either punishes the two highest "
    "contributors (\\emph{bad}) or rewards everyone who gave at least 10 (\\emph{good}); the first election is at round 6. "
    f"Games in which it is voted out at round 6 (pooled: bad {PB['bad_replaced'][0]}/{PB['bad_replaced'][1]}, good "
    f"{PB['good_replaced'][0]}/{PB['good_replaced'][1]}, Fisher $p{{=}}{PB['fisher_p']:.2f}$). Right: elections won by "
    "\\texttt{agent\\_0} and elections with a tie at the top, with the fixed ballot (ties to the lowest index) and with a "
    "ballot shuffled for each voter and ties broken at random.",
    "tab:accountability", "lcccccc",
    "& \\multicolumn{2}{c}{\\textbf{Voted out at round 6}} & \\multicolumn{2}{c}{\\textbf{agent\\_0 wins}} & "
    "\\multicolumn{2}{c}{\\textbf{Tie at the top}} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    "\\textbf{Model} & bad & good & fixed & random & fixed & random", rows, wide=True))

# ---------------------------------------------------------------- aggregate visibility
E = r["E_visibility_contributions"]


def br(x):
    return "---" if not x["stated"] else f"{x['share']:.0f}"


rows = []
for m in ["gpt4o", "gemini", "qwen"] + OSS:
    v = E[m]
    a, b = v["chat_vs_aggregate"], v["elected_vs_aggregate"]
    rows.append(f"{NAME[m]} & {a['contrib']['a']['mean']:.1f} & {a['contrib']['b']['mean']:.1f} & {br(a['broken']['individual'])} & "
                f"{br(a['broken']['aggregate'])} & {b['contrib']['a']['mean']:.1f} & {b['contrib']['b']['mean']:.1f} & "
                f"{br(b['broken']['individual'])} & {br(b['broken']['aggregate'])}")
T.append(table(
    "hgAggregateTable",
    "Visibility of contributions. \\emph{Ind.}: every agent sees what each player gave (the default). \\emph{Agg.}: agents "
    "see only the pool total and the mean (5 games per cell). Contribution and broken stated intentions (\\%).",
    "tab:aggregate", "lcccccccc",
    "& \\multicolumn{4}{c}{\\textbf{Chat only}} & \\multicolumn{4}{c}{\\textbf{Elected manager}} \\\\ "
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n& \\multicolumn{2}{c}{Contribution} & \\multicolumn{2}{c}{Broken \\%} & "
    "\\multicolumn{2}{c}{Contribution} & \\multicolumn{2}{c}{Broken \\%} \\\\\n\\textbf{Model} & Ind. & Agg. & Ind. & Agg. & Ind. & Agg. & Ind. & Agg.",
    rows, wide=True))

# ---------------------------------------------------------------- version vs reasoning
F = r["F_version"]
NS = r["N_llm_judge"]["offer_share_of_election_opportunities"]
rows = []
for m in ("gpt4o", "gpt41", "gpt5min", "gpt5"):
    v = F[m]
    offs = " / ".join("---" if not NS[m]["share"].get(s) else f"{NS[m]['share'][s]['mean']:.0f}" for s in ("elected", "salary", "costly"))
    rows.append(f"{NAME[m]} & {v['chat_contrib']['mean']:.1f} & {v['elected_contrib']['mean']:.1f} & {offs} & "
                f"{v['broken_chat']['broken']}/{v['broken_chat']['stated']}")
T.append(table(
    "hgVersionTable",
    "Version or reasoning? GPT-4.1 is GPT-4o's non-reasoning successor; \\emph{GPT-5 minimal} is GPT-5 with reasoning effort "
    "``minimal'' (3 games per cell for both). Contribution with chat only and with an elected manager; share (\\%) of "
    "election-message opportunities used for a vote-contingent offer (LLM judge) without salary / with salary / costly; "
    "broken stated intentions in chat-only groups.",
    "tab:version", "lcccc",
    "\\textbf{Model} & \\textbf{Chat} & \\textbf{Elected} & \\textbf{Offers \\%} & \\textbf{Broken}", rows, wide=True))

# ---------------------------------------------------------------- prompt robustness and quantisation
G = r["G_no_strategic"]
rows = []
for m in ["gpt4o", "claude", "gemini"] + OSS:
    v = G[m]
    rows.append(f"{NAME[m]} & {v['chat']['contrib']['a']['mean']:.1f} & {v['chat']['contrib']['b']['mean']:.1f} & "
                f"{v['elected']['contrib']['a']['mean']:.1f} & {v['elected']['contrib']['b']['mean']:.1f} & "
                f"{br(v['chat']['broken'][0])} & {br(v['chat']['broken'][1])}")
H = r["H_quantisation"]
rows.append("\\midrule")
rows.append("\\multicolumn{7}{@{}l}{\\emph{Qwen3.8 27B: full precision (BF16) vs NVFP4, 8 games each; contribution / broken \\%}} \\\\")
for s, lbl in (("baseline", "No chat"), ("chat", "Chat only"), ("elected", "Elected"), ("salary", "Salary")):
    v = H[s]
    b16, b4 = v["broken"]["bf16"], v["broken"]["nvfp4"]
    rows.append(f"\\quad {lbl} & {v['contrib']['a']['mean']:.1f} & {v['contrib']['b']['mean']:.1f} & "
                f"{p(v['contrib']['p'])} & & {br(b16)} & {br(b4)}")
T.append(table(
    "hgPromptQuantTable",
    "Robustness. Top: without ``Be strategic.'' in the system prompt (\\emph{No}) against the default (\\emph{Std.}); "
    "contribution with chat only and with an elected manager, and broken stated intentions (\\%) with chat only "
    "(3 games per cell for the API models, 5 for the self-hosted models). Bottom: the 4-bit NVFP4 build of Qwen3.8 against "
    "the full-precision model (columns: BF16, NVFP4, Welch $p$ for contribution, broken \\% BF16 and NVFP4).",
    "tab:promptquant", "lcccccc",
    "& \\multicolumn{2}{c}{\\textbf{Chat only}} & \\multicolumn{2}{c}{\\textbf{Elected}} & \\multicolumn{2}{c}{\\textbf{Broken \\%}} \\\\ "
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n\\textbf{Model} & Std. & No & Std. & No & Std. & No", rows, wide=True))

# ---------------------------------------------------------------- endgame
I = r["I_endgame"]
rows = []
for models in (API_MAIN, NEWER, OSS):
    if rows:
        rows.append("\\midrule")
    for m in models:
        c, e = I[m].get("chat"), I[m].get("elected")

        def b(x, k):
            return f"{x[k]['broken']}/{x[k]['stated']}" if x and x[k]["stated"] else "---"
        rows.append(f"{NAME[m]} & {c['contrib_r1_19']:.1f} & {c['contrib_r20']:.1f} & {b(c, 'broken_r1_19')} & {b(c, 'broken_r20')} & "
                    f"{e['contrib_r1_19']:.1f} & {e['contrib_r20']:.1f} & {b(e, 'broken_r1_19')} & {b(e, 'broken_r20')}")
T.append(table(
    "hgEndgameTable",
    "The last round. Mean contribution in rounds 1--19 and in round 20, and broken / stated intentions in rounds 1--19 and "
    "in round 20, with chat only and with an elected manager.",
    "tab:endgame", "lcccccccc",
    "& \\multicolumn{4}{c}{\\textbf{Chat only}} & \\multicolumn{4}{c}{\\textbf{Elected manager}} \\\\ "
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n& \\multicolumn{2}{c}{Contribution} & \\multicolumn{2}{c}{Broken} & "
    "\\multicolumn{2}{c}{Contribution} & \\multicolumn{2}{c}{Broken} \\\\\n\\textbf{Model} & 1--19 & 20 & 1--19 & 20 & 1--19 & 20 & 1--19 & 20",
    rows, wide=True))

# ---------------------------------------------------------------- re-election, clustered
K = r["K_reelection_clustered"]
LBL = {"main_api": "Main API", "newer": "Newer API", "self_hosted": "Self-hosted"}
rows = []
for g in ("main_api", "newer", "self_hosted"):
    v = K[g]
    cells = []
    for k in ("group_contrib", "own_contrib", "reward", "punish"):
        x = v[k]
        cells.append("---" if "error" in x else f"{num(x['log_odds_per_sd'], '+.2f')} ({p(x['p'])})")
    rows.append(f"{LBL[g]} & {v['reelected']}/{v['elections']} & " + " & ".join(cells))
T.append(table(
    "hgReelectTable",
    "Does the incumbent's record in the five rounds before an election predict re-election? Logistic regression of "
    "re-election on each record measure (standardized, centered within model) with model effects; standard errors "
    "clustered by game. Log-odds per s.d. with $p$. Elected, salary, costly, hidden and anonymous setups.",
    "tab:reelect_clustered", "lccccc",
    "\\textbf{Models} & \\textbf{Re-elected} & \\textbf{Group contr.} & \\textbf{Own contr.} & \\textbf{Rewards} & \\textbf{Punishments}",
    rows, wide=True))

print("% generated by analysis/round3_tables.py from paper_data/round3.json; do not edit by hand\n")
print("\n".join(T))

# The same tables without their float environment (\hgEndgameTableInner etc.), so that the paper can stack several
# tables in one float with fixed spacing (a float page holding several floats spreads them apart).
for t in T:
    name = t.split("{", 2)[1].split("}")[0]
    body = [x for x in t.strip().split("\n")[1:-1] if not x.startswith(("\\begin{table", "\\end{table"))]
    print(f"\\newcommand{{{name}Inner}}{{%\n" + "\n".join(body) + "\n}\n")
