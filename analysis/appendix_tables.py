"""LaTeX tables for the appendix, generated from paper_numbers.json and extras.json.

  python analysis/appendix_tables.py paper_data/paper_numbers.json paper_data/extras.json > appendix_tables.tex
"""
import json
import sys

d = json.load(open(sys.argv[1]))
x = json.load(open(sys.argv[2]))

NAME = {"gpt4o": "GPT-4o", "claude": "Claude", "gemini": "Gemini", "deepseek": "DeepSeek V3", "grok": "Grok",
        "qwen": "Qwen Plus", "gpt5": "GPT-5", "deepseek31": "DeepSeek V3.1", "gpt-oss-120b": "gpt-oss",
        "nemotron-3-super-120b": "Nemotron", "ling-3.0-flash": "Ling", "qwen3.8-27b": "Qwen3.8",
        "gemma-4-31b": "Gemma"}
FRONT = ["gpt4o", "claude", "gemini", "deepseek", "grok", "qwen"]
SAME = ["gpt5", "deepseek31"]
OPEN = ["gpt-oss-120b", "nemotron-3-super-120b", "ling-3.0-flash", "qwen3.8-27b", "gemma-4-31b"]
SETNAME = {"baseline": "No chat", "chat_only": "Chat only", "public_chat": "Public only", "private_chat": "Private only",
           "fixed": "Fixed", "elected": "Elected", "rotating": "Rotating", "salary": "Salary", "costly": "Costly",
           "hidden": "Hidden", "anonymous": "Anonymous", "belief_unknown": "Belief: unknown",
           "belief_human": "Belief: all human", "belief_mixed": "Belief: mixed"}


def pm(v, nd=1):
    return "---" if not v else f"{v['mean']:.{nd}f}$\\pm${v['sd']:.{nd}f}"


def num(v, nd=1):
    return "---" if not v else f"{v['mean']:.{nd}f}"


def table(caption, label, cols, header, rows, size="\\scriptsize", sep="3pt"):
    out = ["\\begin{table}[!htbp]", "\\centering", size, f"\\setlength{{\\tabcolsep}}{{{sep}}}",
           f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r if r.startswith("\\midrule") else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(out)


def cell(v):
    if not v:
        return "---"
    return f"{v['contrib']['mean']:.1f}$\\pm${v['contrib']['sd']:.1f} ({v['coop']['mean']:.0f})"


T = []
# frontier by institution
setups = ["baseline", "chat_only", "fixed", "elected", "rotating"]
T.append(table(
    "Frontier models by institution: mean contribution $\\pm$ s.d. over games (cooperation rate, \\%). "
    "3 games for no chat and chat only, 5 per manager type. Manager setups have full communication.",
    "tab:frontier_full", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups),
    [NAME[m] + " & " + " & ".join(cell(d["frontier"][s][m]) for s in setups) for m in FRONT]))

setups = ["elected", "salary", "costly", "hidden", "anonymous"]
T.append(table(
    "Frontier models under manager pay and sanction visibility (elected manager, full communication): mean contribution "
    "$\\pm$ s.d. (cooperation rate, \\%). 5 games for Elected (the no-salary, transparent control), 3 otherwise.",
    "tab:frontier_pay", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups),
    [NAME[m] + " & " + " & ".join(cell(d["frontier"][s][m]) for s in setups) for m in FRONT]))

# same generation
setups = ["baseline", "fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"]
T.append(table(
    "Newer versions (3 games each): mean contribution $\\pm$ s.d. (cooperation rate, \\%).",
    "tab:samegen_full", "l" + "c" * 4,
    "\\textbf{Setup} & \\textbf{GPT-5} & \\textbf{DeepSeek V3.1} & \\textbf{GPT-4o} & \\textbf{DeepSeek V3}",
    [SETNAME[s] + " & " + " & ".join(cell(d[g][s][m]) for g, m in
                                      (("samegen", "gpt5"), ("samegen", "deepseek31"), ("frontier", "gpt4o"), ("frontier", "deepseek")))
     for s in setups]))

# open models
setups = ["baseline", "public_chat", "private_chat", "chat_only", "fixed", "elected", "rotating", "salary", "costly",
          "hidden", "anonymous", "belief_unknown", "belief_human", "belief_mixed"]


def ocell(v):
    return "---" if not v else f"{v['coop']['mean']:.0f}$\\pm${v['coop']['sd']:.0f} / {v['contrib']['mean']:.1f}"


T.append(table(
    "Open-weight models (3 games each): cooperation rate $\\pm$ s.d. (\\%) / mean contribution. \\emph{Chat only} is full "
    "communication without a manager; all setups below it have full communication and (except where stated) an elected "
    "manager; the belief setups vary what agents are told about the other players (default: all AI).",
    "tab:open_full", "l" + "c" * len(OPEN),
    "\\textbf{Setup} & " + " & ".join(f"\\textbf{{{NAME[m]}}}" for m in OPEN),
    [SETNAME[s] + " & " + " & ".join(ocell(d["open"][s][m]) for m in OPEN) for s in setups]))

# deals
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME), ("open", OPEN)):
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(pm(d[grp][s][m]["deals_strict"]) for s in ("elected", "salary", "costly"))
                    + " & " + " & ".join(num(d[grp][s][m]["deals_keyword"]) for s in ("elected", "salary", "costly")))
    rows.append("\\midrule")
rows = rows[:-1]
T.append(table(
    "Vote-buying offers per 100 agent-rounds by manager pay (elected manager). Strict: private messages that tie a vote "
    "to a material benefit (mean $\\pm$ s.d.). Keyword: the broader deal-offer category of a keyword classifier (mean).",
    "tab:deals_full", "lcccccc",
    "& \\multicolumn{3}{c}{\\textbf{Strict}} & \\multicolumn{3}{c}{\\textbf{Keyword}} \\\\ \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    "\\textbf{Model} & No salary & Salary & Costly & No salary & Salary & Costly", rows))

# promises per model
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME)):
    for m in models:
        cells = []
        for s in ("chat_only", "elected", "hidden", "anonymous"):
            v = d[grp][s].get(m)
            cells.append("---" if not v else f"{v['broken_explicit']}/{v['explicit']} ({v['flag_code_rule']}/{v['promises_any']})")
        rows.append(NAME[m] + " & " + " & ".join(cells))
T.append(table(
    "Broken explicit promises / explicit promises per model, summed over games. In parentheses: messages flagged by a "
    "two-sided rule that also counts giving more than stated and scores vague statements (``I'll contribute generously'') "
    "as fixed numbers; most of these flags are not broken promises.",
    "tab:promises_full", "lcccc",
    "\\textbf{Model} & \\textbf{Chat only} & \\textbf{Transparent} & \\textbf{Hidden} & \\textbf{Anonymous}", rows))

# spending
setups = ["fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"]
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME), ("open", OPEN)):
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(
            f"{x['spend'][grp][s][m]['punish']['mean']:.1f} / {x['spend'][grp][s][m]['reward']['mean']:.1f} "
            f"({x['spend'][grp][s][m]['pct_rounds_punish']['mean']:.0f})" for s in setups))
    rows.append("\\midrule")
rows = rows[:-1]
T.append(table(
    "Manager spending per round: punishment / reward tokens (mean over games; each budget is 10 tokens per round), and in "
    "parentheses the share of rounds (\\%) in which the manager punished anyone.",
    "tab:spend", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups), rows, size="\\tiny", sep="2pt"))

# welfare
setups = ["fixed", "elected", "rotating", "salary", "costly"]
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME)):
    for m in models:
        rows.append(NAME[m] + " & " + " & ".join(
            f"{x['welfare'][grp][s][m]['total']['mean']:.0f}$\\pm${x['welfare'][grp][s][m]['total']['sd']:.0f} "
            f"({x['welfare'][grp][s][m]['from_contributions']['mean']:.0f})" for s in setups))
T.append(table(
    "Group welfare over 20 rounds (total payoff minus endowments, including sanctions and salary), mean $\\pm$ s.d.; in "
    "parentheses the part created by contributions alone ($0.6\\sum c$, at most 1,200). The rest is net transfers.",
    "tab:welfare", "l" + "c" * len(setups),
    "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SETNAME[s]}}}" for s in setups), rows))

# elections
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME), ("open", OPEN)):
    for m in models:
        e = x["elections"][grp][m]
        rows.append(f"{NAME[m]} & {e['turnovers']}/{e['contested']} & {e['turnovers_by_tie']} & {e['mean_winner_votes']:.1f} & "
                    f"{e['pct_majority']:.0f} & {e['pct_tie_at_top']:.0f} & {e['pct_self_votes']:.0f} & {e['pct_agent0_wins']:.0f}")
    rows.append("\\midrule")
e = x["elections"]["mixed"]
rows.append(f"Mixed groups & {e['turnovers']}/{e['contested']} & {e['turnovers_by_tie']} & {e['mean_winner_votes']:.1f} & "
            f"{e['pct_majority']:.0f} & {e['pct_tie_at_top']:.0f} & {e['pct_self_votes']:.0f} & {e['pct_agent0_wins']:.0f}")
T.append(table(
    "Elections in the elected, salary, costly, hidden and anonymous setups (the belief setups of the open models are "
    "not included). Turnovers / contested elections (the three after the first in each game); winner's votes "
    "(of 5); share of elections won with a majority ($\\ge$3 votes), with a tie at the top, share of self-votes, and "
    "share won by \\texttt{agent\\_0}, the first agent on the ballot list (20\\% if list position did not matter). "
    "\\emph{By tie}: turnovers decided by the lowest-index tie rule.",
    "tab:elections", "lccccccc",
    "\\textbf{Model} & \\textbf{Turnover} & \\textbf{By tie} & \\textbf{Winner votes} & \\textbf{Majority \\%} & "
    "\\textbf{Tie \\%} & \\textbf{Self-vote \\%} & \\textbf{agent\\_0 \\%}",
    rows, size="\\tiny", sep="3pt"))

# valid replies
rows = []
for grp, models in (("frontier", FRONT), ("samegen", SAME), ("open", OPEN)):
    for m in models:
        cells = [(s, v["parse_ok"], v["replies"]) for s, sv in d[grp].items() for v in [sv.get(m)] if v]
        ok, n = sum(c[1] for c in cells), sum(c[2] for c in cells)
        low = min(cells, key=lambda c: c[1] / c[2])
        rows.append(f"{NAME[m]} & {n:,} & {100 * ok / n:.1f} & {100 * low[1] / low[2]:.1f} ({SETNAME[low[0]]})")
T.append(table(
    "Valid (parseable JSON) contribution replies per model over all its homogeneous setups; an invalid reply counts as a "
    "contribution of 0. Lowest: the setup with the lowest rate.",
    "tab:parse", "lccc",
    "\\textbf{Model} & \\textbf{Replies} & \\textbf{Valid \\%} & \\textbf{Lowest \\% (setup)}", rows))

# cross-rule
rows = []
PAIR = {"claude_grok": "Claude$\\to$Grok", "claude_qwen": "Claude$\\to$Qwen Plus", "gpt4o_qwen": "GPT-4o$\\to$Qwen Plus",
        "grok_claude": "Grok$\\to$Claude", "grok_qwen": "Grok$\\to$Qwen Plus", "qwen_claude": "Qwen Plus$\\to$Claude"}
for k, v in d["crossrule"].items():
    s = x["crossrule_spend"][k]
    rows.append(f"{PAIR[k]} & {v['worker_coop']['mean']:.1f} & {pm(v['worker_contrib'])} & {s['punish']['mean']:.1f} / "
                f"{s['reward']['mean']:.1f} & {pm(v['welfare'], 0)}")
T.append(table(
    "Cross-Rule (B11): a fixed manager of one family over four workers of another (3 games each). Worker cooperation (\\%), "
    "worker contribution, manager spending per round (punishment / reward), and group welfare.",
    "tab:crossrule", "lcccc",
    "\\textbf{Manager$\\to$Workers} & \\textbf{Coop.} & \\textbf{Contribution} & \\textbf{P / R} & \\textbf{Welfare}", rows))

# mixed
rows = []
for k, v in d["mixed"].items():
    w = ", ".join(f"{NAME[a]} {n}" for a, n in sorted(v["election_winners"].items(), key=lambda z: -z[1]))
    rows.append(f"without {NAME[k.replace('drop_', '')]} & {v['coop']['mean']:.1f} & {pm(v['contrib'])} & "
                f"{v['turnovers']}/{v['contested']} & {w}")
T.append(table(
    "Mixed groups (B4): five of the six frontier families, elected manager, full communication (3 games each). "
    "Cooperation (\\%), contribution, turnovers / contested elections, and election winners by family (of 12 elections).",
    "tab:mixed", "lcccl",
    "\\textbf{Composition} & \\textbf{Coop.} & \\textbf{Contribution} & \\textbf{Turnover} & \\textbf{Winners}", rows))

# tests
METRIC = {"contrib": "contribution", "coop": "cooperation (pp)", "deals": "offers /100", "broken_rate": "broken \\% (pp)"}
rows = []
for t in x["tests"]:
    if t.get("p") is None:
        continue
    rows.append(f"{NAME[t['model']]} & {METRIC[t['metric']]} & {SETNAME[t['from']]} $\\to$ {SETNAME[t['to']]} & "
                f"{t['diff']:+.1f} & [{t['ci'][0]:.1f}, {t['ci'][1]:.1f}] & {t['p']:.3f}")
T.append(table(
    "Welch's $t$-tests for the contrasts discussed in the text (game = unit; $n$ = 3 or 5 per side). Difference = second "
    "setup minus first; 95\\% confidence interval; two-sided $p$, not corrected for multiple comparisons. Contrasts where "
    "both setups have zero variance (e.g., GPT-4o's offers, always 0) are omitted.",
    "tab:tests", "lllccc",
    "\\textbf{Model} & \\textbf{Metric} & \\textbf{Contrast} & \\textbf{Diff.} & \\textbf{95\\% CI} & \\textbf{$p$}",
    rows, size="\\tiny", sep="2pt"))

# robustness
rows = []
for grp, models in (("samegen", ["gpt5"]), ("open", OPEN), ("frontier", ["claude", "qwen"])):
    for m in models:
        cells = []
        for s in ("chat_only", "elected", "hidden", "anonymous"):
            c = x["promises"][grp][s][m]
            if not c.get("logged_explicit"):
                cells.append("---")
                continue
            cells.append(f"{c.get('logged_broken', 0)}/{c['logged_explicit']} \\,|\\, {c.get('broken', 0)}/{c.get('explicit', 0)} "
                         f"\\,|\\, {c.get('broken_uncond', 0)}/{c.get('explicit_uncond', 0)}")
        rows.append(NAME[m] + " & " + " & ".join(cells))
T.append(table(
    "Robustness of promise-breaking. Each cell: broken / explicit promises as reported (GPT-4o-mini extraction with "
    "regular-expression fallback) $|$ regular expression only $|$ regular expression only, excluding conditional promises "
    "(``if'', ``as long as'', ``provided'', ``unless'').",
    "tab:robust", "lcccc",
    "\\textbf{Model} & \\textbf{Chat only} & \\textbf{Transparent} & \\textbf{Hidden} & \\textbf{Anonymous}",
    rows, size="\\tiny", sep="2pt"))

# This file only defines macros; appendix.tex \input's it once and places each table in its section:
# \hgParseTable (A), \hgResultTables (C), \hgTestsTable (D), \hgRobustTable (E).
MACROS = {"tab:parse": "hgParseTable", "tab:tests": "hgTestsTable", "tab:robust": "hgRobustTable"}
print("% Generated by hg-rerun/analysis/appendix_tables.py from paper_numbers.json and extras.json. Do not edit by hand.")
print("% Defines \\hgParseTable, \\hgResultTables, \\hgTestsTable and \\hgRobustTable (placed in appendix.tex).\n")
rest = []
for t in T:
    name = next((m for lab, m in MACROS.items() if f"\\label{{{lab}}}" in t), None)
    if name:
        print(f"\\newcommand{{\\{name}}}{{%\n{t.rstrip()}\n}}\n")
    else:
        rest.append(t.rstrip())
print("\\newcommand{\\hgResultTables}{%\n" + "\n\n".join(rest) + "\n}")
