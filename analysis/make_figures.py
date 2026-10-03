"""Figures for the paper from paper_numbers.json (mean ± sd over games).

  python analysis/make_figures.py <paper_numbers.json> <out_dir>

AAAI author-kit rules these figures follow:
- drawn at their final size (text width 7 in, column width 3.3 in) and included at 100%, so nothing is rescaled;
- all lettering at least 9 pt, in Helvetica, typeset by LaTeX (text.usetex), so every font in the figures is an
  embedded Type 1 font (no Type 3, no CID/Identity-H fonts); needs a LaTeX with helvet, sfmath and type1cm on PATH;
- readable without colour: every series also has its own hatch pattern, and a legend on every multi-series chart.
All values also appear in tables in the paper.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
# The paper's figures are typeset by LaTeX (Helvetica, Type 1 fonts). Without a LaTeX installation the same figures
# are drawn with matplotlib's own font, so the numbers can still be checked.
USETEX = shutil.which("latex") is not None
if not USETEX:
    print("latex not found: drawing the figures without LaTeX (fonts differ from the paper)", file=sys.stderr)
OUT = sys.argv[2]
TEXT_W, COL_W = 7.0, 3.3
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\"]
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e6e5df"
FRONT = [("gpt4o", "GPT-4o"), ("claude", "Claude\nSonnet 4.5"), ("gemini", "Gemini\n2.5 Flash"),
         ("deepseek", "DeepSeek\nV3"), ("grok", "Grok 4.3"), ("qwen", "Qwen Plus")]
OPEN = [("gpt-oss-120b", "gpt-oss\n120B"), ("nemotron-3-super-120b", "Nemotron 3\nSuper 120B"),
        ("ling-3.0-flash", "Ling 3.0\nflash"), ("qwen3.8-27b", "Qwen3.8\n27B"), ("gemma-4-31b", "Gemma 4\n31B")]
OPEN_SHORT = [(k, v) for k, v in zip([m for m, _ in OPEN], ["gpt-oss", "Nemotron", "Ling", "Qwen3.8", "Gemma"])]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "text.usetex": USETEX, "font.family": "sans-serif",
                     "text.latex.preamble": r"\usepackage[scaled=1]{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{sfmath}",
                     "hatch.linewidth": 0.5, "hatch.color": INK, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def grouped(ax, groups, series, getter, ylabel, ylim):
    n = len(series)
    width = 0.8 / n
    for j, (name, color) in enumerate(zip([s[1] for s in series], SLOTS)):
        xs, ys, es = [], [], []
        for i, g in enumerate(groups):
            v = getter(g, series[j][0])
            if v is None:
                continue
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(v["mean"])
            es.append(v["sd"])
        ax.bar(xs, ys, width * 0.9, color=color, label=name, hatch=HATCH[j], edgecolor=INK, linewidth=0.5, zorder=2)
        if any(es):
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def save(fig, name):
    """exactly the figure size (no tight bounding box), so the paper can include it at 100%"""
    fig.savefig(os.path.join(OUT, name), metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    print("wrote", name)


# Figure 1: contribution by institution, main API models (with the chat-only control)
conds = [("baseline", "No chat, no manager"), ("chat_only", "Chat only"), ("fixed", "Chat + fixed manager"),
         ("elected", "Chat + elected manager"), ("rotating", "Chat + rotating manager")]
fig, ax = plt.subplots(figsize=(TEXT_W, 2.3), layout="constrained")
grouped(ax, FRONT, conds, lambda g, s: (d["frontier"][s][g[0]] or {}).get("contrib"), "Contribution (of 20)", (0, 22))
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, handlelength=1.4, columnspacing=1.2,
           handletextpad=0.4)
save(fig, "rr_frontier_contribution.pdf")

# Figure 2: self-hosted models, cooperation by institution
fig, ax = plt.subplots(figsize=(TEXT_W, 2.3), layout="constrained")
grouped(ax, OPEN, conds, lambda g, s: (d["open"][s][g[0]] or {}).get("coop"), "Cooperation (\\%)" if USETEX else "Cooperation (%)", (0, 105))
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, handlelength=1.4, columnspacing=1.2,
           handletextpad=0.4)
save(fig, "rr_open_cooperation.pdf")

# Figure 3: vote-contingent offers (a vote tied to a material benefit) by manager pay
pay = [("elected", "No salary"), ("salary", "Salary +5"), ("costly", "Cost $-$3" if USETEX else "Cost −3")]
groups = [("gpt4o", "GPT-4o"), ("claude", "Claude"), ("gemini", "Gemini"), ("deepseek", "DeepSeek\nV3"), ("grok", "Grok"),
          ("qwen", "Qwen\nPlus"), ("gpt5", "GPT-5"), ("deepseek31", "DeepSeek\nV3.1"), ("qwen3.8-27b", "Qwen3.8"),
          ("gpt-oss-120b", "gpt-oss")]


def deal_get(g, s):
    for grp in ("frontier", "samegen", "open"):
        v = d[grp][s].get(g[0])
        if v:
            return v["deals_strict"]
    return None


fig, ax = plt.subplots(figsize=(TEXT_W, 2.4), layout="constrained")
grouped(ax, groups, pay, deal_get, "Offers per\n100 agent-rounds", (0, 33))
for x in (5.5, 7.5):
    ax.axvline(x, color=MUTED, linewidth=0.6, linestyle=":")
for x, label in ((2.5, "Main API models"), (6.5, "Newer API"), (8.5, "Self-hosted")):
    ax.text(x, 30.5, label, ha="center", color=INK, fontsize=9)
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, handlelength=1.4)
save(fig, "rr_deals_by_pay.pdf")

# Appendix figure: broken stated intentions by sanction visibility, self-hosted models (one column)
vis = [("chat_only", "Chat only"), ("elected", "Transparent"), ("hidden", "Hidden"), ("anonymous", "Anonymous")]


def broken_rate(g, s):
    v = d["open"][s][g[0]]
    if not v or not v["explicit"]:
        return None
    return {"mean": 100 * v["broken_explicit"] / v["explicit"], "sd": 0.0}


fig, ax = plt.subplots(figsize=(COL_W, 2.4), layout="constrained")
grouped(ax, OPEN_SHORT, vis, broken_rate, "Broken (\\%)" if USETEX else "Broken (%)", (0, 55))
ax.tick_params(axis="x", labelrotation=0)
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, handlelength=1.4, columnspacing=1.0)
save(fig, "rr_open_broken_promises.pdf")
