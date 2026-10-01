"""Figures for the paper from paper_numbers.json (mean ± sd over games).

  python analysis/make_figures.py <paper_numbers.json> <out_dir>

Drawn at their final size for the AAAI two-column layout (text width 7 in, column width 3.3 in),
so the paper includes them at 100% and the lettering matches the caption size.
Palette: the validated categorical slots 1-5, fixed order; legend on every multi-series chart;
all values also appear in tables in the paper.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
TEXT_W, COL_W = 7.0, 3.3
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e6e5df"
FRONT = [("gpt4o", "GPT-4o"), ("claude", "Claude\nSonnet 4.5"), ("gemini", "Gemini\n2.5 Flash"),
         ("deepseek", "DeepSeek\nV3"), ("grok", "Grok 4.3"), ("qwen", "Qwen Plus")]
OPEN = [("gpt-oss-120b", "gpt-oss\n120B"), ("nemotron-3-super-120b", "Nemotron 3\nSuper 120B"),
        ("ling-3.0-flash", "Ling 3.0\nflash"), ("qwen3.8-27b", "Qwen3.8\n27B"), ("gemma-4-31b", "Gemma 4\n31B")]
OPEN_SHORT = [(k, v) for k, v in zip([m for m, _ in OPEN], ["gpt-oss", "Nemotron", "Ling", "Qwen3.8", "Gemma"])]

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


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
        ax.bar(xs, ys, width * 0.9, color=color, label=name, edgecolor="white", linewidth=0.5, zorder=2)
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
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# Figure 1: contribution by institution, main API models (with the chat-only control)
conds = [("baseline", "No chat, no manager"), ("chat_only", "Chat only"), ("fixed", "Chat + fixed manager"),
         ("elected", "Chat + elected manager"), ("rotating", "Chat + rotating manager")]
fig, ax = plt.subplots(figsize=(TEXT_W - 0.35, 1.8))
grouped(ax, FRONT, conds, lambda g, s: (d["frontier"][s][g[0]] or {}).get("contrib"), "Contribution (of 20)", (0, 22))
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
save(fig, "rr_frontier_contribution.pdf")

# Figure 2: self-hosted models, cooperation by institution
fig, ax = plt.subplots(figsize=(TEXT_W - 0.35, 1.8))
grouped(ax, OPEN, conds, lambda g, s: (d["open"][s][g[0]] or {}).get("coop"), "Cooperation (%)", (0, 105))
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
save(fig, "rr_open_cooperation.pdf")

# Figure 3: vote-contingent offers (a vote tied to a material benefit) by manager pay
pay = [("elected", "No salary"), ("salary", "Salary +5"), ("costly", "Cost −3")]
groups = [("gpt4o", "GPT-4o"), ("claude", "Claude"), ("gemini", "Gemini"), ("deepseek", "DeepSeek\nV3"), ("grok", "Grok"),
          ("qwen", "Qwen\nPlus"), ("gpt5", "GPT-5"), ("deepseek31", "DeepSeek\nV3.1"), ("qwen3.8-27b", "Qwen3.8"),
          ("gpt-oss-120b", "gpt-oss")]


def deal_get(g, s):
    for grp in ("frontier", "samegen", "open"):
        v = d[grp][s].get(g[0])
        if v:
            return v["deals_strict"]
    return None


fig, ax = plt.subplots(figsize=(TEXT_W + 0.85, 1.9))
grouped(ax, groups, pay, deal_get, "Offers per\n100 agent-rounds", (0, 31))
for x in (5.5, 7.5):
    ax.axvline(x, color=MUTED, linewidth=0.6, linestyle=":")
for x, label in ((2.5, "Main API models"), (6.5, "Newer API models"), (8.5, "Self-hosted")):
    ax.text(x, 28.5, label, ha="center", color=MUTED, fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
save(fig, "rr_deals_by_pay.pdf")

# Appendix figure: broken stated intentions by sanction visibility, self-hosted models (one column)
vis = [("chat_only", "Chat only"), ("elected", "Transparent"), ("hidden", "Hidden"), ("anonymous", "Anonymous")]


def broken_rate(g, s):
    v = d["open"][s][g[0]]
    if not v or not v["explicit"]:
        return None
    return {"mean": 100 * v["broken_explicit"] / v["explicit"], "sd": 0.0}


fig, ax = plt.subplots(figsize=(COL_W, 1.9))
grouped(ax, OPEN_SHORT, vis, broken_rate, "Broken (%)", (0, 55))
ax.tick_params(axis="x", labelsize=7.5)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8, fontsize=7.5)
save(fig, "rr_open_broken_promises.pdf")
