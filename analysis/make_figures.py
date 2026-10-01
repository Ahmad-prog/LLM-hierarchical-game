"""Figures for the revised paper from paper_numbers.json (mean ± sd over trials).

  python analysis/make_figures.py <paper_numbers.json> <out_dir>

Palette: the validated categorical slots 1-5 (light), fixed order; legend on every
multi-series chart; all values also appear in tables in the paper (contrast relief).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e6e5df"
FRONT = [("gpt4o", "GPT-4o"), ("claude", "Claude\nSonnet 4.5"), ("gemini", "Gemini\n2.5 Flash"),
         ("deepseek", "DeepSeek\nV3"), ("grok", "Grok 4.3"), ("qwen", "Qwen Plus")]
OPEN = [("gpt-oss-120b", "gpt-oss\n120B"), ("nemotron-3-super-120b", "Nemotron 3\nSuper 120B"),
        ("ling-3.0-flash", "Ling 3.0\nflash"), ("qwen3.8-27b", "Qwen3.8\n27B"), ("gemma-4-31b", "Gemma 4\n31B")]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


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
        ax.bar(xs, ys, width * 0.9, color=color, label=name, edgecolor="white", linewidth=0.6, zorder=2)
        if any(es):
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# Figure: contribution by institution, frontier models (with the chat-only control)
conds = [("baseline", "No chat, no manager"), ("chat_only", "Chat only"), ("fixed", "Chat + fixed manager"),
         ("elected", "Chat + elected manager"), ("rotating", "Chat + rotating manager")]
fig, ax = plt.subplots(figsize=(5.5, 1.95))
grouped(ax, FRONT, conds, lambda g, s: (d["frontier"][s][g[0]] or {}).get("contrib"), "Mean contribution (of 20)", (0, 23))
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.33), fontsize=7)
save(fig, "rr_frontier_contribution.pdf")

# Figure: open models, cooperation by institution
fig, ax = plt.subplots(figsize=(5.5, 1.95))
grouped(ax, OPEN, conds, lambda g, s: (d["open"][s][g[0]] or {}).get("coop"), "Cooperation rate (%)", (0, 105))
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.33), fontsize=7)
save(fig, "rr_open_cooperation.pdf")

# Figure: vote-buying offers (a vote tied to a material benefit) by manager pay
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


fig, ax = plt.subplots(figsize=(6.4, 2.0))
grouped(ax, groups, pay, deal_get, "Vote-buying offers\nper 100 agent-rounds", (0, 31))
ax.axvline(5.5, color=MUTED, linewidth=0.6, linestyle=":")
ax.axvline(7.5, color=MUTED, linewidth=0.6, linestyle=":")
ax.text(2.5, 28.5, "Main frontier set", ha="center", color=MUTED, fontsize=7)
ax.text(6.5, 28.5, "Same generation", ha="center", color=MUTED, fontsize=7)
ax.text(8.5, 28.5, "Open models", ha="center", color=MUTED, fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7)
save(fig, "rr_deals_by_pay.pdf")

# Figure: broken explicit promises by visibility, open models (frontier models break almost none)
vis = [("chat_only", "Chat only (no manager)"), ("elected", "Transparent"), ("hidden", "Hidden"), ("anonymous", "Anonymous")]


def broken_rate(g, s):
    v = d["open"][s][g[0]]
    if not v or not v["explicit"]:
        return None
    return {"mean": 100 * v["broken_explicit"] / v["explicit"], "sd": 0.0}


fig, ax = plt.subplots(figsize=(5.5, 2.1))
grouped(ax, OPEN, vis, broken_rate, "Explicit promises broken (%)", (0, 60))
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
save(fig, "rr_open_broken_promises.pdf")
