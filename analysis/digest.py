"""Readable digest of paper_numbers.json (for writing; every value comes from the JSON)."""
import json
import sys

d = json.load(open(sys.argv[1]))


def f(x, nd=1):
    if not x:
        return "  -  "
    return f"{x['mean']:.{nd}f}±{x['sd']:.{nd}f}"


def block(title, group, setups, keys):
    print(f"\n=== {title}")
    models = list(next(iter(d[group].values())).keys())
    for s in setups:
        print(f" -- {s}")
        for m in models:
            v = d[group][s][m]
            if not v:
                print(f"   {m:22s} (no data)")
                continue
            parts = []
            for k in keys:
                if k in ("broken_explicit",):
                    parts.append(f"broken {v['broken_explicit']}/{v['explicit']}")
                elif k == "flag":
                    parts.append(f"codeflag {v['flag_code_rule']}/{v['promises_any']}")
                elif k == "turn":
                    parts.append(f"turn {v['turnovers']}/{v['contested']}")
                elif k == "parse":
                    parts.append(f"parse {100 * v['parse_ok'] / v['replies']:.1f}%")
                elif k == "n":
                    parts.append(f"n={v['coop']['n']}")
                elif v.get(k) is not None:
                    parts.append(f"{k} {f(v[k], 0 if k == 'welfare' else 1)}")
            print(f"   {m:22s} " + " | ".join(parts))


std = ["n", "coop", "contrib"]
block("FRONTIER cooperation/contribution", "frontier", ["baseline", "chat_only", "fixed", "elected", "rotating"], std + ["parse"])
block("FRONTIER pay", "frontier", ["elected", "salary", "costly"], ["n", "deals_strict", "deals_keyword", "spend_applied", "welfare", "turn"])
block("FRONTIER visibility", "frontier", ["elected", "hidden", "anonymous"], ["n", "broken_explicit", "flag", "contrib"])
block("FRONTIER manager type welfare/spend/elections", "frontier", ["fixed", "elected", "rotating"], ["n", "welfare", "spend_applied", "turn"])
block("SAMEGEN", "samegen", ["baseline", "fixed", "elected", "rotating", "salary", "costly", "hidden", "anonymous"],
      ["n", "coop", "contrib", "deals_strict", "broken_explicit", "spend_applied", "welfare", "turn", "parse"])
block("OPEN", "open", ["baseline", "chat_only", "public_chat", "private_chat", "fixed", "elected", "rotating", "salary", "costly",
                       "hidden", "anonymous", "belief_unknown", "belief_human", "belief_mixed"],
      ["n", "coop", "contrib", "deals_strict", "broken_explicit", "spend_applied", "turn", "parse"])
print("\n=== MIXED (5-of-6, elected)")
for k, v in d["mixed"].items():
    print(f"   {k:16s} n={v['coop']['n']} coop {f(v['coop'])} contrib {f(v['contrib'])} turn {v['turnovers']}/{v['contested']} "
          f"welfare {f(v['welfare'], 0)} winners {v.get('election_winners')}")
print("\n=== CROSS-RULE")
for k, v in d["crossrule"].items():
    print(f"   {k:16s} n={v['coop']['n']} worker coop {f(v['worker_coop'])} worker contrib {f(v['worker_contrib'])} "
          f"spend {f(v['spend_applied'])} welfare {f(v['welfare'], 0)}")
print("\n=== META", d["meta"])
for grp in ("frontier", "samegen"):
    calls = set()
    for s in d[grp].values():
        for v in s.values():
            if v:
                calls.update(v["models_called"])
    print(grp, "model IDs called:", sorted(calls))
