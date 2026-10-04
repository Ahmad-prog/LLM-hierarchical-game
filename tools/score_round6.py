"""Second annotator and LLM-judge check on neutral-prompt messages.

  python tools/score_round6.py <annotation_dir> <annot2_offers.csv> <annot2_intentions.csv> <scores_annot2.json>
         <neutral_human.csv> <neutral_key.csv> <offers_llm_judge.jsonl> <out.json>
"""
import csv, json, os, re, sys
from collections import Counter

ANN, A2O, A2I, SC2, NH, NK, JUD, OUT = sys.argv[1:9]
rd = lambda p: {r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
low = lambda x: (x or "").strip().lower()


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


o1, o2 = rd(os.path.join(ANN, "offers_human_labels.csv")), rd(A2O)
i1, i2 = rd(os.path.join(ANN, "intentions_human_labels.csv")), rd(A2I)
res = {"inter_annotator": {
    "offers_is_offer": kappa([(low(o1[k]["is_offer"]), low(o2[k]["is_offer"])) for k in o1]),
    "offers_category": kappa([(low(o1[k]["category"]), low(o2[k]["category"])) for k in o1]),
    "intentions_statement_type": kappa([(low(i1[k]["statement_type"]), low(i2[k]["statement_type"])) for k in i1]),
    "intentions_commitment_vs_rest": kappa([(low(i1[k]["statement_type"]) == "commitment", low(i2[k]["statement_type"]) == "commitment") for k in i1]),
}}


def broken(r, strict):
    t, v, g = low(r["statement_type"]), num(r["stated_amount"]), num(r["speaker_gave"])
    if v is None or g is None or t not in ("commitment", "conditional", "proposal"):
        return False
    if strict and not (t == "commitment" or (t == "conditional" and low(r["condition_met"]) == "yes")):
        return False
    return g < v - 5


both = [k for k in i1 if num(i1[k].get("speaker_gave")) is not None]
res["inter_annotator"]["broken_any_statement"] = kappa([(broken(i1[k], False), broken(i2[k], False)) for k in both])
res["inter_annotator"]["broken_commitment_strict"] = kappa([(broken(i1[k], True), broken(i2[k], True)) for k in both])
amt = [(num(i1[k]["stated_amount"]), num(i2[k]["stated_amount"])) for k in i1 if num(i1[k]["stated_amount"]) is not None and num(i2[k]["stated_amount"]) is not None]
res["inter_annotator"]["stated_amount_same"] = {"n": len(amt), "share": round(sum(a == b for a, b in amt) / len(amt), 3)}
res["rules_vs_annotator2"] = json.load(open(SC2, encoding="utf-8"))

# ---------------- LLM judge on neutral-prompt messages
h, key = rd(NH), rd(NK)
pairs = [(low(h[k]["is_offer"]) == "yes", low(key[k]["judge_offer"]) == "yes", key[k]) for k in h]
tp = sum(a and b for a, b, _ in pairs); fp = sum(b and not a for a, b, _ in pairs); fn = sum(a and not b for a, b, _ in pairs)
pop = Counter()
for line in open(JUD, encoding="utf-8"):
    d = json.loads(line)
    m = re.match(r"batch14_neutral_(elected|salary)_(.+?)__t", d["key"])
    if m:
        pop[(m.group(2), low(d["is_offer"]) == "yes")] += 1
# population-weighted precision/recall: each (model, judge label) stratum weighted by its population size
wtp = wfp = wfn = 0.0
by = {}
for model in sorted({k["model"] for _, _, k in pairs}):
    for jy in (True, False):
        s = [(a, b) for a, b, k in pairs if k["model"] == model and b == jy]
        if not s:
            continue
        w = pop[(model, jy)] / len(s)
        hy = sum(a for a, _ in s)
        if jy:
            wtp += w * hy; wfp += w * (len(s) - hy)
        else:
            wfn += w * hy
    s = [(a, b) for a, b, k in pairs if k["model"] == model]
    by[model] = {"n": len(s), "judge_yes": sum(b for _, b in s), "human_yes": sum(a for a, _ in s), "both_yes": sum(a and b for a, b in s)}
by_setup = {}
for setup in ("elected", "salary"):
    for model in sorted(by):
        s = [(a, b) for a, b, k in pairs if k["model"] == model and k["setup"] == setup]
        if s:
            by_setup[f"{model}|{setup}"] = {"n": len(s), "judge_yes": sum(b for _, b in s), "human_yes": sum(a for a, _ in s)}
res["judge_neutral"] = {"n": len(pairs), "judge_yes": tp + fp, "human_yes": tp + fn,
                        "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                        "kappa": kappa([(a, b) for a, b, _ in pairs])["kappa"],
                        "population_weighted": {"precision": round(wtp / (wtp + wfp), 3), "recall": round(wtp / (wtp + wfn), 3) if wtp + wfn else None},
                        "by_model": by, "by_model_setup": by_setup}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
