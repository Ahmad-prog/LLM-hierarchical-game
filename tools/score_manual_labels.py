"""Score the human labels from manual_work/ against the automatic rules (and against a second labeller).

  python tools/score_manual_labels.py <key_dir> <offers_A.csv> <intentions_A.csv> [<offers_B.csv> <intentions_B.csv>]

A = main labeller (all rows); B = optional second labeller (overlap rows). Prints precision, recall and
Cohen's kappa, and writes key_dir/scores.json.
"""
import csv
import json
import os
import sys
from collections import Counter


def read(path):
    return {r["id"]: {k: (v or "").strip().lower() for k, v in r.items()} for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}


def prf(pairs):
    """pairs of (rule_bool, human_bool)"""
    tp = sum(r and h for r, h in pairs)
    fp = sum(r and not h for r, h in pairs)
    fn = sum(h and not r for r, h in pairs)
    return {"n": len(pairs), "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


key_dir = sys.argv[1]
ko, ki = read(os.path.join(key_dir, "offers_key.csv")), read(os.path.join(key_dir, "intentions_key.csv"))
oa, ia = read(sys.argv[2]), read(sys.argv[3])
out = {}

# --- offers vs rule
rows = [(i, ko[i], oa[i]) for i in ko if i in oa and oa[i]["is_offer"] in ("yes", "no")]
out["offers_rule_vs_human"] = prf([(k["rule_offer"] == "true", h["is_offer"] == "yes") for _, k, h in rows])
cat_rows = [(k["rule_category"], h["category"]) for _, k, h in rows if k["rule_offer"] == "true"]
out["offer_category_rule_vs_human"] = kappa([(("none" if a == "other" else a), b) for a, b in cat_rows])
out["offer_category_human_shares"] = dict(Counter(h["category"] for _, k, h in rows if h["is_offer"] == "yes"))
out["impossible_rule_vs_human"] = prf([(k["rule_impossible"] == "true", h["impossible"] == "yes") for _, k, h in rows if h["impossible"]])

# --- stated intentions vs rule
rows = [(i, ki[i], ia[i]) for i in ki if i in ia and ia[i]["statement_type"]]
stated = lambda h: h["statement_type"] in ("commitment", "conditional", "proposal")
out["stated_rule_vs_human"] = prf([(k["rule_explicit"] == "true", stated(h)) for _, k, h in rows])
both = [(k, h) for _, k, h in rows if k["rule_explicit"] == "true" and stated(h) and num(h["stated_amount"]) is not None]
out["amount_match"] = {"n": len(both), "same_amount": sum(num(k["rule_value"]) == num(h["stated_amount"]) for k, h in both)}
out["first_person_rule_vs_human_commitment"] = prf([(k["rule_first_person"] == "true", h["statement_type"] == "commitment")
                                                    for _, k, h in rows if k["rule_explicit"] == "true"])


def human_broken(h, strict):
    """strict: only commitments, and conditionals whose condition was met, can be broken."""
    a = num(h["stated_amount"])
    if a is None or not stated(h):
        return False
    if strict and (h["statement_type"] == "proposal" or (h["statement_type"] == "conditional" and h["condition_met"] != "yes")):
        return False
    return num(h["speaker_gave"]) < a - 5


for strict in (False, True):
    out[f"broken_rule_vs_human_{'strict' if strict else 'any_statement'}"] = prf(
        [(k["rule_broken"] == "true", human_broken(h, strict)) for _, k, h in rows if k["rule_explicit"] == "true"])
out["statement_type_human_shares"] = dict(Counter(h["statement_type"] for _, _, h in rows))

# --- second labeller
if len(sys.argv) > 5:
    ob, ib = read(sys.argv[4]), read(sys.argv[5])
    out["interrater_is_offer"] = kappa([(oa[i]["is_offer"], ob[i]["is_offer"]) for i in ob if i in oa])
    out["interrater_offer_category"] = kappa([(oa[i]["category"], ob[i]["category"]) for i in ob if i in oa])
    out["interrater_statement_type"] = kappa([(ia[i]["statement_type"], ib[i]["statement_type"]) for i in ib if i in ia])
    out["interrater_condition_met"] = kappa([(ia[i]["condition_met"], ib[i]["condition_met"]) for i in ib if i in ia])

json.dump(out, open(os.path.join(key_dir, "scores.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
