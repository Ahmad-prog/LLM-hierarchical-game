#!/usr/bin/env python3
"""Rebuild every number, table and figure of the paper from the game logs in results/ and the labels in annotation/.

  python reproduce.py            # write paper_data/ and figures/
  python reproduce.py --check    # rebuild in a temporary folder and compare with paper_data/

No API key or GPU is needed: this only reads results/{api,oss}/*.json and annotation/. The figures are typeset with
LaTeX when a LaTeX installation is on PATH (as in the paper); without one they are drawn with matplotlib's own font.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ANN = ROOT / "annotation"
STEPS = 15


def run(args, out_file=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True)
    if res.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, args))}\n{res.stderr.decode(errors='replace')}")
    if out_file is not None:
        Path(out_file).write_bytes(res.stdout)


def step(k, text):
    print(f"{k:2d}/{STEPS} {text}", flush=True)


def build(data_dir: Path, fig_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    numbers, extras = data_dir / "paper_numbers.json", data_dir / "extras.json"
    step(1, "per-setup metrics                          -> paper_numbers.json")
    run(["analysis/paper_tables.py", RESULTS], numbers)
    step(2, "spending, elections, secondary tests        -> extras.json")
    run(["analysis/revision_extras.py", RESULTS, extras], data_dir / "extras.txt")
    step(3, "intentions, offers, re-election, ANOVA      -> revision2.json")
    run(["analysis/revision2.py", RESULTS, data_dir / "revision2.json"], data_dir / "revision2.txt")
    step(4, "mechanism controls (batch 13)               -> round2.json")
    run(["analysis/round2.py", RESULTS, data_dir / "round2.json"], data_dir / "round2.txt")
    step(5, "primary tests and all controls, LLM judge   -> round3.json (takes a few minutes)")
    run(["analysis/round3.py", RESULTS, data_dir / "round3.json", ANN / "offers_human_labels.csv",
         ANN / "offers_llm_judge.jsonl"], data_dir / "round3.txt")
    step(6, "human labels against the fixed rules        -> scores_human.json")
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("offers_key.csv", "intentions_key.csv"):
            shutil.copy(ANN / f, tmp)
        run(["tools/score_manual_labels.py", tmp, ANN / "offers_human_labels.csv", ANN / "intentions_human_labels.csv"])
        shutil.copy(Path(tmp) / "scores.json", data_dir / "scores_human.json")
    step(7, "readable digest                             -> digest.txt")
    run(["analysis/digest.py", numbers], data_dir / "digest.txt")
    step(8, "paper tables (LaTeX)                        -> appendix_tables.tex")
    run(["analysis/appendix_tables.py", numbers, extras, data_dir / "revision2.json"], data_dir / "appendix_tables.tex")
    step(9, "paper tables for the controls (LaTeX)       -> round3_tables.tex")
    run(["analysis/round3_tables.py", data_dir / "round3.json"], data_dir / "round3_tables.tex")
    step(10, "fair-ballot equivalence, batch 16, confirmatory batch -> round5.md")
    run(["analysis/round5.py", RESULTS, ROOT / "results_confirm", ANN / "confirm_llm_judge.jsonl", data_dir / "round5.md"])
    step(11, "second annotator against the fixed rules   -> scores_annotator2.json")
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("offers_key.csv", "intentions_key.csv"):
            shutil.copy(ANN / f, tmp)
        run(["tools/score_manual_labels.py", tmp, ANN / "offers_human_labels_annotator2.csv", ANN / "intentions_human_labels_annotator2.csv"])
        shutil.copy(Path(tmp) / "scores.json", data_dir / "scores_annotator2.json")
    step(12, "agreement between annotators, LLM judge under the neutral prompt -> round6_annotation.json")
    run(["tools/score_round6.py", ANN, ANN / "offers_human_labels_annotator2.csv", ANN / "intentions_human_labels_annotator2.csv",
         data_dir / "scores_annotator2.json", ANN / "neutral_offers_human_labels.csv", ANN / "neutral_offers_key.csv",
         ANN / "offers_llm_judge.jsonl", data_dir / "round6_annotation.json"])
    step(13, "first-election offers, vote-level accountability -> round7.json")
    run(["analysis/round7.py", RESULTS, ROOT / "results_confirm", ANN / "offers_llm_judge.jsonl", ANN / "confirm_llm_judge.jsonl", data_dir / "round7.json"])
    step(14, "clean-label prompt, GPT-5 endgame, placebo, fair-ballot managers -> round8.json")
    run(["analysis/round8.py", RESULTS, ANN / "offers_llm_judge.jsonl", data_dir / "round8.json"])
    step(15, "figures (PDF)                              -> figures/")
    run(["analysis/make_figures.py", numbers, fig_dir])


CHECKED = ["paper_numbers.json", "extras.json", "extras.txt", "revision2.json", "revision2.txt", "round2.json",
           "round2.txt", "round3.json", "round3.txt", "scores_human.json", "digest.txt", "appendix_tables.tex",
           "round3_tables.tex", "round5.md", "scores_annotator2.json", "round6_annotation.json", "round7.json", "round8.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in a temp folder and compare with paper_data/")
    a = ap.parse_args()
    n_api, n_oss = len(list((RESULTS / "api").glob("*.json"))), len(list((RESULTS / "oss").glob("*.json")))
    print(f"game logs: {n_api} API games + {n_oss} self-hosted games = {n_api + n_oss}")
    if not a.check:
        build(ROOT / "paper_data", ROOT / "figures")
        print("done: paper_data/ and figures/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp) / "paper_data", Path(tmp) / "figures")
        same = [n for n in CHECKED if filecmp.cmp(Path(tmp) / "paper_data" / n, ROOT / "paper_data" / n, shallow=False)]
        for n in CHECKED:
            print(f"  {'identical' if n in same else 'DIFFERENT'}  paper_data/{n}")
        if len(same) != len(CHECKED):
            sys.exit(1)
        print("all numbers reproduce exactly")


if __name__ == "__main__":
    main()
