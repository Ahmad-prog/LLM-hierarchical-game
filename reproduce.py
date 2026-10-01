#!/usr/bin/env python3
"""Rebuild every number, table and figure of the paper from the game logs in results/.

  python reproduce.py            # write paper_data/ and figures/
  python reproduce.py --check    # rebuild in a temporary folder and compare with paper_data/

No API key or GPU is needed: this only reads results/api/*.json and results/oss/*.json.
"""
import argparse
import filecmp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run(args, out_file=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True)
    if res.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, args))}\n{res.stderr.decode(errors='replace')}")
    if out_file is not None:
        Path(out_file).write_bytes(res.stdout)


def build(data_dir: Path, fig_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    numbers, extras = data_dir / "paper_numbers.json", data_dir / "extras.json"
    print("1/6 per-setup metrics            -> paper_numbers.json")
    run(["analysis/paper_tables.py", RESULTS], numbers)
    print("2/6 spending, elections, tests   -> extras.json")
    run(["analysis/revision_extras.py", RESULTS, extras], data_dir / "extras.txt")
    print("3/6 intentions, offers, re-election, model vs institution -> revision2.json")
    run(["analysis/revision2.py", RESULTS, data_dir / "revision2.json"], data_dir / "revision2.txt")
    print("4/6 readable digest              -> digest.txt")
    run(["analysis/digest.py", numbers], data_dir / "digest.txt")
    print("5/6 paper tables (LaTeX)         -> appendix_tables.tex")
    run(["analysis/appendix_tables.py", numbers, extras, data_dir / "revision2.json"], data_dir / "appendix_tables.tex")
    print("6/6 figures (PDF)                -> figures/")
    run(["analysis/make_figures.py", numbers, fig_dir])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in a temp folder and compare with paper_data/")
    a = ap.parse_args()
    n_api, n_oss = len(list((RESULTS / "api").glob("*.json"))), len(list((RESULTS / "oss").glob("*.json")))
    print(f"game logs: {n_api} API games + {n_oss} open-model games = {n_api + n_oss}")
    if not a.check:
        build(ROOT / "paper_data", ROOT / "figures")
        print("done: paper_data/ and figures/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp) / "paper_data", Path(tmp) / "figures")
        names = ["paper_numbers.json", "extras.json", "extras.txt", "revision2.json", "revision2.txt", "digest.txt", "appendix_tables.tex"]
        same = [n for n in names if filecmp.cmp(Path(tmp) / "paper_data" / n, ROOT / "paper_data" / n, shallow=False)]
        for n in names:
            print(f"  {'identical' if n in same else 'DIFFERENT'}  paper_data/{n}")
        if len(same) != len(names):
            sys.exit(1)
        print("all numbers reproduce exactly")


if __name__ == "__main__":
    main()
