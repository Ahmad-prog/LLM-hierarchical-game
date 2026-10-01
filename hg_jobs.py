#!/usr/bin/env python3
"""
Parallel job runner for the 2026-09 re-run of the Hierarchical Game.

One job = one 20-round game (one trial of one config). Each finished game is written to
  <out>/<track>/<config>__t<k>.json
so the runner is resumable: finished games are skipped on restart.

Tracks
  api   frontier models through OpenRouter, in priority order, with a budget guard:
        no new game starts once our spend reaches --budget-usd or the account's remaining
        credit falls below --min-remaining-usd (games already running finish).
  oss   one open model served locally by vLLM (ModelType.LOCAL, see game/settings.py);
        same-model experiments only.

Progress goes to <out>/status_<track>.json and <out>/STATUS.md (rewritten after every game).

Usage
  python hg_jobs.py --track api --workers 10
  python hg_jobs.py --track oss --tag gpt-oss-120b --workers 16
  python hg_jobs.py --track api --dry-run        # list the jobs only
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from config.enums import ModelType
from experiments.conditions import generate_all_conditions

MODELS = ["gpt4o", "claude", "deepseek", "gemini", "grok", "qwen"]

# API track: unique setups in priority order. The elected-manager setup is identical to
# Manager-Pay no-salary and Punish-Visibility transparent, so it is their matched control.
API_GROUPS = [
    ("1 Baseline + Manager type", [f"batch1_baseline_{m}" for m in MODELS]
     + [f"batch3_mgr_{t}_{m}" for t in ("fixed", "elected", "rotating") for m in MODELS]),
    ("2 Manager pay", [f"batch8_mgr_{s}_{m}" for s in ("salary", "costly") for m in MODELS]),
    ("3 Punishment visibility", [f"batch9_punish_{v}_{m}" for v in ("hidden", "anonymous") for m in MODELS]),
    ("5 Mixed groups (elections)", [f"batch4_hetero_all_drop_{m}" for m in MODELS]),
    # the manager runs include full chat, so the manager effect needs a chat-only control
    ("6 Chat, no manager (control)", [f"batch2_comm_full_{m}" for m in MODELS]),
    ("4 Cross-rule", ["batch11_mgr_claude_qwen", "batch11_mgr_qwen_claude", "batch11_mgr_grok_qwen",
                      "batch11_mgr_gpt4o_qwen", "batch11_mgr_claude_grok", "batch11_mgr_grok_claude"]),
]
# key same-model setups re-run with a newer model version swapped in (--swap-model)
SWAP_GROUPS = [
    ("Newer model: Baseline + Manager type", ["batch1_baseline_gpt4o"] + [f"batch3_mgr_{t}_gpt4o" for t in ("fixed", "elected", "rotating")]),
    ("Newer model: Manager pay", [f"batch8_mgr_{s}_gpt4o" for s in ("salary", "costly")]),
    ("Newer model: Punishment visibility", [f"batch9_punish_{v}_gpt4o" for v in ("hidden", "anonymous")]),
]
# OSS track: same-model setups (templates use the gpt4o version; models are replaced by LOCAL)
OSS_GROUPS = [
    ("1 Baseline + Manager type", ["batch1_baseline_gpt4o"] + [f"batch3_mgr_{t}_gpt4o" for t in ("fixed", "elected", "rotating")]),
    ("2 Manager pay", [f"batch8_mgr_{s}_gpt4o" for s in ("salary", "costly")]),
    ("3 Punishment visibility", [f"batch9_punish_{v}_gpt4o" for v in ("hidden", "anonymous")]),
    ("4 Communication", [f"batch2_comm_{c}_gpt4o" for c in ("public", "private", "full")]),
    ("5 Belief", [f"batch5_belief_{b}_gpt4o" for b in ("unknown", "all_human", "mixed")]),
]


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_credit_cache = {"t": 0.0, "v": None}


def openrouter_credit(max_age=60):
    """(total_credits, total_usage) of the account, cached for max_age seconds; None if unavailable."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    if time.time() - _credit_cache["t"] < max_age and _credit_cache["v"]:
        return _credit_cache["v"]
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                     headers={"Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=20))["data"]
        _credit_cache.update(t=time.time(), v=(float(d["total_credits"]), float(d["total_usage"])))
    except Exception:
        pass
    return _credit_cache["v"]


class Runner:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out) / args.track
        self.out.mkdir(parents=True, exist_ok=True)
        tag = args.tag or args.swap_model
        self.status_path = Path(args.out) / f"status_{args.track}{'_' + tag if tag else ''}.json"
        self.lock = threading.Lock()
        self.stop_reason = None
        self.configs = {c.name: c for c in generate_all_conditions(include_reverse_pairs=False)}
        self.jobs = self._build_jobs()
        self.state = {j["id"]: ("done" if j["path"].exists() else "queued") for j in self.jobs}
        self.errors, self.durations = [], []
        self.start_credit = openrouter_credit() if args.track == "api" else None
        self.started = now()

    def _build_jobs(self):
        groups = OSS_GROUPS if self.a.track == "oss" else (SWAP_GROUPS if self.a.swap_model else API_GROUPS)
        jobs = []
        for gname, names in groups:
            for name in names:
                cfg = copy.deepcopy(self.configs[name])
                if self.a.track == "oss":
                    cfg.agent_models = [ModelType.LOCAL] * cfg.num_agents
                    cfg.name = name.replace("_gpt4o", f"_{self.a.tag}")
                elif self.a.swap_model:
                    cfg.agent_models = [ModelType(self.a.swap_model)] * cfg.num_agents
                    cfg.name = name.replace("_gpt4o", f"_{self.a.swap_model}")
                cfg.num_rounds, cfg.num_trials = 20, self.a.trials
                if self.a.only and not any(o in cfg.name for o in self.a.only.split(",")):
                    continue
                for k in range(self.a.trials):
                    jid = f"{cfg.name}__t{k}"
                    jobs.append({"id": jid, "group": gname, "cfg": cfg, "trial": k,
                                 "path": self.out / f"{jid}.json"})
        return jobs

    # ---------------- budget ----------------
    def spent(self):
        if not self.start_credit:
            return None, None
        cur = openrouter_credit()
        if not cur:
            return None, None
        return cur[1] - self.start_credit[1], cur[0] - cur[1]

    def budget_ok(self):
        if self.a.track != "api":
            return True
        spent, remaining = self.spent()
        if spent is None:
            return True  # credit endpoint unreachable: keep going, the cap is re-checked next time
        if spent >= self.a.budget_usd:
            self.stop_reason = f"budget reached: spent ${spent:.2f} of ${self.a.budget_usd:.0f}"
        elif remaining < self.a.min_remaining_usd:
            self.stop_reason = f"account credit low: ${remaining:.2f} left"
        return self.stop_reason is None

    # ---------------- one game ----------------
    def run_job(self, job):
        from experiments.runner import ExperimentRunner
        with self.lock:
            if self.stop_reason or not self.budget_ok():
                self.state[job["id"]] = "skipped"
                return
            self.state[job["id"]] = "running"
        self.write_status()
        t0 = time.time()
        try:
            runner = ExperimentRunner(job["cfg"], results_dir=self.out, verbose=False)
            trial = runner.run_single_trial(job["trial"])
            out = {"config": job["cfg"].to_dict(), "trials": [trial], "group": job["group"],
                   "track": self.a.track, "finished": now()}
            tmp = job["path"].with_suffix(".tmp")
            tmp.write_text(json.dumps(out, default=str), encoding="utf-8")
            os.replace(tmp, job["path"])
            with self.lock:
                self.state[job["id"]] = "done"
                self.durations.append(time.time() - t0)
        except Exception:
            with self.lock:
                self.state[job["id"]] = "failed"
                self.errors.append(f"{now()} {job['id']}: {traceback.format_exc()[-800:]}")
        finally:
            self.write_status()

    # ---------------- status ----------------
    def write_status(self):
        """Rewrite this track's status JSON and STATUS.md. Never raises: a status problem must
        not stop a game (two threads once raced on the same temp file and killed the runner)."""
        try:
            self._write_status()
        except Exception:
            with self.lock:
                self.errors.append(f"{now()} status write failed: {traceback.format_exc()[-300:]}")

    def _write_status(self):
        spent, remaining = self.spent() if self.a.track == "api" else (None, None)
        with self.lock:
            groups = {}
            for j in self.jobs:
                g = groups.setdefault(j["group"], {"done": 0, "running": 0, "queued": 0, "failed": 0, "skipped": 0})
                g[self.state[j["id"]]] += 1
            left = sum(1 for s in self.state.values() if s in ("queued", "running"))
            avg = sum(self.durations) / len(self.durations) if self.durations else None
            st = {"track": self.a.track, "tag": self.a.tag, "started": self.started, "updated": now(),
                  "workers": self.a.workers, "groups": groups, "spent_usd": spent,
                  "account_remaining_usd": remaining, "budget_usd": self.a.budget_usd if self.a.track == "api" else None,
                  "avg_game_min": round(avg / 60, 1) if avg else None,
                  "eta_hours": round(left * avg / self.a.workers / 3600, 1) if avg else None,
                  "stop_reason": self.stop_reason, "errors": self.errors[-5:],
                  "finished": left == 0}
            tmp = self.status_path.with_name(f"{self.status_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
            os.replace(tmp, self.status_path)
            render_status(Path(self.a.out))

    def run(self):
        todo = [j for j in self.jobs if self.state[j["id"]] == "queued"]
        print(f"{now()} {self.a.track} {self.a.tag or ''}: {len(self.jobs)} jobs, {len(todo)} to run, "
              f"{self.a.workers} workers", flush=True)
        self.write_status()
        done_evt = threading.Event()

        def heartbeat():  # refresh spend and ETA every 2 minutes even when no game finishes
            while not done_evt.wait(120):
                self.write_status()
        threading.Thread(target=heartbeat, daemon=True).start()
        with ThreadPoolExecutor(max_workers=self.a.workers) as pool:
            for f in as_completed([pool.submit(self.run_job, j) for j in todo]):
                try:
                    f.result()
                except Exception:  # never let one game stop the whole track
                    with self.lock:
                        self.errors.append(f"{now()} runner: {traceback.format_exc()[-500:]}")
        # one retry round for failed games
        retry = [j for j in self.jobs if self.state[j["id"]] == "failed"]
        if retry and not self.stop_reason:
            with ThreadPoolExecutor(max_workers=self.a.workers) as pool:
                list(pool.map(self.run_job, retry))
        done_evt.set()
        self.write_status()
        print(f"{now()} finished: " + json.dumps({s: list(self.state.values()).count(s) for s in set(self.state.values())}), flush=True)


def render_status(root: Path):
    """Combine every status_*.json into one STATUS.md."""
    lines = [f"# HG re-run status\n", f"_Updated {now()} (written by hg_jobs.py)_\n"]
    for p in sorted(root.glob("status_*.json")):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        title = st["track"].upper() + (f" — {st['tag']}" if st.get("tag") else "")
        state = "FINISHED" if st["finished"] else ("STOPPED: " + st["stop_reason"] if st["stop_reason"] else "running")
        lines.append(f"## {title} ({state})\n")
        lines.append(f"Started {st['started']}, updated {st['updated']}, {st['workers']} parallel games, "
                     f"avg game {st['avg_game_min']} min, ETA {st['eta_hours']} h\n")
        if st["track"] == "api" and st["spent_usd"] is not None:
            lines.append(f"**Spend this run: ${st['spent_usd']:.2f} of ${st['budget_usd']:.0f}**, "
                         f"account credit left ${st['account_remaining_usd']:.2f}\n")
        lines.append("| group | done | running | queued | failed | skipped |")
        lines.append("|---|---|---|---|---|---|")
        for g, c in st["groups"].items():
            lines.append(f"| {g} | {c['done']} | {c['running']} | {c['queued']} | {c['failed']} | {c['skipped']} |")
        if st["errors"]:
            lines.append("\n<details><summary>Recent errors</summary>\n\n```\n" + "\n".join(st["errors"]) + "\n```\n</details>")
        lines.append("")
    tmp = root / f"STATUS.md.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, root / "STATUS.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["api", "oss"], required=True)
    ap.add_argument("--tag", default="", help="short model tag for the oss track, e.g. gpt-oss-120b")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget-usd", type=float, default=230.0)
    ap.add_argument("--min-remaining-usd", type=float, default=40.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--swap-model", default="", help="api track: re-run the key setups with this ModelType (gpt5, deepseek31)")
    ap.add_argument("--only", default="", help="comma-separated substrings; run only matching configs (pilots)")
    a = ap.parse_args()
    if a.track == "oss" and not a.tag:
        ap.error("--tag is required for the oss track")
    if a.track == "oss":
        # the open-model track must never spend API credit (deception scoring falls back to regex)
        os.environ.pop("OPENROUTER_API_KEY", None)
    r = Runner(a)
    if a.dry_run:
        from collections import Counter
        print(Counter(j["group"] for j in r.jobs), "total", len(r.jobs))
        print("already done:", sum(1 for s in r.state.values() if s == "done"))
    else:
        r.run()
