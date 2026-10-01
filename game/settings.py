"""
Run-time settings for the 2026-09 re-run, read from environment variables.

Every value is also written into each result file (see experiments/runner.py), so a result
always records which rules it was produced under.

HG_SANCTION_COST   "1" (default): the manager pays for every token it spends on punishment or
                   reward, as in Eq. 2 of the paper. "0": sanctions are free for the manager
                   (the behaviour of the code that produced the June 2026 results).
HG_LOCAL_BASE_URL  OpenAI-compatible endpoint for ModelType.LOCAL (a local vLLM server).
HG_LOCAL_MODEL     Model name served at that endpoint (e.g. openai/gpt-oss-120b).
"""
import os

SANCTION_COST_TO_MANAGER: bool = os.getenv("HG_SANCTION_COST", "1") != "0"
LOCAL_BASE_URL: str = os.getenv("HG_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
LOCAL_MODEL: str = os.getenv("HG_LOCAL_MODEL", "")


def as_dict() -> dict:
    return {
        "sanction_cost_to_manager": SANCTION_COST_TO_MANAGER,
        "local_model": LOCAL_MODEL or None,
        "code_commit": os.getenv("HG_CODE_COMMIT", "unknown"),
    }
