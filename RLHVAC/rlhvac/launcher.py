from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional

UI_ENV = "rlhvac-ui"


def build_command(run_dir, runner_env: Optional[str]) -> list[str]:
    """Build the runner invocation.

    runner_env=None  -> use the current interpreter (used for the mock, whose
                        runner_env is rlhvac-ui = the UI env itself).
    runner_env="X"   -> `conda run -n X python -m rlhvac.runner ...`
    """
    spec_path = str(Path(run_dir) / "job.json")
    if runner_env is None or runner_env == UI_ENV:
        return [sys.executable, "-m", "rlhvac.runner", "--spec", spec_path]
    return ["conda", "run", "-n", runner_env, "python", "-m", "rlhvac.runner", "--spec", spec_path]


def spawn(run_dir, runner_env: Optional[str]) -> subprocess.Popen:
    cmd = build_command(run_dir, runner_env)
    log = open(Path(run_dir) / "logs" / "spawn.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
