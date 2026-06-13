from __future__ import annotations
import argparse
import os
import traceback
from pathlib import Path
from rlhvac import run_store
from rlhvac.spec import RunStatus
from rlhvac.adapters import get_adapter


def _run_baseline(run_dir: Path, job) -> None:
    adapter = get_adapter(job.sim)
    env = adapter.make(job.config)
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=job.seed)
    episode: list[dict] = []
    step = 0
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        episode.append({"reward": float(reward), "info": {k: float(v) for k, v in info.items()
                                                          if isinstance(v, (int, float))}})
        if job.visual:
            run_store.append_metric(run_dir, {
                "kind": "step", "step": step, "reward": float(reward),
                **{k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
            })
        step += 1
        done = term or trunc
    summary = adapter.summarize(episode)
    run_store.append_metric(run_dir, {"kind": "summary", **summary})


def run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    job = run_store.read_job(run_dir)
    run_store.write_status(run_dir, RunStatus(state="running", pid=os.getpid()))
    try:
        if job.mode == "baseline":
            _run_baseline(run_dir, job)
        else:
            raise NotImplementedError(f"mode '{job.mode}' arrives in a later phase")
        run_store.write_status(run_dir, RunStatus(state="done", progress=1.0, pid=os.getpid()))
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "logs" / "runner.log").write_text(tb)
        run_store.write_status(run_dir, RunStatus(state="error", pid=os.getpid(), error=tb[-2000:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="path to job.json")
    args = parser.parse_args()
    run(Path(args.spec).parent)


if __name__ == "__main__":
    main()
