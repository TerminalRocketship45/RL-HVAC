from __future__ import annotations
import argparse
import os
import traceback
from pathlib import Path
import numpy as np
from rlhvac import run_store
from rlhvac.spec import RunStatus
from rlhvac.adapters import get_adapter
from rlhvac.adapters.base import default_read_scene


def _read_scene(adapter, env, reward) -> dict:
    fn = getattr(adapter, "read_scene", None)
    if callable(fn):
        try:
            return fn(env)
        except Exception:
            pass
    return default_read_scene(reward=reward)


def _run_baseline(run_dir: Path, job) -> None:
    adapter = get_adapter(job.sim)
    env = adapter.make({**job.config, "scenario": job.scenario})
    policy = adapter.baseline_policy(env)
    for ep in range(max(1, job.episodes)):
        ep_dir = run_store.create_episode(run_dir, ep)
        obs, _ = env.reset(seed=job.seed + ep)
        episode: list[dict] = []
        step = 0
        done = False
        while not done:
            action = policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            episode.append({"reward": float(reward)})
            if job.visual:
                run_store.append_frame(ep_dir, {
                    "step": step, "reward": float(reward),
                    "action": [float(a) for a in np.atleast_1d(action)],
                    "scene": _read_scene(adapter, env, reward),
                })
            step += 1
            done = term or trunc
        summary = adapter.summarize(episode)
        run_store.write_episode_summary(ep_dir, summary)
        run_store.append_rollup(run_dir, {"episode": ep,
                                          "total_reward": float(sum(s["reward"] for s in episode)),
                                          **summary})
        run_store.write_status(run_dir, RunStatus(state="running", pid=os.getpid(),
                                                  current_episode=ep, episodes_total=job.episodes))


def run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    try:
        job = run_store.read_job(run_dir)
        run_store.write_status(run_dir, RunStatus(state="running", pid=os.getpid(),
                                                  episodes_total=job.episodes))
        if job.mode == "baseline":
            _run_baseline(run_dir, job)
        else:
            raise NotImplementedError(f"mode '{job.mode}' arrives in a later phase")
        run_store.write_status(run_dir, RunStatus(state="done", progress=1.0, pid=os.getpid(),
                                                  current_episode=max(0, job.episodes - 1),
                                                  episodes_total=job.episodes))
    except Exception:
        tb = traceback.format_exc()
        try:
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "logs" / "runner.log").write_text(tb)
            run_store.write_status(run_dir, RunStatus(state="error", pid=os.getpid(), error=tb[-2000:]))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="path to job.json")
    args = parser.parse_args()
    run(Path(args.spec).parent)


if __name__ == "__main__":
    main()
