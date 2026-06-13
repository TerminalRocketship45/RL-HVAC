import time
from rlhvac.spec import JobSpec
from rlhvac import run_store, launcher


def test_spawned_runner_completes(tmp_path):
    job = JobSpec(run_id="e2e1", sim="mock", scenario="sine-day",
                  config={"episode_length": 4}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True)
    run_dir = run_store.create_run(tmp_path, job)
    proc = launcher.spawn(run_dir, runner_env=None)
    proc.wait(timeout=60)
    assert run_store.read_status(run_dir).state == "done"
    steps = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    assert len(steps) == 4
