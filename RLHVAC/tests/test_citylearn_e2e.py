import importlib.util
import pytest
from rlhvac.spec import JobSpec
from rlhvac import run_store, launcher

citylearn_installed = importlib.util.find_spec("citylearn") is not None


@pytest.mark.skipif(citylearn_installed, reason="run from an env WITHOUT citylearn (the UI env)")
def test_ui_env_spawns_citylearn_runner(tmp_path):
    # From rlhvac-ui: spawn the runner into rlhvac-citylearn via conda run.
    job = JobSpec(run_id="cl-e2e", sim="citylearn",
                  scenario="citylearn_challenge_2022_phase_2",
                  config={"simulation_steps": 48, "seed": 0},
                  mode="baseline", algo=None, timesteps=0, seed=0, visual=True)
    run_dir = run_store.create_run(tmp_path, job)
    proc = launcher.spawn(run_dir, runner_env="rlhvac-citylearn")
    proc.wait(timeout=600)  # first run downloads the dataset
    status = run_store.read_status(run_dir)
    assert status.state == "done", f"state={status.state} error={status.error}"
    steps = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    assert len(steps) > 1
    summary = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "summary"]
    assert summary and len(summary[-1]) > 1
