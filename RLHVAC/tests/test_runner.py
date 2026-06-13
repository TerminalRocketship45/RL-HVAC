from rlhvac.spec import JobSpec
from rlhvac import run_store, runner


def _job(run_id, episode_length=5):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": episode_length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True)


def test_runner_baseline_completes_and_writes_metrics(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r1", episode_length=5))
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "done"
    metrics = run_store.read_metrics(run_dir)
    step_metrics = [m for m in metrics if m.get("kind") == "step"]
    assert len(step_metrics) == 5


def test_runner_writes_summary_on_done(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2", episode_length=3))
    runner.run(run_dir)
    metrics = run_store.read_metrics(run_dir)
    summary = [m for m in metrics if m.get("kind") == "summary"][-1]
    assert "episode_reward" in summary


def test_runner_records_error_on_bad_sim(tmp_path):
    bad = JobSpec(run_id="r3", sim="does-not-exist", scenario="x", config={},
                  mode="baseline", algo=None, timesteps=0, seed=1, visual=True)
    run_dir = run_store.create_run(tmp_path, bad)
    runner.run(run_dir)
    status = run_store.read_status(run_dir)
    assert status.state == "error"
    assert status.error  # traceback captured


def test_runner_records_error_on_corrupt_job(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r4"))
    (run_dir / "job.json").write_text("{ this is not valid json")
    runner.run(run_dir)
    status = run_store.read_status(run_dir)
    assert status.state == "error"
    assert status.error
