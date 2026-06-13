import json
from rlhvac.spec import JobSpec
from rlhvac import run_store


def _job(run_id):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day", config={},
                   mode="baseline", algo=None, timesteps=0, seed=7, visual=True)


def test_create_run_writes_job_and_queued_status(tmp_path):
    job = _job("r1")
    run_dir = run_store.create_run(tmp_path, job)
    assert (run_dir / "job.json").exists()
    assert run_store.read_status(run_dir).state == "queued"


def test_append_and_read_metrics(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2"))
    run_store.append_metric(run_dir, {"step": 0, "reward": -3.0})
    run_store.append_metric(run_dir, {"step": 1, "reward": -2.0})
    metrics = run_store.read_metrics(run_dir)
    assert [m["step"] for m in metrics] == [0, 1]


def test_new_run_id_is_unique():
    assert run_store.new_run_id() != run_store.new_run_id()
