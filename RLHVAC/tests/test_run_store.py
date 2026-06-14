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


def test_new_run_id_is_unique():
    assert run_store.new_run_id() != run_store.new_run_id()


def test_episode_dir_and_frames(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2"))
    ep = run_store.create_episode(run_dir, 0)
    run_store.append_frame(ep, {"step": 0, "reward": -3.0, "scene": {"zone": {"temp": 15.0}}})
    run_store.append_frame(ep, {"step": 1, "reward": -2.0, "scene": {"zone": {"temp": 16.0}}})
    frames = run_store.read_frames(ep)
    assert [f["step"] for f in frames] == [0, 1]
    assert run_store.list_episodes(run_dir) == [0]


def test_episode_summary_and_rollup(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r3"))
    ep = run_store.create_episode(run_dir, 0)
    run_store.write_episode_summary(ep, {"episode_reward": -10.0})
    assert run_store.read_episode_summary(ep)["episode_reward"] == -10.0
    run_store.append_rollup(run_dir, {"episode": 0, "total_reward": -10.0})
    run_store.append_rollup(run_dir, {"episode": 1, "total_reward": -8.0})
    rollup = run_store.read_rollup(run_dir)
    assert [r["total_reward"] for r in rollup] == [-10.0, -8.0]
