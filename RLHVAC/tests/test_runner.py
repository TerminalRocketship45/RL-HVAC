from rlhvac.spec import JobSpec
from rlhvac import run_store, runner


def _job(run_id, episode_length=5, episodes=1):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": episode_length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True, episodes=episodes)


def test_runner_writes_frames_per_episode(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r1", episode_length=5, episodes=2))
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "done"
    assert run_store.list_episodes(run_dir) == [0, 1]
    frames0 = run_store.read_frames(run_dir / "episodes" / "000")
    assert len(frames0) == 5
    assert "scene" in frames0[0] and "zone" in frames0[0]["scene"]


def test_runner_writes_rollup_row_per_episode(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2", episode_length=3, episodes=3))
    runner.run(run_dir)
    rollup = run_store.read_rollup(run_dir)
    assert [r["episode"] for r in rollup] == [0, 1, 2]
    assert all("total_reward" in r for r in rollup)


def test_runner_done_status_reports_final_episode(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("rdone", episode_length=2, episodes=3))
    runner.run(run_dir)
    status = run_store.read_status(run_dir)
    assert status.state == "done"
    assert status.current_episode == 2 and status.episodes_total == 3


def test_runner_records_error_on_bad_sim(tmp_path):
    bad = JobSpec(run_id="r3", sim="nope", scenario="x", config={}, mode="baseline",
                  algo=None, timesteps=0, seed=1, visual=True, episodes=1)
    run_dir = run_store.create_run(tmp_path, bad)
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "error"


def test_runner_records_error_on_corrupt_job(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r4"))
    (run_dir / "job.json").write_text("{ not json")
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "error"
