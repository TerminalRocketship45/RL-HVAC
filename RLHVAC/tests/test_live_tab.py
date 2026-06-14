from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.tabs.live_tab import list_run_dirs, episode_frame_at


def _job(run_id, episodes=1, length=4):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True, episodes=episodes)


def test_list_run_dirs_newest_first(tmp_path):
    run_store.create_run(tmp_path, _job("a"))
    run_store.create_run(tmp_path, _job("b"))
    dirs = list_run_dirs(tmp_path)
    assert {d.name for d in dirs} == {"a", "b"}


def test_episode_frame_at_returns_named_scene(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r", length=4))
    runner.run(run_dir)
    frame = episode_frame_at(run_dir, episode=0, step_index=2)
    assert frame["step"] == 2
    assert "zone" in frame["scene"]
