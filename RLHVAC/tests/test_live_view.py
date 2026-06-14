from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.live_view import latest_episode_frames


def test_latest_episode_frames(tmp_path):
    job = JobSpec(run_id="lv1", sim="mock", scenario="sine-day",
                  config={"episode_length": 4}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True, episodes=1)
    run_dir = run_store.create_run(tmp_path, job)
    runner.run(run_dir)
    df = latest_episode_frames(run_dir)
    assert list(df["step"]) == [0, 1, 2, 3]
    assert "reward" in df.columns
