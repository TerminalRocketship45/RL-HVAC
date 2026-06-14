from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.tabs.metrics_tab import rollup_metric_names


def test_rollup_metric_names(tmp_path):
    job = JobSpec(run_id="m", sim="mock", scenario="sine-day",
                  config={"episode_length": 3}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True, episodes=2)
    run_dir = run_store.create_run(tmp_path, job)
    runner.run(run_dir)
    names = rollup_metric_names(run_dir)
    assert "total_reward" in names
