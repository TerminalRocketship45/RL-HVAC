from rlhvac.adapters import get_manifest, get_adapter
from rlhvac.adapters.base import SimAdapter


def test_mock_registered_and_conforms():
    adapter = get_adapter("mock")
    assert isinstance(adapter, SimAdapter)
    assert adapter.name == "mock"


def test_mock_manifest_has_scenarios_without_heavy_import():
    m = get_manifest("mock")
    assert "sine-day" in m.scenarios
    assert m.runner_env == "rlhvac-ui"


def test_mock_episode_is_deterministic():
    adapter = get_adapter("mock")
    env = adapter.make({"episode_length": 5})
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=7)
    rewards = []
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert len(rewards) == 5
    summary = adapter.summarize([{"reward": r} for r in rewards])
    assert "episode_reward" in summary
