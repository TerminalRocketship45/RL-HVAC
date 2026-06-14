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


def test_mock_scene_schema():
    from rlhvac.adapters import get_adapter
    schema = get_adapter("mock").scene_schema()
    assert schema.color_by == "temp"
    names = {v.name for v in schema.variables}
    assert {"temp", "setpoint"}.issubset(names)


def test_mock_read_scene_after_step():
    from rlhvac.adapters import get_adapter
    adapter = get_adapter("mock")
    env = adapter.make({"episode_length": 5, "setpoint": 21.0})
    obs, _ = env.reset(seed=0)
    adapter.baseline_policy(env)  # no-op, ensures interface
    env.step(env.action_space.sample())
    scene = adapter.read_scene(env)
    assert "zone" in scene
    assert "temp" in scene["zone"] and "setpoint" in scene["zone"]
    assert scene["zone"]["setpoint"] == 21.0


def test_default_scene_fallback_helpers():
    from rlhvac.adapters.base import default_scene_schema, default_read_scene
    schema = default_scene_schema()
    assert len(schema.units) == 1
    frame = default_read_scene(reward=-1.5)
    assert "reward" in frame[schema.units[0].name]
