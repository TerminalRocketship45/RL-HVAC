from rlhvac.adapters import get_manifest, REGISTRY


def test_citylearn_registered():
    assert "citylearn" in REGISTRY


def test_citylearn_manifest_loads_without_importing_citylearn():
    # This test runs in rlhvac-ui where citylearn is NOT installed.
    m = get_manifest("citylearn")
    assert m.runner_env == "rlhvac-citylearn"
    assert any("citylearn_challenge_2022" in s for s in m.scenarios)
    field_names = {f.name for f in m.config_schema}
    assert {"simulation_steps", "seed"}.issubset(field_names)


def test_citylearn_module_top_has_no_heavy_import():
    import ast, pathlib
    src = pathlib.Path("rlhvac/adapters/citylearn.py").read_text()
    tree = ast.parse(src)
    top_imports = []
    for node in tree.body:  # module-level only
        if isinstance(node, ast.Import):
            top_imports += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    assert not any("citylearn" in name for name in top_imports), top_imports


import importlib.util
import pytest

citylearn_installed = importlib.util.find_spec("citylearn") is not None
requires_citylearn = pytest.mark.skipif(not citylearn_installed, reason="citylearn not in this env")


@requires_citylearn
def test_citylearn_make_and_short_baseline_episode():
    import numpy as np
    from rlhvac.adapters.citylearn import CityLearnAdapter
    adapter = CityLearnAdapter()
    env = adapter.make({"scenario": "citylearn_challenge_2022_phase_2",
                        "simulation_steps": 48, "seed": 0})
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(policy(obs))
        steps += 1
        done = term or trunc
        assert steps <= 60  # guard against runaway
    assert 1 < steps <= 48
    summary = adapter.summarize([])
    assert isinstance(summary, dict) and len(summary) > 0
