from rlhvac.spec import AdapterManifest, ConfigField
from rlhvac.ui.manifest_view import default_config, coerce_config


def _manifest():
    return AdapterManifest(
        name="mock", scenarios=["sine-day"],
        config_schema=[
            ConfigField(name="episode_length", type="int", label="Len", default=24),
            ConfigField(name="setpoint", type="number", label="SP", default=21.0),
        ],
        runner_env="rlhvac-ui",
    )


def test_default_config_from_schema():
    assert default_config(_manifest()) == {"episode_length": 24, "setpoint": 21.0}


def test_coerce_casts_types():
    raw = {"episode_length": "30", "setpoint": "22.5"}
    out = coerce_config(_manifest(), raw)
    assert out == {"episode_length": 30, "setpoint": 22.5}
    assert isinstance(out["episode_length"], int)
