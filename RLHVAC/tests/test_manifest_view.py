from rlhvac.spec import AdapterManifest, ConfigField
from rlhvac.ui.manifest_view import default_config, coerce_config, render_config_form


def _manifest():
    return AdapterManifest(
        name="mock", scenarios=["sine-day"],
        config_schema=[
            ConfigField(name="episode_length", type="int", label="Len", default=24,
                        min=1, max=168),
            ConfigField(name="setpoint", type="number", label="SP", default=21.0,
                        min=10, max=30),
        ],
        runner_env="rlhvac-ui",
    )


class _FakeSt:
    """Records number_input kwargs and echoes back the value."""

    def __init__(self):
        self.calls = []

    def number_input(self, label, value=None, **kwargs):
        self.calls.append({"label": label, "value": value, **kwargs})
        return value


def test_render_config_form_keeps_numeric_bounds_same_type_as_value():
    # Regression: a float field with int min/max must not pass mixed types to
    # Streamlit's number_input (StreamlitMixedNumericTypesError).
    st = _FakeSt()
    out = render_config_form(st, _manifest())
    assert out == {"episode_length": 24, "setpoint": 21.0}
    by_label = {c["label"]: c for c in st.calls}
    sp = by_label["SP"]
    assert type(sp["value"]) is float
    assert type(sp["min_value"]) is float and type(sp["max_value"]) is float
    length = by_label["Len"]
    assert type(length["value"]) is int
    assert type(length["min_value"]) is int and type(length["max_value"]) is int


def test_default_config_from_schema():
    assert default_config(_manifest()) == {"episode_length": 24, "setpoint": 21.0}


def test_coerce_casts_types():
    raw = {"episode_length": "30", "setpoint": "22.5"}
    out = coerce_config(_manifest(), raw)
    assert out == {"episode_length": 30, "setpoint": 22.5}
    assert isinstance(out["episode_length"], int)
