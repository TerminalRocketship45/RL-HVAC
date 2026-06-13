from __future__ import annotations
from typing import Any
from rlhvac.spec import AdapterManifest, ConfigField


def default_config(manifest: AdapterManifest) -> dict:
    return {f.name: f.default for f in manifest.config_schema}


def _cast(field: ConfigField, value: Any) -> Any:
    if field.type == "int":
        return int(value)
    if field.type == "number":
        return float(value)
    if field.type == "bool":
        return bool(value)
    return value


def coerce_config(manifest: AdapterManifest, raw: dict) -> dict:
    return {f.name: _cast(f, raw[f.name]) for f in manifest.config_schema if f.name in raw}


def _bound(value: Any, cast) -> Any:
    """Coerce a min/max bound to the field's numeric type, leaving None as-is.

    Streamlit's number_input requires value, min_value, and max_value to share
    one numeric type, so int bounds on a float field (or vice versa) must match."""
    return None if value is None else cast(value)


def render_config_form(st, manifest: AdapterManifest) -> dict:
    """Render Streamlit widgets for each field; return a coerced config dict.

    `st` is the streamlit module (passed in so this stays import-light/testable)."""
    raw: dict = {}
    for f in manifest.config_schema:
        if f.type == "int":
            raw[f.name] = st.number_input(f.label, value=int(f.default), step=1,
                                          min_value=_bound(f.min, int),
                                          max_value=_bound(f.max, int))
        elif f.type == "number":
            raw[f.name] = st.number_input(f.label, value=float(f.default),
                                          min_value=_bound(f.min, float),
                                          max_value=_bound(f.max, float))
        elif f.type == "bool":
            raw[f.name] = st.checkbox(f.label, value=bool(f.default))
        elif f.type == "select":
            raw[f.name] = st.selectbox(f.label, options=f.options or [])
        else:
            raw[f.name] = st.text_input(f.label, value=str(f.default))
    return coerce_config(manifest, raw)
