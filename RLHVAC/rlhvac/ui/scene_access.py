from __future__ import annotations
from rlhvac.spec import SceneSchema
from rlhvac.adapters import get_adapter
from rlhvac.adapters.base import default_scene_schema


def get_scene_schema(sim: str) -> SceneSchema:
    """Return the adapter's Scene schema, or the default fallback if it has none.
    Must work in the rlhvac-ui env without importing the simulator package."""
    adapter = get_adapter(sim)
    fn = getattr(adapter, "scene_schema", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    return default_scene_schema()
