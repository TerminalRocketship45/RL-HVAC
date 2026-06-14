from rlhvac.ui.scene_access import get_scene_schema


def test_mock_uses_its_own_schema():
    schema = get_scene_schema("mock")
    assert schema.color_by == "temp"


def test_citylearn_scene_schema_via_scene_access():
    # CityLearn now has its own scene_schema (Plan 3) — frame-driven, units=[].
    # Must not import citylearn at the module level (lazy-import rule).
    schema = get_scene_schema("citylearn")
    assert schema.color_by == "indoor_dry_bulb_temperature"
    assert len(schema.variables) >= 1
    import sys
    assert "citylearn.citylearn" not in sys.modules
