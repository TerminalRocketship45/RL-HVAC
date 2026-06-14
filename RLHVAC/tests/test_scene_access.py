from rlhvac.ui.scene_access import get_scene_schema


def test_mock_uses_its_own_schema():
    schema = get_scene_schema("mock")
    assert schema.color_by == "temp"


def test_adapter_without_scene_falls_back():
    # CityLearn has no scene_schema yet (Plan 3) -> default fallback, no crash,
    # and crucially this runs in rlhvac-ui WITHOUT importing citylearn.
    schema = get_scene_schema("citylearn")
    assert len(schema.units) >= 1
    assert schema.color_by  # non-empty
