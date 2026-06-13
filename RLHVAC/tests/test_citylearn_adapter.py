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
