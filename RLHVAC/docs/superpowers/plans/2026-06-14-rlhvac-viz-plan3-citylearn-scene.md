# RLHVAC Viz Plan 3 — CityLearn Building-Grid Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give CityLearn a real, accurate visual: a grid of building cells colored by indoor temperature, hover showing every per-building variable CityLearn exposes (load, solar, storage SoC, net electricity, price, carbon, occupancy). Make the heatmap **frame-driven** (units come from the actual run frames) so it works for any dataset's building count without the UI importing CityLearn.

**Architecture:** Evolve `SceneSchema` to carry flat **variable metadata** (`variables: list[VarSpec]`) separate from concrete units. Refactor the Plotly renderers to lay out units from the **frame's scene keys**, using the schema for labels/units + `color_by`/`color_range`. Add `scene_schema()`/`read_scene()` to the CityLearn adapter: schema is static (no `citylearn` import); `read_scene` reads each `Building.observations()` in the citylearn env.

**Tech Stack:** CityLearn v2 (2.3.1), Python 3.11, existing rlhvac package.

**IMPORTANT:** Validate CityLearn's `Building.observations()` signature and the exact observation keys against the INSTALLED package (probe in Task 3 Step 1). Adapt the variable name list to what the chosen dataset actually exposes.

---

## File Structure

| File | Change |
| ---- | ------ |
| `rlhvac/spec.py` | `SceneSchema`: add `variables: list[VarSpec]` + `variable_meta(name)`; make `units` optional |
| `rlhvac/adapters/mock.py` | populate `variables` in `scene_schema()` |
| `rlhvac/adapters/base.py` | `default_scene_schema()` sets `variables` |
| `rlhvac/ui/viz_plotly.py` | `heatmap_figure`/`variable_timeseries` become frame-driven for units |
| `rlhvac/adapters/citylearn.py` | add `scene_schema()` + `read_scene()` |
| `tests/*` | update viz/scene tests; add CityLearn scene tests |

---

## Task 1: Evolve SceneSchema with variable metadata

**Files:** Modify `rlhvac/spec.py`, `rlhvac/adapters/base.py`, `rlhvac/adapters/mock.py`; Tests `tests/test_spec.py`, `tests/test_mock_adapter.py`

- [ ] **Step 1: Failing tests** — add to `tests/test_spec.py`:
```python
def test_scene_schema_variable_metadata_lookup():
    from rlhvac.spec import SceneSchema, VarSpec
    s = SceneSchema(variables=[VarSpec("temp", "Temperature", "C", "temperature")],
                    color_by="temp", color_range=(10.0, 30.0))
    assert s.variable_meta("temp").label == "Temperature"
    assert s.variable_meta("missing") is None
    assert s.units == []  # units now optional
```
and update `tests/test_mock_adapter.py::test_mock_scene_schema` to also assert variables:
```python
def test_mock_scene_schema():
    from rlhvac.adapters import get_adapter
    schema = get_adapter("mock").scene_schema()
    assert schema.color_by == "temp"
    names = {v.name for v in schema.variables}
    assert {"temp", "setpoint"}.issubset(names)
```

- [ ] **Step 2: Run to verify fail** → `conda run -n rlhvac-ui pytest tests/test_spec.py tests/test_mock_adapter.py -v` FAIL.

- [ ] **Step 3: Edit `rlhvac/spec.py`** — replace the `SceneSchema` dataclass with:
```python
@dataclass
class SceneSchema:
    color_by: str
    color_range: tuple[float, float] = (0.0, 1.0)
    layout: Literal["grid", "row", "diagram"] = "grid"
    variables: list[VarSpec] = field(default_factory=list)
    units: list[UnitSpec] = field(default_factory=list)

    def variable_meta(self, name: str) -> "VarSpec | None":
        return next((v for v in self.variables if v.name == name), None)
```
(Note: `color_by` is now the only required field; all callers use keyword args so this is safe.)

- [ ] **Step 4: Update `rlhvac/adapters/base.py`** `default_scene_schema()`:
```python
def default_scene_schema() -> SceneSchema:
    return SceneSchema(
        color_by="reward", color_range=(-10.0, 0.0), layout="grid",
        variables=[VarSpec(name="reward", label="Reward", kind="other")],
        units=[UnitSpec(name="system", label="System",
                        variables=[VarSpec(name="reward", label="Reward", kind="other")])],
    )
```

- [ ] **Step 5: Update `rlhvac/adapters/mock.py`** `scene_schema()` to set `variables` (keep `units` too):
```python
    @staticmethod
    def scene_schema() -> SceneSchema:
        vars_ = [
            VarSpec(name="temp", label="Temperature", unit="C", kind="temperature"),
            VarSpec(name="setpoint", label="Setpoint", unit="C", kind="setpoint"),
        ]
        return SceneSchema(color_by="temp", color_range=(10.0, 30.0), layout="grid",
                           variables=vars_,
                           units=[UnitSpec(name="zone", label="Zone", variables=vars_)])
```

- [ ] **Step 6: Run to verify pass** → both test files PASS. Then full suite `conda run -n rlhvac-ui pytest -q` (some viz tests may fail until Task 2 — that is expected; if so, proceed to Task 2 and they go green there). If viz tests still pass, even better.

- [ ] **Step 7: Commit**
```bash
git add rlhvac/spec.py rlhvac/adapters/base.py rlhvac/adapters/mock.py tests/test_spec.py tests/test_mock_adapter.py
git commit -m "feat: SceneSchema variable metadata; units optional"
```

---

## Task 2: Frame-driven Plotly renderers

**Files:** Modify `rlhvac/ui/viz_plotly.py`; Test `tests/test_viz_plotly.py`

- [ ] **Step 1: Update the viz tests** in `tests/test_viz_plotly.py` to the frame-driven contract. Replace `_schema()` to use `variables` (no `units`) and keep `_frame()`:
```python
def _schema():
    return SceneSchema(
        color_by="temp", color_range=(10.0, 30.0), layout="grid",
        variables=[VarSpec("temp", "Temp", "C", "temperature"),
                   VarSpec("load", "Load", "kW", "power")])
```
Keep `test_heatmap_figure_colors_by_color_by_and_sets_range` and
`test_heatmap_hover_lists_all_variables` (units now come from the frame's scene keys
`b0`/`b1`). Update `test_variable_timeseries_one_trace_per_unit` to not rely on
`schema.units` (units come from frames):
```python
def test_variable_timeseries_one_trace_per_unit():
    frames = [_frame(20, 25), _frame(21, 26), _frame(22, 27)]
    fig = viz.variable_timeseries(frames, _schema(), "temp")
    assert len(fig.data) == 2
    ys = {tuple(tr.y) for tr in fig.data}
    assert (20, 21, 22) in ys and (25, 26, 27) in ys
```

- [ ] **Step 2: Run to verify fail** (renderers still use schema.units) → FAIL.

- [ ] **Step 3: Rewrite the unit-driven parts of `rlhvac/ui/viz_plotly.py`**

Replace `_hover_for_unit` and `heatmap_figure` and `variable_timeseries` so units come from the frame:
```python
def _unit_names_from_frame(frame: dict) -> list[str]:
    return list((frame or {}).get("scene", {}).keys())


def _hover_for_unit(schema: SceneSchema, unit_name: str, values: dict) -> str:
    lines = [f"<b>{unit_name}</b>"]
    for var_name, val in values.items():
        meta = schema.variable_meta(var_name)
        label = meta.label if meta else var_name
        unit = f" {meta.unit}" if meta and meta.unit else ""
        shown = "n/a" if val is None else (f"{val:.3g}" if isinstance(val, (int, float)) else val)
        lines.append(f"{label}: {shown}{unit}")
    return "<br>".join(lines)


def heatmap_figure(schema: SceneSchema, frame: dict) -> go.Figure:
    scene = (frame or {}).get("scene", {})
    names = _unit_names_from_frame(frame) or [u.name for u in schema.units]
    rows, cols = _grid_shape(len(names))
    z, text, labels = [], [], []
    for r in range(rows):
        zr, tr, lr = [], [], []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(names):
                name = names[idx]
                vals = scene.get(name, {})
                cval = vals.get(schema.color_by)
                zr.append(cval if isinstance(cval, (int, float)) else None)
                tr.append(_hover_for_unit(schema, name, vals))
                lr.append(f"{name}<br>{'' if not isinstance(cval, (int, float)) else f'{cval:.3g}'}")
            else:
                zr.append(None); tr.append(""); lr.append("")
        z.append(zr); text.append(tr); labels.append(lr)
    fig = go.Figure(go.Heatmap(z=z, text=text, hoverinfo="text",
                               zmin=schema.color_range[0], zmax=schema.color_range[1],
                               colorscale="RdYlBu_r", showscale=True))
    fig.update_traces(texttemplate="%{customdata}", customdata=labels)
    fig.update_layout(yaxis=dict(autorange="reversed", showticklabels=False),
                      xaxis=dict(showticklabels=False),
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Colored by {schema.color_by}")
    return fig


def variable_timeseries(frames: list[dict], schema: SceneSchema, var: str) -> go.Figure:
    steps = [f.get("step") for f in frames]
    names: list[str] = []
    for f in frames:
        for n in f.get("scene", {}):
            if n not in names:
                names.append(n)
    fig = go.Figure()
    for name in names:
        ys = [f.get("scene", {}).get(name, {}).get(var) for f in frames]
        if any(y is not None for y in ys):
            fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines", name=name))
    fig.update_layout(title=var, margin=dict(l=10, r=10, t=30, b=10))
    return fig
```

- [ ] **Step 4: Run to verify pass** → `conda run -n rlhvac-ui pytest tests/test_viz_plotly.py -v` PASS, then full suite green.

- [ ] **Step 5: Commit**
```bash
git add rlhvac/ui/viz_plotly.py tests/test_viz_plotly.py
git commit -m "feat: frame-driven heatmap and timeseries (units from scene keys)"
```

---

## Task 3: CityLearn scene_schema + read_scene

**Files:** Modify `rlhvac/adapters/citylearn.py`; Test `tests/test_citylearn_adapter.py`

- [ ] **Step 1: Probe the installed Building API**

Run:
```bash
conda run -n rlhvac-citylearn python -c "from citylearn.citylearn import CityLearnEnv; e=CityLearnEnv(schema='citylearn_challenge_2022_phase_2', central_agent=True, simulation_end_time_step=3); e.reset(); b=e.buildings[0]; o=b.observations(); print('NAME', b.name); print('KEYS', sorted(o.keys()))"
```
Note the real observation keys for buildings (e.g. `indoor_dry_bulb_temperature`,
`net_electricity_consumption`, `electrical_storage_soc`, `solar_generation`,
`non_shiftable_load`, `occupant_count`, etc.). If `observations()` requires args in
2.3.1, find the working call. Record the exact keys — you'll list them in `_BUILDING_VARS`.

- [ ] **Step 2: Failing env-gated test** — append to `tests/test_citylearn_adapter.py`:
```python
@requires_citylearn
def test_citylearn_scene_has_named_buildings():
    from rlhvac.adapters.citylearn import CityLearnAdapter
    adapter = CityLearnAdapter()
    env = adapter.make({"scenario": "citylearn_challenge_2022_phase_2",
                        "simulation_steps": 4, "seed": 0})
    env.reset(seed=0)
    env.step(adapter.baseline_policy(env)(env.observation_space.sample() * 0))
    scene = adapter.read_scene(env)
    assert len(scene) >= 1                      # one entry per building
    first = next(iter(scene.values()))
    assert "indoor_dry_bulb_temperature" in first


def test_citylearn_scene_schema_no_heavy_import():
    # runs in rlhvac-ui: scene_schema must not import citylearn
    from rlhvac.adapters import get_adapter
    schema = get_adapter("citylearn").scene_schema()
    assert schema.color_by == "indoor_dry_bulb_temperature"
    assert any(v.name == "indoor_dry_bulb_temperature" for v in schema.variables)
```

- [ ] **Step 3: Run to verify fail**
- env test: `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -k scene -v` → FAIL
- ui test: `conda run -n rlhvac-ui pytest tests/test_citylearn_adapter.py -k scene_schema_no_heavy -v` → FAIL

- [ ] **Step 4: Implement in `rlhvac/adapters/citylearn.py`**

Add a module-level variable list (light dataclass imports only) and the two methods.
Adjust the key list to match the Step-1 probe:
```python
from rlhvac.spec import AdapterManifest, ConfigField, CheckResult, SceneSchema, VarSpec  # extend existing import

_BUILDING_VARS = [
    VarSpec("indoor_dry_bulb_temperature", "Indoor temp", "C", "temperature"),
    VarSpec("indoor_dry_bulb_temperature_cooling_set_point", "Cooling setpoint", "C", "setpoint"),
    VarSpec("indoor_dry_bulb_temperature_heating_set_point", "Heating setpoint", "C", "setpoint"),
    VarSpec("non_shiftable_load", "Load", "kWh", "power"),
    VarSpec("solar_generation", "Solar", "kWh", "power"),
    VarSpec("electrical_storage_soc", "Battery SoC", "", "soc"),
    VarSpec("dhw_storage_soc", "DHW SoC", "", "soc"),
    VarSpec("net_electricity_consumption", "Net electricity", "kWh", "energy"),
    VarSpec("occupant_count", "Occupants", "", "count"),
    VarSpec("carbon_intensity", "Carbon intensity", "kgCO2/kWh", "carbon"),
    VarSpec("electricity_pricing", "Price", "$/kWh", "price"),
]
```
Add to `CityLearnAdapter`:
```python
    @staticmethod
    def scene_schema() -> SceneSchema:
        return SceneSchema(
            color_by="indoor_dry_bulb_temperature",
            color_range=(15.0, 35.0), layout="grid",
            variables=list(_BUILDING_VARS),
        )

    def read_scene(self, env) -> dict:
        base = getattr(self, "_citylearn_env", None) or env.unwrapped
        wanted = {v.name for v in _BUILDING_VARS}
        scene: dict = {}
        for b in base.buildings:
            try:
                obs = b.observations()  # adapt call if 2.3.1 needs args
            except TypeError:
                obs = b.observations(normalize=False, periodic_normalization=False)
            scene[b.name] = {k: float(obs[k]) for k in wanted if k in obs and obs[k] is not None}
        return scene
```

- [ ] **Step 5: Run to verify pass**
- `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -v` → all pass (scene test included)
- `conda run -n rlhvac-ui pytest tests/test_citylearn_adapter.py -v` → the ui-side tests pass (lazy + scene_schema), env-gated ones skip.

- [ ] **Step 6: Commit**
```bash
git add rlhvac/adapters/citylearn.py tests/test_citylearn_adapter.py
git commit -m "feat: CityLearn building-grid scene (schema + read_scene)"
```

---

## Task 4: Cross-env e2e — CityLearn frames carry multiple buildings

**Files:** Modify `tests/test_citylearn_e2e.py`

- [ ] **Step 1: Strengthen the e2e** to assert the scene now has named buildings (not the fallback). Update the assertions block:
```python
    frames = run_store.read_frames(run_dir / "episodes" / "000")
    assert len(frames) > 1
    scene = frames[-1]["scene"]
    assert len(scene) >= 1
    a_building = next(iter(scene.values()))
    assert "indoor_dry_bulb_temperature" in a_building   # real scene, not the {"system": {...}} fallback
    rollup = run_store.read_rollup(run_dir)
    assert rollup and len(rollup[-1]) > 1
```

- [ ] **Step 2: Run to verify pass**
Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_e2e.py -v`
Expected: PASS (spawns into rlhvac-citylearn; frames now contain building-keyed scenes).

- [ ] **Step 3: Commit**
```bash
git add tests/test_citylearn_e2e.py
git commit -m "test: CityLearn e2e asserts building-keyed scene"
```

---

## Task 5: UI smoke (human gate)

**Files:** none

- [ ] **Step 1:** Confirm the suite is green in both envs:
- `conda run -n rlhvac-ui pytest -q`
- `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -q`

- [ ] **Step 2:** Report for the human to verify in the browser: launch `host_ui.py`, Run
tab → `citylearn` → a dataset → small simulation steps → Run; Live tab should show a
**grid of building cells** colored by indoor temperature, hover listing the building's
load/solar/SoC/net-electricity/price/carbon. The agent confirms imports + suite green
and defers the visual confirmation to the human.

---

## Self-Review Notes (author)

- **Frame-driven units** decouple the UI (rlhvac-ui, no citylearn) from CityLearn's
  dataset-dependent building set: the heatmap reads unit names from the run's frames;
  the static `scene_schema()` only provides variable metadata + `color_by`.
- **Lazy rule preserved:** CityLearn `scene_schema()` builds only `VarSpec`/`SceneSchema`
  dataclasses (no citylearn import); the heavy reads are in `read_scene` (runner/citylearn env).
- **Robust read_scene:** tolerates the 2.3.1 `observations()` signature (try/except) and
  only includes keys actually present + non-None (datasets vary in active observations).
- **Backward compatibility:** mock keeps a real scene; default fallback still covers any
  adapter without a scene (e.g. BOPTEST until Plan 4).
- **Scope guard:** BOPTEST adapter + its zone scene is Plan 4; SB3 training is a later spec.
