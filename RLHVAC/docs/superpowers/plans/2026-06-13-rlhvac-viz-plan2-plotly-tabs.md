# RLHVAC Viz Plan 2 — Plotly Visuals + 3-Tab UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the single-page UI with a 3-tab console (Run / Live Simulation / Metrics) where the Live tab renders an interactive **Plotly schematic heatmap** of the building/zones (colored by temperature, hover shows every variable) that animates as the episode runs, plus per-episode bar charts and per-step time-series; and the Metrics tab shows the cross-episode reward curve and KPI trends. Works against mock (real Scene) and CityLearn (fallback Scene until Plan 3).

**Architecture:** Add `plotly` to the UI env. New `rlhvac/ui/viz_plotly.py` holds **pure figure-builder functions** (unit-tested without a browser by asserting figure structure). New `rlhvac/ui/tabs/` package holds `run_tab`, `live_tab`, `metrics_tab`, each a `render(st, ...)`. `app.py` becomes a thin `st.tabs([...])` shell. The UI obtains a `SceneSchema` per simulator via a helper that calls `adapter.scene_schema()` when present, else `default_scene_schema()`.

**Tech Stack:** Python 3.11, Streamlit, **Plotly**, pandas, the existing rlhvac package.

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `envs/environment-ui.yml` | add `plotly` |
| `rlhvac/ui/scene_access.py` | `get_scene_schema(sim) -> SceneSchema` (adapter or fallback) |
| `rlhvac/ui/viz_plotly.py` | pure Plotly figure builders |
| `rlhvac/ui/tabs/__init__.py` | package marker |
| `rlhvac/ui/tabs/run_tab.py` | launch form |
| `rlhvac/ui/tabs/live_tab.py` | run/episode pickers, heatmap, bars, time-series |
| `rlhvac/ui/tabs/metrics_tab.py` | rollup curve + KPI trends |
| `app.py` | `st.tabs` shell wiring the three tabs |
| `tests/test_viz_plotly.py` | figure-structure tests |
| `tests/test_scene_access.py` | schema-or-fallback test |

`rlhvac/ui/live_view.py` from Plan 1 is superseded by `viz_plotly.py` + `live_tab.py`; delete it and migrate its one test (`tests/test_live_view.py`) into the new tests.

---

## Task 1: Add Plotly to the UI env

**Files:** Modify `envs/environment-ui.yml`

- [ ] **Step 1:** Add `- plotly>=5.18` to the `pip:` list in `envs/environment-ui.yml`.
- [ ] **Step 2:** Install into the existing env:

Run: `conda run -n rlhvac-ui pip install "plotly>=5.18"`
Then: `conda run -n rlhvac-ui python -c "import plotly.graph_objects as go; print('plotly', __import__('plotly').__version__)"`
Expected: prints a version.

- [ ] **Step 3: Commit**
```bash
git add envs/environment-ui.yml
git commit -m "chore: add plotly to rlhvac-ui env"
```

---

## Task 2: Scene-schema access helper

**Files:** Create `rlhvac/ui/scene_access.py`; Test `tests/test_scene_access.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_scene_access.py
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
```

- [ ] **Step 2: Run to verify fail**
Run: `conda run -n rlhvac-ui pytest tests/test_scene_access.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `rlhvac/ui/scene_access.py`**
```python
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
```

- [ ] **Step 4: Run to verify pass** → `conda run -n rlhvac-ui pytest tests/test_scene_access.py -v` PASS.
- [ ] **Step 5: Commit**
```bash
git add rlhvac/ui/scene_access.py tests/test_scene_access.py
git commit -m "feat: scene-schema access with fallback"
```

---

## Task 3: Plotly figure builders (pure)

**Files:** Create `rlhvac/ui/viz_plotly.py`; Test `tests/test_viz_plotly.py`

- [ ] **Step 1: Failing tests**
```python
# tests/test_viz_plotly.py
import math
from rlhvac.spec import SceneSchema, UnitSpec, VarSpec
from rlhvac.ui import viz_plotly as viz


def _schema():
    return SceneSchema(
        units=[
            UnitSpec(name="b0", label="Building 0", variables=[
                VarSpec("temp", "Temp", "C", "temperature"),
                VarSpec("load", "Load", "kW", "power")]),
            UnitSpec(name="b1", label="Building 1", variables=[
                VarSpec("temp", "Temp", "C", "temperature"),
                VarSpec("load", "Load", "kW", "power")]),
        ],
        color_by="temp", color_range=(10.0, 30.0), layout="grid",
    )


def _frame(t0=20.0, t1=25.0):
    return {"step": 3, "reward": -1.0,
            "scene": {"b0": {"temp": t0, "load": 2.0}, "b1": {"temp": t1, "load": 3.0}}}


def test_heatmap_figure_colors_by_color_by_and_sets_range():
    fig = viz.heatmap_figure(_schema(), _frame())
    hm = fig.data[0]
    assert hm.type == "heatmap"
    assert hm.zmin == 10.0 and hm.zmax == 30.0
    # both unit temperatures appear somewhere in the z grid
    flat = [v for row in hm.z for v in row if v is not None]
    assert 20.0 in flat and 25.0 in flat


def test_heatmap_hover_lists_all_variables():
    fig = viz.heatmap_figure(_schema(), _frame())
    text = "".join(str(c) for row in fig.data[0].text for c in row)
    assert "Temp" in text and "Load" in text and "Building 0" in text


def test_variable_timeseries_one_trace_per_unit():
    frames = [_frame(20, 25), _frame(21, 26), _frame(22, 27)]
    fig = viz.variable_timeseries(frames, _schema(), "temp")
    assert len(fig.data) == 2  # b0, b1
    assert list(fig.data[0].y) == [20, 21, 22]


def test_reward_timeseries():
    frames = [{"step": 0, "reward": -3.0}, {"step": 1, "reward": -2.0}]
    fig = viz.reward_timeseries(frames)
    assert list(fig.data[0].y) == [-3.0, -2.0]


def test_episode_bar_figure_from_summary():
    fig = viz.episode_bar_figure({"cost_total": 1.2, "carbon_total": 0.8, "note": "x"})
    bar = fig.data[0]
    assert set(bar.x) == {"cost_total", "carbon_total"}  # non-numeric 'note' dropped


def test_rollup_curve_figure():
    rollup = [{"episode": 0, "total_reward": -10.0}, {"episode": 1, "total_reward": -8.0}]
    fig = viz.rollup_curve_figure(rollup, "total_reward")
    assert list(fig.data[0].x) == [0, 1]
    assert list(fig.data[0].y) == [-10.0, -8.0]
```

- [ ] **Step 2: Run to verify fail** → FAIL (module missing).

- [ ] **Step 3: Implement `rlhvac/ui/viz_plotly.py`**
```python
from __future__ import annotations
import math
from typing import Optional
import plotly.graph_objects as go
from rlhvac.spec import SceneSchema


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n)) if n else 1
    rows = math.ceil(n / cols) if cols else 1
    return rows, cols


def _hover_for_unit(schema: SceneSchema, unit_name: str, values: dict) -> str:
    unit = next((u for u in schema.units if u.name == unit_name), None)
    label = unit.label if unit else unit_name
    lines = [f"<b>{label}</b>"]
    if unit:
        for v in unit.variables:
            val = values.get(v.name)
            shown = "n/a" if val is None else (f"{val:.3g}" if isinstance(val, (int, float)) else val)
            unit_suffix = f" {v.unit}" if v.unit else ""
            lines.append(f"{v.label}: {shown}{unit_suffix}")
    return "<br>".join(lines)


def heatmap_figure(schema: SceneSchema, frame: dict) -> go.Figure:
    scene = (frame or {}).get("scene", {})
    names = [u.name for u in schema.units]
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
                unit = schema.units[idx]
                lr.append(f"{unit.label}<br>{'' if cval is None else f'{cval:.3g}'}")
            else:
                zr.append(None); tr.append(""); lr.append("")
        z.append(zr); text.append(tr); labels.append(lr)
    fig = go.Figure(go.Heatmap(
        z=z, text=text, hoverinfo="text",
        zmin=schema.color_range[0], zmax=schema.color_range[1],
        colorscale="RdYlBu_r", showscale=True,
    ))
    fig.update_traces(texttemplate="%{customdata}", customdata=labels)
    fig.update_layout(yaxis=dict(autorange="reversed", showticklabels=False),
                      xaxis=dict(showticklabels=False),
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Colored by {schema.color_by}")
    return fig


def variable_timeseries(frames: list[dict], schema: SceneSchema, var: str) -> go.Figure:
    steps = [f.get("step") for f in frames]
    fig = go.Figure()
    for u in schema.units:
        ys = [f.get("scene", {}).get(u.name, {}).get(var) for f in frames]
        if any(y is not None for y in ys):
            fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines", name=u.label))
    fig.update_layout(title=var, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def reward_timeseries(frames: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[f.get("step") for f in frames],
                               y=[f.get("reward") for f in frames], mode="lines", name="reward"))
    fig.update_layout(title="Reward", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def episode_bar_figure(summary: dict) -> go.Figure:
    items = [(k, v) for k, v in (summary or {}).items() if isinstance(v, (int, float))]
    fig = go.Figure(go.Bar(x=[k for k, _ in items], y=[v for _, v in items]))
    fig.update_layout(title="Episode metrics", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def rollup_curve_figure(rollup: list[dict], metric: str = "total_reward") -> go.Figure:
    rows = [r for r in rollup if metric in r]
    fig = go.Figure(go.Scatter(x=[r.get("episode") for r in rows],
                               y=[r.get(metric) for r in rows], mode="lines+markers", name=metric))
    fig.update_layout(title=f"{metric} per episode", xaxis_title="episode",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
```

- [ ] **Step 4: Run to verify pass** → `conda run -n rlhvac-ui pytest tests/test_viz_plotly.py -v` PASS. If a Plotly API detail differs (e.g. `customdata`/`texttemplate` on Heatmap), adapt so the asserted structure (z/text/zmin/zmax, trace counts, x/y) holds; the tests are the contract.
- [ ] **Step 5: Commit**
```bash
git add rlhvac/ui/viz_plotly.py tests/test_viz_plotly.py
git commit -m "feat: pure Plotly figure builders for scenes and metrics"
```

---

## Task 4: Run tab

**Files:** Create `rlhvac/ui/tabs/__init__.py`, `rlhvac/ui/tabs/run_tab.py`

- [ ] **Step 1:** Create `rlhvac/ui/tabs/__init__.py`:
```python
"""Streamlit tab renderers."""
```

- [ ] **Step 2:** Create `rlhvac/ui/tabs/run_tab.py` (moves the launch form out of app.py):
```python
from __future__ import annotations
from pathlib import Path
from rlhvac import run_store, launcher
from rlhvac.spec import JobSpec
from rlhvac.adapters import available_sims, get_manifest
from rlhvac.ui.manifest_view import render_config_form

RUNS_DIR = Path("runs")


def render(st) -> None:
    st.subheader("Configure a run")
    sim = st.selectbox("Simulator", available_sims())
    manifest = get_manifest(sim)
    scenario = st.selectbox("Scenario", manifest.scenarios)
    st.caption(f"Runner env: `{manifest.runner_env}`")
    if manifest.dashboard:
        st.markdown(f"[Open native dashboard ↗]({manifest.dashboard})")
    config = render_config_form(st, manifest)
    episodes = st.number_input("Episodes", min_value=1, max_value=50, value=1, step=1)
    visual = st.checkbox("Stream per-step frames (visual)", value=True)
    if st.button("Run baseline", type="primary"):
        job = JobSpec(run_id=run_store.new_run_id(), sim=sim, scenario=scenario,
                      config=config, mode="baseline", algo=None, timesteps=0, seed=7,
                      visual=visual, episodes=int(episodes))
        run_dir = run_store.create_run(RUNS_DIR, job)
        runner_env = None if manifest.runner_env == launcher.UI_ENV else manifest.runner_env
        launcher.spawn(run_dir, runner_env)
        st.session_state["active_run"] = str(run_dir)
        st.success(f"Launched run {job.run_id} — see the Live Simulation tab.")
```

- [ ] **Step 3: Commit**
```bash
git add rlhvac/ui/tabs/__init__.py rlhvac/ui/tabs/run_tab.py
git commit -m "feat: Run tab (launch form + native dashboard link)"
```

---

## Task 5: Live Simulation tab

**Files:** Create `rlhvac/ui/tabs/live_tab.py`; Test `tests/test_live_tab.py`

- [ ] **Step 1: Failing test** (pure helper, no Streamlit server):
```python
# tests/test_live_tab.py
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.tabs.live_tab import list_run_dirs, episode_frame_at


def _job(run_id, episodes=1, length=4):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True, episodes=episodes)


def test_list_run_dirs_newest_first(tmp_path):
    run_store.create_run(tmp_path, _job("a"))
    run_store.create_run(tmp_path, _job("b"))
    dirs = list_run_dirs(tmp_path)
    assert {d.name for d in dirs} == {"a", "b"}


def test_episode_frame_at_returns_named_scene(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r", length=4))
    runner.run(run_dir)
    frame = episode_frame_at(run_dir, episode=0, step_index=2)
    assert frame["step"] == 2
    assert "zone" in frame["scene"]
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Implement `rlhvac/ui/tabs/live_tab.py`**
```python
from __future__ import annotations
from pathlib import Path
from rlhvac import run_store
from rlhvac.ui.scene_access import get_scene_schema
from rlhvac.ui import viz_plotly as viz


def list_run_dirs(base) -> list[Path]:
    base = Path(base)
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and (p / "job.json").exists()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def episode_frame_at(run_dir, episode: int, step_index: int) -> dict:
    frames = run_store.read_frames(run_store.episode_dir(run_dir, episode))
    if not frames:
        return {}
    step_index = max(0, min(step_index, len(frames) - 1))
    return frames[step_index]


def render(st, runs_dir="runs") -> None:
    st.subheader("Live simulation")
    runs = list_run_dirs(runs_dir)
    if not runs:
        st.info("No runs yet — launch one in the Run tab.")
        return
    default = st.session_state.get("active_run")
    names = [d.name for d in runs]
    idx = names.index(Path(default).name) if default and Path(default).name in names else 0
    run_name = st.selectbox("Run", names, index=idx)
    run_dir = next(d for d in runs if d.name == run_name)

    job = run_store.read_job(run_dir)
    status = run_store.read_status(run_dir)
    schema = get_scene_schema(job.sim)
    eps = run_store.list_episodes(run_dir)
    if not eps:
        st.info("Waiting for frames...")
        return
    ep = st.selectbox("Episode", eps, index=len(eps) - 1)
    frames = run_store.read_frames(run_store.episode_dir(run_dir, ep))
    st.caption(f"Status: {status.state} · episode {status.current_episode + 1}/{status.episodes_total}")
    if not frames:
        st.info("Waiting for frames...")
        return

    live = status.state == "running"
    step = len(frames) - 1 if live else st.slider("Step", 0, len(frames) - 1, len(frames) - 1)
    st.plotly_chart(viz.heatmap_figure(schema, frames[step]), use_container_width=True)

    summary = run_store.read_episode_summary(run_store.episode_dir(run_dir, ep))
    if summary:
        st.plotly_chart(viz.episode_bar_figure(summary), use_container_width=True)
    st.plotly_chart(viz.reward_timeseries(frames), use_container_width=True)
    temp_var = schema.color_by
    st.plotly_chart(viz.variable_timeseries(frames, schema, temp_var), use_container_width=True)
```

- [ ] **Step 4: Run to verify pass** → `conda run -n rlhvac-ui pytest tests/test_live_tab.py -v` PASS.
- [ ] **Step 5: Commit**
```bash
git add rlhvac/ui/tabs/live_tab.py tests/test_live_tab.py
git commit -m "feat: Live Simulation tab with animated heatmap and graphs"
```

---

## Task 6: Metrics tab

**Files:** Create `rlhvac/ui/tabs/metrics_tab.py`; Test `tests/test_metrics_tab.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_metrics_tab.py
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.tabs.metrics_tab import rollup_metric_names


def test_rollup_metric_names(tmp_path):
    job = JobSpec(run_id="m", sim="mock", scenario="sine-day",
                  config={"episode_length": 3}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True, episodes=2)
    run_dir = run_store.create_run(tmp_path, job)
    runner.run(run_dir)
    names = rollup_metric_names(run_dir)
    assert "total_reward" in names
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Implement `rlhvac/ui/tabs/metrics_tab.py`**
```python
from __future__ import annotations
from pathlib import Path
from rlhvac import run_store
from rlhvac.ui.tabs.live_tab import list_run_dirs
from rlhvac.ui import viz_plotly as viz


def rollup_metric_names(run_dir) -> list[str]:
    rollup = run_store.read_rollup(run_dir)
    keys: list[str] = []
    for row in rollup:
        for k, v in row.items():
            if k != "episode" and isinstance(v, (int, float)) and k not in keys:
                keys.append(k)
    return keys


def render(st, runs_dir="runs") -> None:
    st.subheader("Metrics across episodes")
    runs = list_run_dirs(runs_dir)
    if not runs:
        st.info("No runs yet.")
        return
    default = st.session_state.get("active_run")
    names = [d.name for d in runs]
    idx = names.index(Path(default).name) if default and Path(default).name in names else 0
    run_name = st.selectbox("Run", names, index=idx, key="metrics_run")
    run_dir = next(d for d in runs if d.name == run_name)

    rollup = run_store.read_rollup(run_dir)
    if not rollup:
        st.info("No completed episodes yet.")
        return
    metrics = rollup_metric_names(run_dir)
    st.caption("Cross-episode trends (training curves slot in here once RL training is added).")
    chosen = st.multiselect("Metrics", metrics, default=[m for m in ["total_reward"] if m in metrics] or metrics[:1])
    for m in chosen:
        st.plotly_chart(viz.rollup_curve_figure(rollup, m), use_container_width=True)
```

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit**
```bash
git add rlhvac/ui/tabs/metrics_tab.py tests/test_metrics_tab.py
git commit -m "feat: Metrics tab with per-episode trend curves"
```

---

## Task 7: app.py shell + remove live_view

**Files:** Modify `app.py`; Delete `rlhvac/ui/live_view.py`, `tests/test_live_view.py`

- [ ] **Step 1: Rewrite `app.py`**
```python
"""RLHVAC Streamlit UI — Run / Live / Metrics tabs."""
from __future__ import annotations
import time
import streamlit as st
from rlhvac import run_store
from rlhvac.ui.tabs import run_tab, live_tab, metrics_tab

st.set_page_config(page_title="RLHVAC", layout="wide")
st.title("RLHVAC — Simulator Control Panel")

run_t, live_t, metrics_t = st.tabs(["Run", "Live Simulation", "Metrics"])
with run_t:
    run_tab.render(st)
with live_t:
    live_tab.render(st)
with metrics_t:
    metrics_tab.render(st)

# auto-refresh while an active run is still in progress
active = st.session_state.get("active_run")
if active:
    status = run_store.read_status(active)
    if status.state in ("queued", "running"):
        time.sleep(1.0)
        st.rerun()
```

- [ ] **Step 2: Delete superseded files**
```bash
git rm rlhvac/ui/live_view.py tests/test_live_view.py
```

- [ ] **Step 3: Confirm no references to live_view remain**
Run: `conda run -n rlhvac-ui python -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
And grep: there must be no remaining `live_view` import anywhere in `rlhvac/` or `tests/`.

- [ ] **Step 4: Full suite + app import probe**
Run: `conda run -n rlhvac-ui pytest -q` → all pass.
Run: `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -q` → pass (untouched).

- [ ] **Step 5: Commit**
```bash
git add app.py
git commit -m "feat: 3-tab app shell; remove superseded live_view"
```

---

## Task 8: Manual UI smoke (human-verifiable)

**Files:** none

- [ ] **Step 1:** Launch and verify the tabs render and the heatmap animates:
Run: `conda run -n rlhvac-ui python host_ui.py`
Expected (manual): Run tab launches a mock run with N episodes; Live tab shows the
zone heatmap coloring by temperature with hover listing temp+setpoint, a reward
time-series, and an episode-metrics bar chart; Metrics tab shows total_reward per
episode. CityLearn (if its env is up) runs via the fallback scene (single "system"
cell) — rich CityLearn scene arrives in Plan 3. Note any rendering errors for fixing.

This task is a human gate; the agent reports that the app imports and the suite is green, and defers the visual confirmation to the user.

---

## Self-Review Notes (author)

- **Pure viz functions** are browser-free and unit-tested by figure structure (trace counts, zmin/zmax, x/y, hovertext) — the Plotly API specifics are adapted to keep those assertions true.
- **Fallback path:** `scene_access.get_scene_schema` returns the default schema for adapters without a scene, so CityLearn renders (single cell) without Plan 3; no import of the sim in the UI env.
- **live_view superseded:** deleted with its test; grep gate ensures no dangling imports.
- **Type consistency:** tabs read frames (`step`/`reward`/`scene`), rollup (`episode`/`total_reward`+KPIs), and `SceneSchema` (`color_by`/`color_range`/`units[].variables[]`) exactly as written by the Plan 1 runner and the adapters.
- **Scope guard:** real CityLearn/BOPTEST Scenes are Plans 3-4; SB3 training (feeding the Metrics tab) is a later spec.
