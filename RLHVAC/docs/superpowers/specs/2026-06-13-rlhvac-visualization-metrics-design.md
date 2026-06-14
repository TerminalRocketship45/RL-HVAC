# RLHVAC Visualization & Metrics Overhaul — Design Spec

**Date:** 2026-06-13
**Status:** Approved for planning
**Builds on:** Phase 0 (skeleton) + Phase 1 (CityLearn adapter)

## 1. Purpose

Turn the single-page baseline UI into a multi-tab research console that (a) shows a
**live, per-simulator schematic visual** of the building/zones with values changing
as an episode runs, (b) surfaces **every variable a simulator exposes** (hover +
graphs), and (c) reports **metrics across episodes** within a run, built to receive
RL training curves later. Also wires in **BOPTEST** as the second real simulator.

The guiding constraint: **these engines expose no building geometry/coordinates**
(CityLearn = independent buildings; BOPTEST/Sinergym = named zones, no layout). The
visual is therefore a faithful **schematic** — units laid out in a grid/diagram,
colored by a thermal variable, hover-revealing all variables — not an architectural
floor plan. Accuracy is bounded by, and driven entirely from, what each simulator
actually reports.

### User-confirmed decisions
- Build the visual + metrics **framework now**; real SB3 training curves come next.
- Target **CityLearn + BOPTEST** concretely (mock as the dependency-free fallback).
- **Plotly inside Streamlit** for interactive heatmaps + hover + graphs.
- **Run = a session of N episodes** (default 1).
- Map colored by **indoor/zone temperature** by default (toggle to comfort).
- "Live" = poll frames ~1×/sec.

### Non-goals
- No true geometric/CFD floor plans (engines don't provide it).
- No SB3 training yet (the framework is built ready for it; training is the next spec).
- No deep export INTO native dashboards (CityLearn UI / BOPTEST dashboard) — those
  remain optional external links.

## 2. The Scene contract (the thing that varies per simulator)

The UI stays generic by rendering **Scenes** that adapters produce. Two new methods
on the adapter, both lazy/per-sim:

```python
@staticmethod
def scene_schema() -> SceneSchema:
    """Static description of the visual for this simulator:
       - units: list[UnitSpec]  (CityLearn->buildings, BOPTEST->zones, mock->1 zone)
         each UnitSpec: name, label, variables: list[VarSpec]
         VarSpec: name, label, unit, kind (temperature|power|energy|soc|price|
                  carbon|comfort|count|other)
       - color_by: variable name used to color the map (default a temperature var)
       - color_range: (min, max) for the color scale
       - layout: 'grid' | 'row' | 'diagram'
       Must NOT import the simulator."""

def read_scene(self, env) -> dict:
    """Per-step snapshot pulled from live env state:
       {unit_name: {variable_name: float|None, ...}, ...}
       Read from named env attributes (e.g. env.buildings[i].<attr>), not the
       flattened observation vector, so values are correctly labeled."""
```

`SceneSchema`, `UnitSpec`, `VarSpec` are new dataclasses in `rlhvac/spec.py`. The
schema is what the Run/Live tabs read to render and to build hover tooltips; the
per-step `read_scene` output is written into the episode's `frames.jsonl`.

Adapters that don't implement a scene (or before one exists) fall back to a trivial
one-unit scene derived from the observation/reward so the UI never breaks.

## 3. Run = session of N episodes (data model)

```
runs/<id>/
  job.json            # adds: episodes:int, episode_steps:int (horizon per episode)
  status.json         # adds: current_episode:int, episodes_total:int
  episodes/
    000/
      frames.jsonl    # one row per step: {step, reward, action, scene:{...}, obs_named:{...}}
      summary.json    # episode totals + KPIs (from adapter.summarize)
    001/ ...
  rollup.jsonl        # one row per finished episode: {episode, total_reward, **kpis}
  logs/
```

- `run_store` gains: episode-dir creation, `append_frame`, `read_frames`,
  `write_episode_summary`, `read_episode_summary`, `append_rollup`, `read_rollup`,
  `list_episodes`.
- The Phase 0 single-episode `metrics.jsonl` API is superseded; the mock-era tests
  migrate to the episode-based layout. Keep `kind`-tagged rows only where still used.

## 4. Runner changes

`runner._run_baseline` becomes an episode loop:
```
for ep in range(job.episodes):
    obs, _ = env.reset(seed=job.seed + ep)
    create episodes/<ep>/
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        scene = adapter.read_scene(env)            # named per-unit values
        append_frame(ep_dir, {step, reward, action, scene, obs_named})
    summary = adapter.summarize(...)               # per-episode KPIs
    write_episode_summary(ep_dir, summary)
    append_rollup(run_dir, {episode: ep, total_reward, **summary})
    update status.current_episode
```
Crash handling and the "scenario passthrough into make()" behavior from prior phases
are preserved. `visual` flag still gates per-step frame richness.

## 5. UI: three tabs (Streamlit + Plotly)

New module set under `rlhvac/ui/`:
- `viz_plotly.py` — pure functions returning Plotly figures:
  - `heatmap_figure(schema, frame)` — units laid out per `layout`, colored by
    `color_by`, hovertext = all variables for that unit.
  - `timeseries_figure(frames, var)` and `multi_timeseries(frames, vars)`.
  - `episode_bar_figure(summary)` — KPIs/metrics as bars (replaces raw JSON).
  - `rollup_curve_figure(rollup, metric)` — per-episode metric across the run.
- `tabs/run_tab.py`, `tabs/live_tab.py`, `tabs/metrics_tab.py` — each a `render(st, ...)`.
- `app.py` becomes a thin shell with `st.tabs(["Run", "Live Simulation", "Metrics"])`.

**Run tab:** sim/scenario/episodes/horizon/seed/visual → launch (existing flow + episodes).

**Live Simulation tab:** select run → select episode (from `list_episodes`, defaults
to latest/active) → animated heatmap (polls `frames.jsonl` while `status` is running),
a step slider when not live, the episode's metric **bar charts**, and per-step
**time-series graphs** for every scene variable. Hover any unit → all its variables.

**Metrics tab:** within the selected run — **reward-per-episode curve** (from
`rollup.jsonl`), KPI trends, and component graphs. Layout/series mirror what the
native tools graph (loads, SoC, temps, setpoints, energy, price, carbon, KPIs).
Structured so SB3 training curves slot in unchanged later.

## 6. Per-simulator Scenes (concrete)

- **mock** — 1 unit ("zone"), variables: temp, setpoint, action; color_by temp.
- **CityLearn** — units = buildings (`env.buildings`), variables from each building's
  named observations: indoor_dry_bulb_temperature (+ setpoint), non_shiftable_load,
  solar_generation, electrical_storage_soc, dhw_storage_soc, net_electricity_consumption,
  occupant_count, carbon_intensity, electricity_pricing; color_by indoor temperature;
  layout grid.
- **BOPTEST** — units = zones of the chosen test case; variables = zone operative
  temperature, heating/cooling setpoints, heating/cooling power, CO₂, energy; color_by
  zone temperature; layout diagram/grid. Exact measurement names resolved against the
  live test case during planning.

## 7. BOPTEST adapter (new simulator)

- New env `rlhvac-boptest`; `boptest-gym` pointed at the hosted service
  `https://api.boptest.net` (no Docker). Adapter wraps `BoptestGymEnv` to the
  Gymnasium single-agent interface behind the existing contract.
- `manifest()` scenarios = a curated set of BOPTEST test cases (e.g.
  `bestest_hydronic_heat_pump`); lazy-import rule preserved (no boptestgym at module top).
- `check()` probes service reachability; `summarize()` reads BOPTEST KPIs; `read_scene`
  maps measurements to zone variables.

## 8. Phasing (each its own plan + review)

1. **Run/episode data model + Scene contract + runner** — proven with mock only.
2. **Plotly visuals + 3-tab UI** (Live + Metrics) — mock + CityLearn fallback scene.
3. **CityLearn Scene** — named building observations + grid heatmap.
4. **BOPTEST adapter + its Scene** — hosted service.

## 9. Testing strategy

- Scene contract tests: every adapter's `scene_schema()` is importable without the
  sim, `read_scene` returns the declared units/variables after a step (sim-env-gated).
- Run-model tests: multi-episode runner writes N episode dirs, frames, summaries, and
  N rollup rows; crash still yields `error`; mock e2e through the spawned runner.
- Viz tests: `viz_plotly` functions are pure — assert figure structure (trace counts,
  colors mapped from `color_by`, hovertext contains all variables) without a browser.
- UI helper tests via a fake-`st`/no-server harness, as established in Phase 0/1.

## 10. Open questions / future
- SB3 training (the next spec) feeds `rollup.jsonl` with training-episode rewards and
  adds loss/exploration curves to the Metrics tab.
- Optional later: export a CityLearn run's CSVs into the native CityLearn UI format;
  post BOPTEST results to its dashboard.
- Sinergym/Energym Scenes when those adapters land.
