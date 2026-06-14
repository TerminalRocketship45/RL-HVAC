# RLHVAC Viz Plan 1 — Run/Episode Model + Scene Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Introduce the per-simulator **Scene** contract and the **run = N episodes** data model, update the runner to write per-step frames + per-episode summaries + a run-level rollup, and keep mock + CityLearn + the existing UI working — all proven with the mock simulator.

**Architecture:** Reuse Phase 0/1. Add Scene dataclasses to `spec.py`; add `scene_schema()`/`read_scene()` to the adapter contract with a **default fallback** for adapters that don't implement them yet (so CityLearn keeps running). Restructure run directories to `episodes/NNN/frames.jsonl` + `episodes/NNN/summary.json` + `rollup.jsonl`. The runner loops episodes. The existing `live_view`/`app` are minimally updated to read the new layout so the app stays runnable; the full 3-tab Plotly UI is Plan 2.

**Tech Stack:** Python 3.11, existing rlhvac package (no new deps in this plan).

---

## File Structure

| File | Change |
| ---- | ------ |
| `rlhvac/spec.py` | Add `VarSpec`, `UnitSpec`, `SceneSchema`; add `episodes` to `JobSpec`; add `current_episode`/`episodes_total` to `RunStatus` |
| `rlhvac/adapters/base.py` | Add `scene_schema()`/`read_scene()` to Protocol; add `default_scene_schema()` + `default_read_scene()` fallbacks |
| `rlhvac/adapters/mock.py` | Implement `scene_schema()` + `read_scene()` |
| `rlhvac/run_store.py` | Episode/frame/rollup functions |
| `rlhvac/runner.py` | Episode loop writing frames/summary/rollup + status |
| `rlhvac/ui/live_view.py` | Read latest episode's frames + rollup (keep app working) |
| `app.py` | Pass `episodes` into the JobSpec |
| `tests/*` | Migrate Phase 0 metrics-based tests to the episode model |

**Contract note:** the Scene `read_scene` reads NAMED values from live env state. The mock reads `env.unwrapped.setpoint`/temp. Adapters without a scene use the fallbacks so nothing breaks.

---

## Task 1: Scene dataclasses + JobSpec/RunStatus fields

**Files:**
- Modify: `rlhvac/spec.py`
- Test: `tests/test_spec.py`

- [ ] **Step 1: Add failing tests** to `tests/test_spec.py`:

```python
def test_scene_schema_dataclasses():
    from rlhvac.spec import VarSpec, UnitSpec, SceneSchema
    schema = SceneSchema(
        units=[UnitSpec(name="z0", label="Zone 0",
                        variables=[VarSpec(name="temp", label="Temp", unit="C", kind="temperature")])],
        color_by="temp", color_range=(10.0, 30.0), layout="grid",
    )
    assert schema.units[0].variables[0].kind == "temperature"
    assert schema.color_range == (10.0, 30.0)


def test_jobspec_has_episodes_default():
    from rlhvac.spec import JobSpec
    import json
    job = JobSpec(run_id="r", sim="mock", scenario="s", config={}, mode="baseline",
                  algo=None, timesteps=0, seed=0, visual=True)
    assert job.episodes == 1
    restored = JobSpec.from_json(json.loads(job.to_json()))
    assert restored.episodes == 1


def test_runstatus_has_episode_progress_fields():
    from rlhvac.spec import RunStatus
    s = RunStatus(state="running", current_episode=2, episodes_total=5)
    assert s.current_episode == 2 and s.episodes_total == 5
```

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n rlhvac-ui pytest tests/test_spec.py -v`
Expected: FAIL (ImportError VarSpec / unexpected keyword `episodes`).

- [ ] **Step 3: Implement in `rlhvac/spec.py`**

Add near the top (after existing imports), new dataclasses:
```python
VarKind = Literal["temperature", "power", "energy", "soc", "price", "carbon",
                  "comfort", "count", "setpoint", "other"]


@dataclass
class VarSpec:
    name: str
    label: str
    unit: str = ""
    kind: VarKind = "other"


@dataclass
class UnitSpec:
    name: str
    label: str
    variables: list[VarSpec] = field(default_factory=list)


@dataclass
class SceneSchema:
    units: list[UnitSpec]
    color_by: str
    color_range: tuple[float, float] = (0.0, 1.0)
    layout: Literal["grid", "row", "diagram"] = "grid"
```

Add `episodes: int = 1` as the LAST field of `JobSpec` (after `visual`). Add to
`RunStatus` two new fields after `error`: `current_episode: int = 0` and
`episodes_total: int = 1`.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n rlhvac-ui pytest tests/test_spec.py -v`
Expected: PASS (all, including the 3 new).

- [ ] **Step 5: Commit**

```bash
git add rlhvac/spec.py tests/test_spec.py
git commit -m "feat: Scene dataclasses and episode fields on Job/RunStatus"
```

---

## Task 2: Scene contract on the adapter + mock implementation + fallbacks

**Files:**
- Modify: `rlhvac/adapters/base.py`
- Modify: `rlhvac/adapters/mock.py`
- Test: `tests/test_mock_adapter.py`

- [ ] **Step 1: Add failing tests** to `tests/test_mock_adapter.py`:

```python
def test_mock_scene_schema():
    from rlhvac.adapters import get_adapter
    schema = get_adapter("mock").scene_schema()
    assert schema.color_by == "temp"
    assert len(schema.units) == 1
    var_names = {v.name for v in schema.units[0].variables}
    assert {"temp", "setpoint"}.issubset(var_names)


def test_mock_read_scene_after_step():
    from rlhvac.adapters import get_adapter
    adapter = get_adapter("mock")
    env = adapter.make({"episode_length": 5, "setpoint": 21.0})
    obs, _ = env.reset(seed=0)
    adapter.baseline_policy(env)  # no-op, ensures interface
    env.step(env.action_space.sample())
    scene = adapter.read_scene(env)
    assert "zone" in scene
    assert "temp" in scene["zone"] and "setpoint" in scene["zone"]
    assert scene["zone"]["setpoint"] == 21.0


def test_default_scene_fallback_helpers():
    from rlhvac.adapters.base import default_scene_schema, default_read_scene
    schema = default_scene_schema()
    assert len(schema.units) == 1
    frame = default_read_scene(reward=-1.5)
    assert "reward" in frame[schema.units[0].name]
```

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n rlhvac-ui pytest tests/test_mock_adapter.py -v`
Expected: FAIL (no `scene_schema`/`read_scene`/fallbacks).

- [ ] **Step 3: Extend the Protocol + add fallbacks in `rlhvac/adapters/base.py`**

Add to the `SimAdapter` Protocol (after `summarize`):
```python
    @staticmethod
    def scene_schema() -> "SceneSchema":
        """Static schema describing the per-simulator visual. No heavy import."""
        ...

    def read_scene(self, env) -> dict:
        """Per-step {unit_name: {var_name: value}} pulled from live env state."""
        ...
```
Add the import `from rlhvac.spec import AdapterManifest, CheckResult, SceneSchema, UnitSpec, VarSpec`
and these module-level fallbacks:
```python
def default_scene_schema() -> SceneSchema:
    return SceneSchema(
        units=[UnitSpec(name="system", label="System",
                        variables=[VarSpec(name="reward", label="Reward", kind="other")])],
        color_by="reward", color_range=(-10.0, 0.0), layout="grid",
    )


def default_read_scene(reward: float = 0.0) -> dict:
    return {"system": {"reward": float(reward)}}
```

- [ ] **Step 4: Implement scene methods on `MockAdapter` in `rlhvac/adapters/mock.py`**

Add to `MockAdapter` (import `SceneSchema, UnitSpec, VarSpec` from `rlhvac.spec` at top — these are light dataclasses, no sim dependency):
```python
    @staticmethod
    def scene_schema() -> SceneSchema:
        return SceneSchema(
            units=[UnitSpec(name="zone", label="Zone", variables=[
                VarSpec(name="temp", label="Temperature", unit="C", kind="temperature"),
                VarSpec(name="setpoint", label="Setpoint", unit="C", kind="setpoint"),
            ])],
            color_by="temp", color_range=(10.0, 30.0), layout="grid",
        )

    def read_scene(self, env) -> dict:
        u = env.unwrapped
        return {"zone": {"temp": float(u._temp), "setpoint": float(u.setpoint)}}
```

- [ ] **Step 5: Run to verify pass**

Run: `conda run -n rlhvac-ui pytest tests/test_mock_adapter.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rlhvac/adapters/base.py rlhvac/adapters/mock.py tests/test_mock_adapter.py
git commit -m "feat: Scene contract on adapter with mock impl and fallbacks"
```

---

## Task 3: run_store episode/frame/rollup functions

**Files:**
- Modify: `rlhvac/run_store.py`
- Test: `tests/test_run_store.py`

- [ ] **Step 1: Replace the metrics tests** in `tests/test_run_store.py` with episode-model tests (delete `test_append_and_read_metrics`, add):

```python
def test_episode_dir_and_frames(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2"))
    ep = run_store.create_episode(run_dir, 0)
    run_store.append_frame(ep, {"step": 0, "reward": -3.0, "scene": {"zone": {"temp": 15.0}}})
    run_store.append_frame(ep, {"step": 1, "reward": -2.0, "scene": {"zone": {"temp": 16.0}}})
    frames = run_store.read_frames(ep)
    assert [f["step"] for f in frames] == [0, 1]
    assert run_store.list_episodes(run_dir) == [0]


def test_episode_summary_and_rollup(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r3"))
    ep = run_store.create_episode(run_dir, 0)
    run_store.write_episode_summary(ep, {"episode_reward": -10.0})
    assert run_store.read_episode_summary(ep)["episode_reward"] == -10.0
    run_store.append_rollup(run_dir, {"episode": 0, "total_reward": -10.0})
    run_store.append_rollup(run_dir, {"episode": 1, "total_reward": -8.0})
    rollup = run_store.read_rollup(run_dir)
    assert [r["total_reward"] for r in rollup] == [-10.0, -8.0]
```

Keep `test_create_run_writes_job_and_queued_status` and `test_new_run_id_is_unique`.

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n rlhvac-ui pytest tests/test_run_store.py -v`
Expected: FAIL (missing functions).

- [ ] **Step 3: Implement in `rlhvac/run_store.py`**

Remove `append_metric`/`read_metrics` (superseded) and `(run_dir/"metrics.jsonl").touch()` from `create_run`. Add:
```python
def _ep_name(ep: int) -> str:
    return f"{ep:03d}"


def create_episode(run_dir, ep: int) -> Path:
    ep_dir = Path(run_dir) / "episodes" / _ep_name(ep)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "frames.jsonl").touch()
    return ep_dir


def list_episodes(run_dir) -> list[int]:
    base = Path(run_dir) / "episodes"
    if not base.exists():
        return []
    return sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit())


def append_frame(ep_dir, frame: dict) -> None:
    with open(Path(ep_dir) / "frames.jsonl", "a") as f:
        f.write(json.dumps(frame) + "\n")


def read_frames(ep_dir) -> list[dict]:
    path = Path(ep_dir) / "frames.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def write_episode_summary(ep_dir, summary: dict) -> None:
    (Path(ep_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


def read_episode_summary(ep_dir) -> dict:
    path = Path(ep_dir) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def append_rollup(run_dir, row: dict) -> None:
    with open(Path(run_dir) / "rollup.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def read_rollup(run_dir) -> list[dict]:
    path = Path(run_dir) / "rollup.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n rlhvac-ui pytest tests/test_run_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlhvac/run_store.py tests/test_run_store.py
git commit -m "feat: episode/frame/rollup run-store layout"
```

---

## Task 4: Runner episode loop

**Files:**
- Modify: `rlhvac/runner.py`
- Test: `tests/test_runner.py`, `tests/test_e2e_pipe.py`

- [ ] **Step 1: Rewrite the runner tests** in `tests/test_runner.py` for the episode model:

```python
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner


def _job(run_id, episode_length=5, episodes=1):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": episode_length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True, episodes=episodes)


def test_runner_writes_frames_per_episode(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r1", episode_length=5, episodes=2))
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "done"
    assert run_store.list_episodes(run_dir) == [0, 1]
    frames0 = run_store.read_frames(run_dir / "episodes" / "000")
    assert len(frames0) == 5
    assert "scene" in frames0[0] and "zone" in frames0[0]["scene"]


def test_runner_writes_rollup_row_per_episode(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2", episode_length=3, episodes=3))
    runner.run(run_dir)
    rollup = run_store.read_rollup(run_dir)
    assert [r["episode"] for r in rollup] == [0, 1, 2]
    assert all("total_reward" in r for r in rollup)


def test_runner_records_error_on_bad_sim(tmp_path):
    bad = JobSpec(run_id="r3", sim="nope", scenario="x", config={}, mode="baseline",
                  algo=None, timesteps=0, seed=1, visual=True, episodes=1)
    run_dir = run_store.create_run(tmp_path, bad)
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "error"


def test_runner_records_error_on_corrupt_job(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r4"))
    (run_dir / "job.json").write_text("{ not json")
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "error"
```

Also update `tests/test_e2e_pipe.py` to assert frames in episode 0:
```python
    proc.wait(timeout=60)
    assert run_store.read_status(run_dir).state == "done"
    frames = run_store.read_frames(run_dir / "episodes" / "000")
    assert len(frames) == 4
```
(keep the rest of that test; the job uses `config={"episode_length": 4}`.)

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n rlhvac-ui pytest tests/test_runner.py tests/test_e2e_pipe.py -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite `rlhvac/runner.py`**

Replace `_run_baseline` with an episode loop and a scene helper:
```python
from rlhvac.adapters.base import default_read_scene


def _read_scene(adapter, env, reward) -> dict:
    fn = getattr(adapter, "read_scene", None)
    if callable(fn):
        try:
            return fn(env)
        except Exception:
            pass
    return default_read_scene(reward=reward)


def _run_baseline(run_dir: Path, job) -> None:
    adapter = get_adapter(job.sim)
    env = adapter.make({**job.config, "scenario": job.scenario})
    policy = adapter.baseline_policy(env)
    for ep in range(max(1, job.episodes)):
        ep_dir = run_store.create_episode(run_dir, ep)
        obs, _ = env.reset(seed=job.seed + ep)
        episode: list[dict] = []
        step = 0
        done = False
        while not done:
            action = policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            episode.append({"reward": float(reward)})
            if job.visual:
                run_store.append_frame(ep_dir, {
                    "step": step, "reward": float(reward),
                    "action": [float(a) for a in np.atleast_1d(action)],
                    "scene": _read_scene(adapter, env, reward),
                })
            step += 1
            done = term or trunc
        summary = adapter.summarize(episode)
        run_store.write_episode_summary(ep_dir, summary)
        run_store.append_rollup(run_dir, {"episode": ep,
                                          "total_reward": float(sum(s["reward"] for s in episode)),
                                          **summary})
        run_store.write_status(run_dir, RunStatus(state="running", pid=os.getpid(),
                                                  current_episode=ep, episodes_total=job.episodes))
```
Add `import numpy as np` at the top of `runner.py`. Leave `run()`'s try/except and the
`done` status write intact (it already wraps the whole body from the Phase 0 fix). The
final `RunStatus(state="done", ...)` should also set `episodes_total=job.episodes`.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n rlhvac-ui pytest tests/test_runner.py tests/test_e2e_pipe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlhvac/runner.py tests/test_runner.py tests/test_e2e_pipe.py
git commit -m "feat: runner episode loop writing frames, summaries, rollup"
```

---

## Task 5: Keep the UI working on the new layout

**Files:**
- Modify: `rlhvac/ui/live_view.py`
- Modify: `app.py`
- Test: `tests/test_live_view.py`

- [ ] **Step 1: Update the live_view test** in `tests/test_live_view.py`:

```python
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.live_view import latest_episode_frames


def test_latest_episode_frames(tmp_path):
    job = JobSpec(run_id="lv1", sim="mock", scenario="sine-day",
                  config={"episode_length": 4}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True, episodes=1)
    run_dir = run_store.create_run(tmp_path, job)
    runner.run(run_dir)
    df = latest_episode_frames(run_dir)
    assert list(df["step"]) == [0, 1, 2, 3]
    assert "reward" in df.columns
```

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n rlhvac-ui pytest tests/test_live_view.py -v`
Expected: FAIL (`latest_episode_frames` missing).

- [ ] **Step 3: Rewrite `rlhvac/ui/live_view.py`** to read the episode layout:

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rlhvac import run_store


def latest_episode_frames(run_dir) -> pd.DataFrame:
    eps = run_store.list_episodes(run_dir)
    if not eps:
        return pd.DataFrame()
    ep_dir = Path(run_dir) / "episodes" / f"{eps[-1]:03d}"
    return pd.DataFrame(run_store.read_frames(ep_dir))


def render_live(st, run_dir: Path) -> None:
    status = run_store.read_status(run_dir)
    st.write(f"**Status:** {status.state}  ·  episode {status.current_episode + 1}/{status.episodes_total}")
    if status.state == "error":
        st.error(status.error or "Run failed")
        return
    df = latest_episode_frames(run_dir)
    if df.empty:
        st.info("Waiting for frames...")
        return
    if "step" in df.columns and "reward" in df.columns:
        st.line_chart(df.set_index("step")["reward"])
    rollup = run_store.read_rollup(run_dir)
    if rollup:
        st.write("**Episode rewards**")
        st.bar_chart(pd.DataFrame(rollup).set_index("episode")["total_reward"])
```

- [ ] **Step 4: Update `app.py`** to pass `episodes`. Add an episodes input near the visual toggle and include it in the JobSpec:

```python
episodes = st.number_input("Episodes", min_value=1, max_value=50, value=1, step=1)
```
and in the `JobSpec(...)` call add `episodes=int(episodes),`.

- [ ] **Step 5: Run to verify pass + full suite**

Run: `conda run -n rlhvac-ui pytest -q`
Expected: PASS (whole UI-env suite green). Also confirm the app parses:
`conda run -n rlhvac-ui python -c "import ast; ast.parse(open('app.py').read()); print('ok')"`

- [ ] **Step 6: Commit**

```bash
git add rlhvac/ui/live_view.py app.py tests/test_live_view.py
git commit -m "feat: UI reads episode/rollup layout; episodes input"
```

---

## Task 6: CityLearn still runs under the new model (cross-env check)

**Files:** none (verification gate)

- [ ] **Step 1: Run the cross-env CityLearn e2e** (it uses the spawned runner, now episode-based; CityLearn has no scene yet so the runner uses the fallback):

Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_e2e.py -v`
Expected: the test currently asserts the OLD metrics layout. UPDATE it to the new layout:
```python
    proc.wait(timeout=600)
    status = run_store.read_status(run_dir)
    assert status.state == "done", f"state={status.state} error={status.error}"
    frames = run_store.read_frames(run_dir / "episodes" / "000")
    assert len(frames) > 1
    rollup = run_store.read_rollup(run_dir)
    assert rollup and len(rollup[-1]) > 1
```
Then run again; expect PASS (CityLearn baseline runs an episode, fallback scene written,
rollup has its KPIs).

- [ ] **Step 2: Confirm the citylearn-env adapter test still passes**

Run: `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -q`
Expected: PASS (Plan 1 didn't touch CityLearn's adapter; its KPIs flow into the rollup).

- [ ] **Step 3: Commit**

```bash
git add tests/test_citylearn_e2e.py
git commit -m "test: CityLearn e2e on episode/rollup layout"
```

---

## Self-Review Notes (author)

- **Backward compatibility:** the Scene fallback (`default_read_scene`) keeps CityLearn
  (no scene yet) runnable; Task 6 proves it cross-env. mock implements a real scene.
- **Data-model migration:** Phase 0 `metrics.jsonl` is fully replaced by
  `episodes/NNN/frames.jsonl` + `rollup.jsonl`; every test that referenced metrics is
  migrated (Tasks 3-6). No `append_metric`/`read_metrics`/`metrics_dataframe` callers remain.
- **Type consistency:** frame rows use keys `step`/`reward`/`action`/`scene`; rollup rows
  use `episode`/`total_reward` + KPI keys; `RunStatus` carries `current_episode`/`episodes_total`.
  The runner (writer) and live_view (reader) agree on these.
- **Scope guard:** this plan is data-model + contract + runner + minimal UI keep-alive
  only. Plotly visuals and the 3-tab UI are Plan 2; CityLearn/BOPTEST real Scenes are
  Plans 3-4.
