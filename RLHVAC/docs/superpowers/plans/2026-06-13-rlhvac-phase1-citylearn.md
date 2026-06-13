# RLHVAC Phase 1 — CityLearn Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a real CityLearn simulator behind the existing `SimAdapter` contract, running in its own `rlhvac-citylearn` conda env, and prove the cross-env subprocess launch end-to-end (UI env spawns runner in the CityLearn env).

**Architecture:** Reuse everything from Phase 0 unchanged. Add `rlhvac/adapters/citylearn.py` (lazy-imports `citylearn` only inside `make()`/`check()`), register it, and create the `rlhvac-citylearn` conda env. CityLearn runs in `central_agent=True` mode wrapped by `StableBaselines3Wrapper` so it presents a single-agent Gymnasium interface. The launcher's existing `conda run -n <env>` path carries the runner into the CityLearn env.

**Tech Stack:** CityLearn v2 (gymnasium), Python 3.11, the existing rlhvac package.

**IMPORTANT for the implementer:** CityLearn's exact import paths/return shapes must be validated against the *installed* package. After Task 1 installs it, run quick `python -c` probes to confirm symbol names before writing adapter code, and adapt the code below if the installed version differs. The code here reflects the documented v2 API but the installed version is the source of truth.

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `envs/environment-citylearn.yml` | `rlhvac-citylearn` conda env (Python 3.11 + CityLearn + rlhvac + pytest) |
| `rlhvac/adapters/citylearn.py` | CityLearn adapter (lazy import; make/baseline/summarize) |
| `rlhvac/adapters/__init__.py` | Add `"citylearn"` to `REGISTRY` |
| `tests/test_citylearn_adapter.py` | Manifest/lazy-import tests (run in `rlhvac-ui`) + availability-gated env tests (run in `rlhvac-citylearn`) |
| `tests/test_citylearn_e2e.py` | Cross-env spawned-runner integration test |
| `docs/setup/citylearn.md` | Setup notes |
| `README.md` | Mention CityLearn |

**Contract reminder (unchanged from Phase 0):** `manifest()` and `check()` must not heavy-import `citylearn` at module top. Put `import citylearn...` lines *inside* the methods that need them. The dataset list in `manifest()` is a hardcoded curated constant — do NOT call `citylearn.data.DataSet.get_names()` there (that would import citylearn in the UI env).

---

## Task 1: CityLearn conda env

**Files:**
- Create: `envs/environment-citylearn.yml`

- [ ] **Step 1: Create `envs/environment-citylearn.yml`**

```yaml
name: rlhvac-citylearn
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - CityLearn
      - -e ..[dev]
```

- [ ] **Step 2: Create the env**

Run: `conda env create -f envs/environment-citylearn.yml`
Expected: completes without error. If CityLearn fails to resolve on Python 3.11, retry with `python=3.10` in the yml and report the change.

- [ ] **Step 3: Probe the installed API (confirm symbols before coding)**

Run:
```bash
conda run -n rlhvac-citylearn python -c "from citylearn.citylearn import CityLearnEnv; from citylearn.wrappers import StableBaselines3Wrapper; from citylearn.data import DataSet; print('names:', DataSet.get_names()[:5]); print('ok')"
```
Expected: prints a list of dataset names and `ok`. If any import path differs (e.g. `get_names` vs `get_dataset_names`, or wrapper module path), NOTE the correct names — you will use them in Task 3. Report findings.

- [ ] **Step 4: Commit**

```bash
git add envs/environment-citylearn.yml
git commit -m "chore: rlhvac-citylearn conda env"
```

---

## Task 2: CityLearn manifest + lazy-import contract

**Files:**
- Create: `rlhvac/adapters/citylearn.py` (manifest + check only in this task)
- Modify: `rlhvac/adapters/__init__.py`
- Test: `tests/test_citylearn_adapter.py`

- [ ] **Step 1: Write the failing test (runs in the UI env — must NOT need citylearn installed)**

```python
# tests/test_citylearn_adapter.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_adapter.py -v`
Expected: FAIL (`citylearn` not in REGISTRY / module missing).

- [ ] **Step 3: Create `rlhvac/adapters/citylearn.py` (manifest + check)**

```python
"""CityLearn adapter. Heavy `citylearn` imports happen ONLY inside make()/check()
so this module is importable in the rlhvac-ui env where citylearn is absent."""
from __future__ import annotations
from typing import Any, Callable
from rlhvac.spec import AdapterManifest, ConfigField, CheckResult

# Curated built-in datasets (static — must not import citylearn to list them).
_DATASETS = [
    "citylearn_challenge_2022_phase_1",
    "citylearn_challenge_2022_phase_2",
    "citylearn_challenge_2023_phase_1",
    "baeda_3dem",
]


class CityLearnAdapter:
    name = "citylearn"

    @staticmethod
    def manifest() -> AdapterManifest:
        return AdapterManifest(
            name="citylearn",
            scenarios=list(_DATASETS),
            config_schema=[
                ConfigField(name="simulation_steps", type="int",
                            label="Simulation steps (hours)", default=168, min=24, max=8760),
                ConfigField(name="seed", type="int", label="Random seed", default=0, min=0, max=99999),
            ],
            runner_env="rlhvac-citylearn",
            requirements=["pip install CityLearn (in rlhvac-citylearn env)"],
            dashboard=None,
        )

    @staticmethod
    def check() -> CheckResult:
        try:
            import citylearn  # noqa: F401
            return CheckResult(available=True, hint="")
        except Exception:
            return CheckResult(available=False,
                               hint="Create the env: conda env create -f envs/environment-citylearn.yml")
```

- [ ] **Step 4: Register it in `rlhvac/adapters/__init__.py`**

Add the citylearn entry to `REGISTRY`:
```python
REGISTRY: dict[str, tuple[str, str]] = {
    "mock": ("rlhvac.adapters.mock", "MockAdapter"),
    "citylearn": ("rlhvac.adapters.citylearn", "CityLearnAdapter"),
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_adapter.py -v`
Expected: PASS (3 passed). Also run the whole UI-env suite to confirm no regression: `conda run -n rlhvac-ui pytest -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add rlhvac/adapters/citylearn.py rlhvac/adapters/__init__.py tests/test_citylearn_adapter.py
git commit -m "feat: CityLearn manifest and lazy registry entry"
```

---

## Task 3: CityLearn make / baseline / summarize

**Files:**
- Modify: `rlhvac/adapters/citylearn.py`
- Test: `tests/test_citylearn_adapter.py` (add availability-gated tests run in the citylearn env)

- [ ] **Step 1: Add the failing env-level test**

Append to `tests/test_citylearn_adapter.py`:
```python
import importlib.util
import pytest

citylearn_installed = importlib.util.find_spec("citylearn") is not None
requires_citylearn = pytest.mark.skipif(not citylearn_installed, reason="citylearn not in this env")


@requires_citylearn
def test_citylearn_make_and_short_baseline_episode():
    import numpy as np
    from rlhvac.adapters.citylearn import CityLearnAdapter
    adapter = CityLearnAdapter()
    env = adapter.make({"scenario": "citylearn_challenge_2022_phase_2",
                        "simulation_steps": 48, "seed": 0})
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(policy(obs))
        steps += 1
        done = term or trunc
        assert steps <= 60  # guard against runaway
    assert 1 < steps <= 48
    summary = adapter.summarize([])
    assert isinstance(summary, dict) and len(summary) > 0
```

NOTE: the runner passes `scenario` separately; the adapter's `make(config)` must accept the dataset name. The runner already calls `adapter.make(job.config)`. So `make` must read the scenario from config. Update the runner contract expectation: ensure the scenario reaches `make`. See Step 3 — `make` reads `config["scenario"]` with the manifest's first dataset as fallback. The runner already includes `scenario` in the JobSpec but passes only `job.config` to `make`. **Therefore Task 3 Step 4 also patches the runner to merge scenario into the config it passes.**

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -k baseline -v`
Expected: FAIL (`make`/`baseline_policy`/`summarize` not implemented).

- [ ] **Step 3: Implement make/baseline_policy/summarize in `rlhvac/adapters/citylearn.py`**

Add these methods to `CityLearnAdapter` (adjust import paths to match the Task 1 Step 3 probe if they differed):
```python
    def make(self, config: dict):
        import numpy as np  # noqa: F401
        from citylearn.citylearn import CityLearnEnv
        from citylearn.wrappers import StableBaselines3Wrapper

        scenario = config.get("scenario") or _DATASETS[1]
        steps = int(config.get("simulation_steps", 168))
        seed = int(config.get("seed", 0))
        base = CityLearnEnv(
            schema=scenario,
            central_agent=True,
            simulation_start_time_step=0,
            simulation_end_time_step=max(1, steps - 1),
            random_seed=seed,
        )
        self._citylearn_env = base  # stashed for summarize()
        return StableBaselines3Wrapper(base)

    def baseline_policy(self, env) -> Callable[[Any], Any]:
        import numpy as np
        zeros = np.zeros(env.action_space.shape, dtype=np.float32)

        def policy(obs):
            return zeros.copy()

        return policy

    def summarize(self, episode) -> dict:
        env = getattr(self, "_citylearn_env", None)
        if env is None:
            return {"note": "no environment available"}
        df = env.evaluate()
        # district-level KPIs: rows where level == 'district', keyed by cost_function
        out: dict = {}
        try:
            district = df[df["level"] == "district"]
            for _, row in district.iterrows():
                val = row["value"]
                if val is not None:
                    out[str(row["cost_function"])] = float(val)
        except Exception as exc:  # evaluate() schema can vary by version
            out = {"evaluate_error": str(exc)[:200]}
        return out
```

- [ ] **Step 4: Patch the runner so the scenario reaches `make()`**

In `rlhvac/runner.py`, in `_run_baseline`, change the `make` call so the chosen scenario is available to adapters that need it:
```python
def _run_baseline(run_dir: Path, job) -> None:
    adapter = get_adapter(job.sim)
    env = adapter.make({**job.config, "scenario": job.scenario})
    ...
```
(Leave the rest of `_run_baseline` unchanged.) The mock ignores the extra `scenario` key, so this is backward-compatible.

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n rlhvac-citylearn pytest tests/test_citylearn_adapter.py -v`
Expected: PASS including the baseline episode test (first run downloads dataset; allow time).
Also confirm mock still works in both envs: `conda run -n rlhvac-ui pytest tests/test_runner.py -q` → pass.

- [ ] **Step 6: Commit**

```bash
git add rlhvac/adapters/citylearn.py rlhvac/runner.py tests/test_citylearn_adapter.py
git commit -m "feat: CityLearn make/baseline/summarize and scenario passthrough"
```

---

## Task 4: Cross-env spawned-runner integration

**Files:**
- Test: `tests/test_citylearn_e2e.py`

This proves the headline architecture claim: the UI env spawns a runner in the CityLearn env via `conda run`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citylearn_e2e.py
import importlib.util
import pytest
from rlhvac.spec import JobSpec
from rlhvac import run_store, launcher

citylearn_installed = importlib.util.find_spec("citylearn") is not None


@pytest.mark.skipif(citylearn_installed, reason="run from an env WITHOUT citylearn (the UI env)")
def test_ui_env_spawns_citylearn_runner(tmp_path):
    # From rlhvac-ui: spawn the runner into rlhvac-citylearn via conda run.
    job = JobSpec(run_id="cl-e2e", sim="citylearn",
                  scenario="citylearn_challenge_2022_phase_2",
                  config={"simulation_steps": 48, "seed": 0},
                  mode="baseline", algo=None, timesteps=0, seed=0, visual=True)
    run_dir = run_store.create_run(tmp_path, job)
    proc = launcher.spawn(run_dir, runner_env="rlhvac-citylearn")
    proc.wait(timeout=600)  # first run downloads the dataset
    status = run_store.read_status(run_dir)
    assert status.state == "done", f"state={status.state} error={status.error}"
    steps = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    assert len(steps) > 1
    summary = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "summary"]
    assert summary and len(summary[-1]) > 1
```

- [ ] **Step 2: Run test to verify it fails (or errors) before the adapter works**

Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_e2e.py -v`
Expected: initially FAIL if anything in the chain is broken. Once Tasks 1-3 are correct it should PASS. Treat green as success.

- [ ] **Step 3: No new implementation** — verification gate. If it fails, read `runs/cl-e2e/logs/runner.log` (captured by the Phase 0 crash handler) to diagnose, fix the relevant adapter/runner code, and re-run.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_citylearn_e2e.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_citylearn_e2e.py
git commit -m "test: cross-env CityLearn spawned-runner integration"
```

---

## Task 5: Setup docs + README + UI smoke

**Files:**
- Create: `docs/setup/citylearn.md`
- Modify: `README.md`

- [ ] **Step 1: Create `docs/setup/citylearn.md`**

```markdown
# CityLearn setup

CityLearn runs in its own conda env (dependency isolation from the UI env).

```bash
conda env create -f envs/environment-citylearn.yml
```

First baseline run downloads the selected dataset (cached afterwards).
In the UI, pick **citylearn** in the sidebar, choose a dataset scenario,
set the simulation-steps horizon (default 168 = one week), and Run baseline.
Mode is central single-agent (StableBaselines3Wrapper); baseline policy is
zero-action (no distributed-energy-resource control). KPIs come from
`CityLearnEnv.evaluate()` (district level).
```

- [ ] **Step 2: Add a line to `README.md`** under the phase note:

```markdown
## Phase 1: CityLearn

CityLearn (multi-building demand response) runs in the `rlhvac-citylearn` env.
See `docs/setup/citylearn.md`. The UI spawns its runner in that env via conda run.
```

- [ ] **Step 3: UI smoke check (manifest renders without importing citylearn)**

Run:
```bash
conda run -n rlhvac-ui python -c "from rlhvac.adapters import get_manifest; from rlhvac.ui.manifest_view import render_config_form
class F:
    def number_input(self,label,value=None,**k):
        assert type(value) is type(k.get('min_value')) is type(k.get('max_value')); return value
    def selectbox(self,label,options=None): return (options or [None])[0]
print(render_config_form(F(), get_manifest('citylearn')))"
```
Expected: prints a config dict with no error (proves the CityLearn form renders in the UI env).

- [ ] **Step 4: Commit**

```bash
git add docs/setup/citylearn.md README.md
git commit -m "docs: CityLearn setup and README"
```

---

## Self-Review Notes (author)

- **Lazy-import rule:** enforced by `test_citylearn_module_top_has_no_heavy_import` (Task 2) — the UI env can read the manifest without citylearn.
- **Cross-env claim:** proven by Task 4 (UI env → conda run → citylearn env).
- **Contract unchanged:** `summarize(episode)` signature preserved; CityLearn KPIs sourced via stashed `self._citylearn_env` set in `make()` (same adapter instance is reused across make→policy→summarize in `runner._run_baseline`).
- **Scenario passthrough:** Task 3 Step 4 patches the runner to merge `job.scenario` into the config dict given to `make()`; mock ignores the extra key (backward-compatible) — verify mock tests still pass.
- **Snappy runs:** `simulation_steps` config caps the horizon (default 168) so baseline runs finish quickly; first run pays a one-time dataset download.
- **API drift risk:** Task 1 Step 3 probes the installed package; implementer adapts import paths (`StableBaselines3Wrapper`, `DataSet.get_names`, `evaluate()` columns) if the version differs.
