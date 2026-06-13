# RLHVAC Phase 0 — Skeleton & Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the process-isolated adapter skeleton and prove the full UI → runner → adapter pipe end-to-end with a deterministic **mock** simulator, with zero real-simulator dependencies.

**Architecture:** A Streamlit UI process writes a `job.json` into a per-run directory and spawns a runner subprocess (via `conda run -n <env>`, or the current interpreter for the mock). The runner loads an adapter through a registry, runs a baseline episode against a Gymnasium env, and streams `status.json` + `metrics.jsonl` back. The UI tails those files for live charts. Phase 0 ships only the `mock` adapter so everything runs in the `rlhvac-ui` env.

**Tech Stack:** Python 3.11, Gymnasium, Streamlit, pytest. (Stable-Baselines3 and real simulators are later phases.)

---

## File Structure

| File | Responsibility |
| ---- | -------------- |
| `pyproject.toml` | Package metadata, deps, pytest config |
| `envs/environment-ui.yml` | `rlhvac-ui` conda env definition |
| `.gitignore` | Ignore `runs/`, caches |
| `rlhvac/__init__.py` | Package marker |
| `rlhvac/spec.py` | Data schemas: `ConfigField`, `AdapterManifest`, `CheckResult`, `JobSpec`, `RunStatus` + (de)serialization |
| `rlhvac/adapters/base.py` | `SimAdapter` Protocol (the adapter contract) |
| `rlhvac/adapters/mock.py` | Deterministic mock adapter (gym env + baseline + summary) |
| `rlhvac/adapters/__init__.py` | `REGISTRY` + lazy `get_manifest()` / `get_adapter()` |
| `rlhvac/run_store.py` | UI-side create/read of run directories |
| `rlhvac/runner.py` | Subprocess entry: baseline episode loop, status/metrics writing, crash handling |
| `rlhvac/launcher.py` | Build + spawn the runner command |
| `rlhvac/ui/manifest_view.py` | Render sidebar picker + dynamic config form from a manifest |
| `rlhvac/ui/live_view.py` | Render status + live charts from a run dir |
| `app.py` | Streamlit entry wiring UI + launcher + run_store |
| `host_ui.py` | One-command localhost launch of Streamlit |
| `tests/...` | Tests per task |

**Key design rule (enforced in every adapter):** adapter modules must NOT import their simulator at module top level. `manifest()` and `check()` must work in the `rlhvac-ui` env without the sim installed; the heavy import happens lazily inside `make()`. The mock has no heavy import but follows the same shape.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `envs/environment-ui.yml`
- Create: `rlhvac/__init__.py`
- Create: `rlhvac/adapters/__init__.py` (empty placeholder, filled in Task 4)
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "rlhvac"
version = "0.0.1"
description = "Upper-level UI wrapper over building-energy RL simulators"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=0.29.1",
    "numpy>=1.26",
    "streamlit>=1.33",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["rlhvac", "rlhvac.adapters", "rlhvac.ui"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
runs/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 3: Create `envs/environment-ui.yml`**

```yaml
name: rlhvac-ui
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - -e ..[dev]
```

- [ ] **Step 4: Create package markers**

`rlhvac/__init__.py`:
```python
"""RLHVAC: upper-level UI wrapper over building-energy RL simulators."""
```

`rlhvac/adapters/__init__.py`:
```python
"""Adapter registry (populated in Task 4)."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Create the conda env and install**

Run:
```bash
conda env create -f envs/environment-ui.yml
conda run -n rlhvac-ui python -c "import gymnasium, streamlit; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore envs/environment-ui.yml rlhvac/__init__.py rlhvac/adapters/__init__.py tests/__init__.py
git commit -m "chore: project scaffold and rlhvac-ui env"
```

---

## Task 2: Data schemas (`spec.py`)

**Files:**
- Create: `rlhvac/spec.py`
- Test: `tests/test_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spec.py
import json
from rlhvac.spec import JobSpec, RunStatus, ConfigField, AdapterManifest, CheckResult


def test_jobspec_roundtrip():
    job = JobSpec(
        run_id="r1", sim="mock", scenario="sine-day",
        config={"episode_length": 24}, mode="baseline",
        algo=None, timesteps=0, seed=7, visual=True,
    )
    restored = JobSpec.from_json(json.loads(job.to_json()))
    assert restored == job


def test_runstatus_defaults_and_roundtrip():
    s = RunStatus(state="running", progress=0.5, pid=123)
    restored = RunStatus.from_json(json.loads(s.to_json()))
    assert restored.state == "running"
    assert restored.error is None
    assert restored == s


def test_manifest_holds_config_schema():
    m = AdapterManifest(
        name="mock",
        scenarios=["sine-day"],
        config_schema=[ConfigField(name="episode_length", type="int", label="Episode length", default=24)],
        runner_env="rlhvac-ui",
        requirements=[],
        dashboard=None,
    )
    assert m.config_schema[0].type == "int"


def test_checkresult_shape():
    c = CheckResult(available=False, hint="Install EnergyPlus")
    assert c.available is False and "EnergyPlus" in c.hint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.spec'`

- [ ] **Step 3: Write `rlhvac/spec.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

FieldType = Literal["number", "int", "bool", "select", "text"]


@dataclass
class ConfigField:
    name: str
    type: FieldType
    label: str
    default: Any
    options: Optional[list[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class AdapterManifest:
    name: str
    scenarios: list[str]
    config_schema: list[ConfigField]
    runner_env: str
    requirements: list[str] = field(default_factory=list)
    dashboard: Optional[str] = None


@dataclass
class CheckResult:
    available: bool
    hint: str = ""


@dataclass
class JobSpec:
    run_id: str
    sim: str
    scenario: str
    config: dict
    mode: Literal["baseline", "train"]
    algo: Optional[str]
    timesteps: int
    seed: int
    visual: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: dict) -> "JobSpec":
        return cls(**data)


@dataclass
class RunStatus:
    state: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    pid: Optional[int] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: dict) -> "RunStatus":
        return cls(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_spec.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/spec.py tests/test_spec.py
git commit -m "feat: run/job/manifest data schemas"
```

---

## Task 3: Adapter contract (`adapters/base.py`)

**Files:**
- Create: `rlhvac/adapters/base.py`
- Test: `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
from rlhvac.adapters.base import SimAdapter


def test_protocol_is_runtime_checkable():
    class Incomplete:
        name = "x"
    assert not isinstance(Incomplete(), SimAdapter)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.adapters.base'`

- [ ] **Step 3: Write `rlhvac/adapters/base.py`**

```python
from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable
import gymnasium as gym
from rlhvac.spec import AdapterManifest, CheckResult


@runtime_checkable
class SimAdapter(Protocol):
    name: str

    @staticmethod
    def manifest() -> AdapterManifest:
        """Static metadata for the UI. MUST NOT import the simulator."""
        ...

    @staticmethod
    def check() -> CheckResult:
        """Probe whether this simulator can run on this machine. Lightweight."""
        ...

    def make(self, config: dict) -> gym.Env:
        """Return a Gymnasium-compatible env. Heavy sim import happens here."""
        ...

    def baseline_policy(self, env: gym.Env) -> Callable[[Any], Any]:
        """Default controller used by 'Run baseline'."""
        ...

    def summarize(self, episode: list[dict]) -> dict:
        """Compute sim-specific KPIs from the recorded episode steps."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_base.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/adapters/base.py tests/test_base.py
git commit -m "feat: SimAdapter protocol contract"
```

---

## Task 4: Mock adapter + registry

**Files:**
- Create: `rlhvac/adapters/mock.py`
- Modify: `rlhvac/adapters/__init__.py`
- Test: `tests/test_mock_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mock_adapter.py
from rlhvac.adapters import get_manifest, get_adapter
from rlhvac.adapters.base import SimAdapter


def test_mock_registered_and_conforms():
    adapter = get_adapter("mock")
    assert isinstance(adapter, SimAdapter)
    assert adapter.name == "mock"


def test_mock_manifest_has_scenarios_without_heavy_import():
    m = get_manifest("mock")
    assert "sine-day" in m.scenarios
    assert m.runner_env == "rlhvac-ui"


def test_mock_episode_is_deterministic():
    adapter = get_adapter("mock")
    env = adapter.make({"episode_length": 5})
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=7)
    rewards = []
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert len(rewards) == 5
    summary = adapter.summarize([{"reward": r} for r in rewards])
    assert "episode_reward" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_mock_adapter.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_manifest'`

- [ ] **Step 3: Write `rlhvac/adapters/mock.py`**

```python
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rlhvac.spec import AdapterManifest, ConfigField, CheckResult


class _MockEnv(gym.Env):
    """A trivial deterministic 'thermostat': drive temperature to a setpoint."""

    def __init__(self, episode_length: int = 24, setpoint: float = 21.0):
        self.episode_length = int(episode_length)
        self.setpoint = float(setpoint)
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self._t = 0
        self._temp = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._temp = 15.0  # fixed start → deterministic
        return np.array([self._temp], dtype=np.float32), {}

    def step(self, action):
        a = float(np.clip(action, -5.0, 5.0))
        self._temp += a
        self._t += 1
        reward = -abs(self._temp - self.setpoint)
        terminated = False
        truncated = self._t >= self.episode_length
        obs = np.array([self._temp], dtype=np.float32)
        return obs, reward, terminated, truncated, {"temp": self._temp}


class MockAdapter:
    name = "mock"

    @staticmethod
    def manifest() -> AdapterManifest:
        return AdapterManifest(
            name="mock",
            scenarios=["sine-day", "flat-day"],
            config_schema=[
                ConfigField(name="episode_length", type="int", label="Episode length", default=24, min=1, max=168),
                ConfigField(name="setpoint", type="number", label="Setpoint (C)", default=21.0, min=10, max=30),
            ],
            runner_env="rlhvac-ui",
            requirements=[],
            dashboard=None,
        )

    @staticmethod
    def check() -> CheckResult:
        return CheckResult(available=True, hint="")

    def make(self, config: dict) -> gym.Env:
        return _MockEnv(
            episode_length=config.get("episode_length", 24),
            setpoint=config.get("setpoint", 21.0),
        )

    def baseline_policy(self, env: gym.Env) -> Callable[[Any], Any]:
        setpoint = getattr(env.unwrapped, "setpoint", 21.0)

        def policy(obs):
            # proportional controller toward setpoint
            error = setpoint - float(obs[0])
            return np.array([np.clip(error, -5.0, 5.0)], dtype=np.float32)

        return policy

    def summarize(self, episode: list[dict]) -> dict:
        rewards = [step["reward"] for step in episode]
        return {
            "episode_reward": float(sum(rewards)),
            "steps": len(rewards),
            "final_reward": float(rewards[-1]) if rewards else 0.0,
        }
```

- [ ] **Step 4: Write the registry in `rlhvac/adapters/__init__.py`**

```python
"""Adapter registry with lazy module loading.

REGISTRY maps a simulator name to the dotted module path of its adapter class.
Real-simulator modules import their heavy deps only inside `make()`, so calling
`get_manifest()`/`get_adapter()` is safe in the rlhvac-ui env.
"""
from __future__ import annotations
import importlib
from rlhvac.spec import AdapterManifest

REGISTRY: dict[str, tuple[str, str]] = {
    # name: (module, class)
    "mock": ("rlhvac.adapters.mock", "MockAdapter"),
}


def get_adapter(name: str):
    module_path, class_name = REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def get_manifest(name: str) -> AdapterManifest:
    return get_adapter(name).manifest()


def available_sims() -> list[str]:
    return list(REGISTRY.keys())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_mock_adapter.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add rlhvac/adapters/mock.py rlhvac/adapters/__init__.py tests/test_mock_adapter.py
git commit -m "feat: mock adapter and lazy registry"
```

---

## Task 5: Run store (`run_store.py`)

**Files:**
- Create: `rlhvac/run_store.py`
- Test: `tests/test_run_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_store.py
import json
from rlhvac.spec import JobSpec
from rlhvac import run_store


def _job(run_id):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day", config={},
                   mode="baseline", algo=None, timesteps=0, seed=7, visual=True)


def test_create_run_writes_job_and_queued_status(tmp_path):
    job = _job("r1")
    run_dir = run_store.create_run(tmp_path, job)
    assert (run_dir / "job.json").exists()
    assert run_store.read_status(run_dir).state == "queued"


def test_append_and_read_metrics(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2"))
    run_store.append_metric(run_dir, {"step": 0, "reward": -3.0})
    run_store.append_metric(run_dir, {"step": 1, "reward": -2.0})
    metrics = run_store.read_metrics(run_dir)
    assert [m["step"] for m in metrics] == [0, 1]


def test_new_run_id_is_unique():
    assert run_store.new_run_id() != run_store.new_run_id()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_run_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.run_store'`

- [ ] **Step 3: Write `rlhvac/run_store.py`**

```python
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path
from rlhvac.spec import JobSpec, RunStatus


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def create_run(base_dir: Path, job: JobSpec) -> Path:
    run_dir = Path(base_dir) / job.run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(job.to_json())
    write_status(run_dir, RunStatus(state="queued"))
    (run_dir / "metrics.jsonl").touch()
    return run_dir


def write_status(run_dir: Path, status: RunStatus) -> None:
    (Path(run_dir) / "status.json").write_text(status.to_json())


def read_status(run_dir: Path) -> RunStatus:
    data = json.loads((Path(run_dir) / "status.json").read_text())
    return RunStatus.from_json(data)


def read_job(run_dir: Path) -> JobSpec:
    data = json.loads((Path(run_dir) / "job.json").read_text())
    return JobSpec.from_json(data)


def append_metric(run_dir: Path, metric: dict) -> None:
    with open(Path(run_dir) / "metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\n")


def read_metrics(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_run_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/run_store.py tests/test_run_store.py
git commit -m "feat: file-based run store"
```

---

## Task 6: Runner (`runner.py`)

**Files:**
- Create: `rlhvac/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner


def _job(run_id, episode_length=5):
    return JobSpec(run_id=run_id, sim="mock", scenario="sine-day",
                   config={"episode_length": episode_length}, mode="baseline",
                   algo=None, timesteps=0, seed=7, visual=True)


def test_runner_baseline_completes_and_writes_metrics(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r1", episode_length=5))
    runner.run(run_dir)
    assert run_store.read_status(run_dir).state == "done"
    metrics = run_store.read_metrics(run_dir)
    step_metrics = [m for m in metrics if m.get("kind") == "step"]
    assert len(step_metrics) == 5


def test_runner_writes_summary_on_done(tmp_path):
    run_dir = run_store.create_run(tmp_path, _job("r2", episode_length=3))
    runner.run(run_dir)
    metrics = run_store.read_metrics(run_dir)
    summary = [m for m in metrics if m.get("kind") == "summary"][-1]
    assert "episode_reward" in summary


def test_runner_records_error_on_bad_sim(tmp_path):
    bad = JobSpec(run_id="r3", sim="does-not-exist", scenario="x", config={},
                  mode="baseline", algo=None, timesteps=0, seed=1, visual=True)
    run_dir = run_store.create_run(tmp_path, bad)
    runner.run(run_dir)
    status = run_store.read_status(run_dir)
    assert status.state == "error"
    assert status.error  # traceback captured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.runner'`

- [ ] **Step 3: Write `rlhvac/runner.py`**

```python
from __future__ import annotations
import argparse
import os
import traceback
from pathlib import Path
from rlhvac import run_store
from rlhvac.spec import RunStatus
from rlhvac.adapters import get_adapter


def _run_baseline(run_dir: Path, job) -> None:
    adapter = get_adapter(job.sim)
    env = adapter.make(job.config)
    policy = adapter.baseline_policy(env)
    obs, _ = env.reset(seed=job.seed)
    episode: list[dict] = []
    step = 0
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        episode.append({"reward": float(reward), "info": {k: float(v) for k, v in info.items()
                                                          if isinstance(v, (int, float))}})
        if job.visual:
            run_store.append_metric(run_dir, {
                "kind": "step", "step": step, "reward": float(reward),
                **{k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
            })
        step += 1
        done = term or trunc
    summary = adapter.summarize(episode)
    run_store.append_metric(run_dir, {"kind": "summary", **summary})


def run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    job = run_store.read_job(run_dir)
    run_store.write_status(run_dir, RunStatus(state="running", pid=os.getpid()))
    try:
        if job.mode == "baseline":
            _run_baseline(run_dir, job)
        else:
            raise NotImplementedError(f"mode '{job.mode}' arrives in a later phase")
        run_store.write_status(run_dir, RunStatus(state="done", progress=1.0, pid=os.getpid()))
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "logs" / "runner.log").write_text(tb)
        run_store.write_status(run_dir, RunStatus(state="error", pid=os.getpid(), error=tb[-2000:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="path to job.json")
    args = parser.parse_args()
    run(Path(args.spec).parent)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_runner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/runner.py tests/test_runner.py
git commit -m "feat: baseline runner with metrics streaming and crash capture"
```

---

## Task 7: Launcher (`launcher.py`)

**Files:**
- Create: `rlhvac/launcher.py`
- Test: `tests/test_launcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_launcher.py
import sys
from rlhvac import launcher


def test_command_uses_current_interpreter_when_env_is_none():
    cmd = launcher.build_command("/runs/r1", runner_env=None)
    assert cmd[0] == sys.executable
    assert "rlhvac.runner" in cmd
    assert cmd[-1].endswith("job.json")


def test_command_uses_conda_run_for_named_env():
    cmd = launcher.build_command("/runs/r1", runner_env="rlhvac-sinergym")
    assert cmd[:4] == ["conda", "run", "-n", "rlhvac-sinergym"]
    assert "rlhvac.runner" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_launcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.launcher'`

- [ ] **Step 3: Write `rlhvac/launcher.py`**

```python
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional

UI_ENV = "rlhvac-ui"


def build_command(run_dir, runner_env: Optional[str]) -> list[str]:
    """Build the runner invocation.

    runner_env=None  -> use the current interpreter (used for the mock, whose
                        runner_env is rlhvac-ui = the UI env itself).
    runner_env="X"   -> `conda run -n X python -m rlhvac.runner ...`
    """
    spec_path = str(Path(run_dir) / "job.json")
    if runner_env is None or runner_env == UI_ENV:
        return [sys.executable, "-m", "rlhvac.runner", "--spec", spec_path]
    return ["conda", "run", "-n", runner_env, "python", "-m", "rlhvac.runner", "--spec", spec_path]


def spawn(run_dir, runner_env: Optional[str]) -> subprocess.Popen:
    cmd = build_command(run_dir, runner_env)
    log = open(Path(run_dir) / "logs" / "spawn.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_launcher.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/launcher.py tests/test_launcher.py
git commit -m "feat: runner launcher (current-interp or conda-run)"
```

---

## Task 8: End-to-end integration test (UI-less pipe)

**Files:**
- Test: `tests/test_e2e_pipe.py`

This proves create_run → spawn → runner → metrics works as separate processes
before wiring Streamlit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_pipe.py
import time
from rlhvac.spec import JobSpec
from rlhvac import run_store, launcher


def test_spawned_runner_completes(tmp_path):
    job = JobSpec(run_id="e2e1", sim="mock", scenario="sine-day",
                  config={"episode_length": 4}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True)
    run_dir = run_store.create_run(tmp_path, job)
    proc = launcher.spawn(run_dir, runner_env=None)
    proc.wait(timeout=60)
    assert run_store.read_status(run_dir).state == "done"
    steps = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    assert len(steps) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_e2e_pipe.py -v`
Expected: FAIL (the test asserts behavior that only passes once spawn works end-to-end; if any wiring is wrong it fails here). If everything from Tasks 1-7 is correct it may already PASS — that is acceptable; treat a green run as success.

- [ ] **Step 3: No new implementation**

This task is a verification gate. If it fails, fix the wiring in the relevant prior module (most often `launcher.build_command` or the `__main__` block in `runner.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_e2e_pipe.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_pipe.py
git commit -m "test: end-to-end spawned runner pipe"
```

---

## Task 9: UI helper — manifest view

**Files:**
- Create: `rlhvac/ui/__init__.py`
- Create: `rlhvac/ui/manifest_view.py`
- Test: `tests/test_manifest_view.py`

The form-building logic is pure (returns a config dict from raw inputs) so it can
be unit-tested without a running Streamlit server.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_view.py
from rlhvac.spec import AdapterManifest, ConfigField
from rlhvac.ui.manifest_view import default_config, coerce_config


def _manifest():
    return AdapterManifest(
        name="mock", scenarios=["sine-day"],
        config_schema=[
            ConfigField(name="episode_length", type="int", label="Len", default=24),
            ConfigField(name="setpoint", type="number", label="SP", default=21.0),
        ],
        runner_env="rlhvac-ui",
    )


def test_default_config_from_schema():
    assert default_config(_manifest()) == {"episode_length": 24, "setpoint": 21.0}


def test_coerce_casts_types():
    raw = {"episode_length": "30", "setpoint": "22.5"}
    out = coerce_config(_manifest(), raw)
    assert out == {"episode_length": 30, "setpoint": 22.5}
    assert isinstance(out["episode_length"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_manifest_view.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.ui.manifest_view'`

- [ ] **Step 3: Write the modules**

`rlhvac/ui/__init__.py`:
```python
"""Streamlit UI helpers."""
```

`rlhvac/ui/manifest_view.py`:
```python
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


def render_config_form(st, manifest: AdapterManifest) -> dict:
    """Render Streamlit widgets for each field; return a coerced config dict.

    `st` is the streamlit module (passed in so this stays import-light/testable)."""
    raw: dict = {}
    for f in manifest.config_schema:
        if f.type == "int":
            raw[f.name] = st.number_input(f.label, value=int(f.default), step=1,
                                          min_value=f.min, max_value=f.max)
        elif f.type == "number":
            raw[f.name] = st.number_input(f.label, value=float(f.default),
                                          min_value=f.min, max_value=f.max)
        elif f.type == "bool":
            raw[f.name] = st.checkbox(f.label, value=bool(f.default))
        elif f.type == "select":
            raw[f.name] = st.selectbox(f.label, options=f.options or [])
        else:
            raw[f.name] = st.text_input(f.label, value=str(f.default))
    return coerce_config(manifest, raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_manifest_view.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add rlhvac/ui/__init__.py rlhvac/ui/manifest_view.py tests/test_manifest_view.py
git commit -m "feat: manifest-driven config form helpers"
```

---

## Task 10: UI helper — live view + Streamlit app + host script

**Files:**
- Create: `rlhvac/ui/live_view.py`
- Create: `app.py`
- Create: `host_ui.py`
- Test: `tests/test_live_view.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_view.py
from rlhvac.spec import JobSpec
from rlhvac import run_store, runner
from rlhvac.ui.live_view import metrics_dataframe


def test_metrics_dataframe_has_step_rows(tmp_path):
    job = JobSpec(run_id="lv1", sim="mock", scenario="sine-day",
                  config={"episode_length": 4}, mode="baseline",
                  algo=None, timesteps=0, seed=7, visual=True)
    run_dir = run_store.create_run(tmp_path, job)
    runner.run(run_dir)
    df = metrics_dataframe(run_dir)
    assert list(df["step"]) == [0, 1, 2, 3]
    assert "reward" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n rlhvac-ui pytest tests/test_live_view.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rlhvac.ui.live_view'`

- [ ] **Step 3: Write `rlhvac/ui/live_view.py`**

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
from rlhvac import run_store


def metrics_dataframe(run_dir) -> pd.DataFrame:
    rows = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    return pd.DataFrame(rows)


def render_live(st, run_dir: Path) -> None:
    status = run_store.read_status(run_dir)
    st.write(f"**Status:** {status.state}")
    if status.state == "error":
        st.error(status.error or "Run failed")
        return
    df = metrics_dataframe(run_dir)
    if df.empty:
        st.info("Waiting for metrics…")
        return
    if "reward" in df:
        st.line_chart(df.set_index("step")["reward"])
    if "temp" in df:
        st.line_chart(df.set_index("step")["temp"])
    summary = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "summary"]
    if summary:
        st.json(summary[-1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n rlhvac-ui pytest tests/test_live_view.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write `app.py`**

```python
"""RLHVAC Streamlit UI — Phase 0 shell."""
from __future__ import annotations
import time
from pathlib import Path
import streamlit as st
from rlhvac import run_store, launcher
from rlhvac.spec import JobSpec
from rlhvac.adapters import available_sims, get_manifest
from rlhvac.ui.manifest_view import render_config_form
from rlhvac.ui.live_view import render_live

RUNS_DIR = Path("runs")

st.set_page_config(page_title="RLHVAC", layout="wide")
st.title("RLHVAC — Simulator Control Panel")

with st.sidebar:
    st.header("Simulator")
    sim = st.selectbox("Choose a simulator", available_sims())
    manifest = get_manifest(sim)
    scenario = st.selectbox("Scenario", manifest.scenarios)
    st.caption(f"Runner env: `{manifest.runner_env}`")

st.subheader("Configuration")
config = render_config_form(st, manifest)
visual = st.checkbox("Stream per-step metrics (visual)", value=True)

if st.button("Run baseline", type="primary"):
    job = JobSpec(run_id=run_store.new_run_id(), sim=sim, scenario=scenario,
                  config=config, mode="baseline", algo=None, timesteps=0,
                  seed=7, visual=visual)
    run_dir = run_store.create_run(RUNS_DIR, job)
    runner_env = None if manifest.runner_env == launcher.UI_ENV else manifest.runner_env
    launcher.spawn(run_dir, runner_env)
    st.session_state["active_run"] = str(run_dir)

if "active_run" in st.session_state:
    st.subheader("Live")
    run_dir = Path(st.session_state["active_run"])
    placeholder = st.empty()
    for _ in range(600):  # ~10 min cap of 1s polls
        with placeholder.container():
            render_live(st, run_dir)
        if run_store.read_status(run_dir).state in ("done", "error"):
            break
        time.sleep(1)
```

- [ ] **Step 6: Write `host_ui.py`**

```python
"""Launch the RLHVAC UI on localhost: `python host_ui.py`."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                    "--server.address", "localhost", "--server.port", "8501"])
```

- [ ] **Step 7: Manually verify the UI runs end-to-end**

Run: `conda run -n rlhvac-ui python host_ui.py`
Expected: browser opens at `http://localhost:8501`; selecting **mock** → **Run baseline** shows a live reward curve that converges and a summary JSON with `episode_reward`. Stop with Ctrl-C.

- [ ] **Step 8: Commit**

```bash
git add rlhvac/ui/live_view.py app.py host_ui.py tests/test_live_view.py
git commit -m "feat: live view, Streamlit app shell, and host script"
```

---

## Task 11: Full suite + README quickstart

**Files:**
- Create: `README.md`
- Test: (run full suite)

- [ ] **Step 1: Run the whole test suite**

Run: `conda run -n rlhvac-ui pytest -v`
Expected: PASS — all tests from Tasks 2–10 green.

- [ ] **Step 2: Write `README.md`**

```markdown
# RLHVAC

Upper-level Streamlit UI over building-energy RL simulators
(CityLearn, BOPTEST, Sinergym, Energym). Process-isolated adapters:
the UI never imports a simulator — it spawns a runner in that simulator's
own conda env and reads results from a per-run directory.

## Phase 0 (this milestone): skeleton + mock simulator

### Setup
```bash
conda env create -f envs/environment-ui.yml
```

### Run the UI
```bash
conda run -n rlhvac-ui python host_ui.py
# open http://localhost:8501 → pick "mock" → Run baseline
```

### Test
```bash
conda run -n rlhvac-ui pytest -v
```

## Architecture
See `docs/superpowers/specs/2026-06-13-rlhvac-simulator-ui-design.md`.
Real simulators arrive in later phases (each its own conda env + adapter).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README quickstart for Phase 0"
```

---

## Self-Review Notes (completed by author)

**Spec coverage (Phase 0 scope only):**
- Process-isolated architecture (§3) → Tasks 6, 7, 8 (runner + launcher + e2e).
- Adapter interface (§4) → Task 3 (`SimAdapter`), Task 4 (mock conforms).
- Runner protocol & run dir (§5) → Tasks 5, 6 (`run_store`, `runner`; job/status/metrics/logs).
- UI design (§6) → Tasks 9, 10 (manifest-driven form, live charts, sidebar picker, host script).
- Testing strategy (§10) → contract test (Task 4), runner protocol tests (Task 6), mock-driven integration (Task 8). Real-sim smoke tests are out of Phase 0 scope by design.
- Phases 1–4 (§9) deliberately deferred to their own plans.

**Lazy-import rule** (UI env can't import real sims) is stated in File Structure and encoded in the registry (Task 4) — real adapters added in Phase 1 must import their sim only inside `make()`.

**Type consistency:** `JobSpec`/`RunStatus`/`AdapterManifest`/`ConfigField`/`CheckResult` field names are identical across spec.py, adapters, run_store, runner, launcher, and UI. `runner.run(run_dir)` and `runner.main()` (parses `--spec` → parent dir) are consistent with `launcher.build_command` passing `job.json`. Metric `kind` values (`step`, `summary`) match between runner (writer) and live_view/tests (readers).
