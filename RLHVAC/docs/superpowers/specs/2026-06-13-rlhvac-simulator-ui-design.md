# RLHVAC Simulator UI — Design Spec

**Date:** 2026-06-13
**Status:** Approved for planning
**Author:** Rohan (with Claude)

## 1. Purpose

A Python upper-level wrapper + Streamlit UI that sits on top of multiple
building-energy RL simulators, so a researcher (and the user's mentor) can pick a
simulator, configure a scenario, run it, and watch a baseline episode — and later
an RL training run — without learning each simulator's bespoke API.

This is a **starting-point research tool**, not a production product. The goal is
to get working quickly with the simulators and provide a clean, extensible base
for future RL-in-HVAC research once a specific topic is chosen.

### Target simulators

| Simulator | Interface | Runtime requirement | Status target |
| --------- | --------- | ------------------- | ------------- |
| CityLearn | Gymnasium | pip only | Phase 1 |
| BOPTEST   | Gym via `BoptestGymEnv` | **Hosted service** `api.boptest.net` (no Docker) or local Docker | Phase 1 |
| Sinergym  | Gymnasium | EnergyPlus ~23.1.0 on `PYTHONPATH` | Phase 2 |
| Energym   | Custom `make/step/get_output` (not gym) | Docker (FMU images) | Phase 3 |

### Non-goals (YAGNI)

- No multi-user / auth / cloud deployment (localhost only).
- No custom RL algorithm implementations — use Stable-Baselines3.
- No attempt to share one Python env across all simulators (see §3).
- No hyperparameter search, experiment database, or comparison dashboards in v1.

## 2. Constraints (current machine)

- Windows 11, Anaconda, base Python 3.13 (too new for these libs → dedicated envs).
- **No Docker, no WSL** at design time. User will install Docker (for Energym /
  optional local BOPTEST) and EnergyPlus (for Sinergym) as phases require them.
- BOPTEST Phase 1 uses the **public hosted service**, so it needs neither.

## 3. Architecture — Process-Isolated Adapters (Approach A)

The four simulators have genuinely conflicting dependencies (different
gym/gymnasium versions, EnergyPlus `PYTHONPATH`, Docker images) and Energym is not
even a gym. They cannot reliably co-exist in one Python environment. Therefore:

- **One conda env per simulator** (`rlhvac-citylearn`, `rlhvac-sinergym`,
  `rlhvac-boptest`, `rlhvac-energym`), plus a light UI env (`rlhvac-ui`, Py 3.11).
- The **UI process never imports a simulator.** It writes a job spec, spawns a
  runner subprocess in the correct env via `conda run -n <sim-env>`, and reads
  result files back.
- Communication is **file-based** through a per-run directory — no long-lived
  socket or RPC server.

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI  (env: rlhvac-ui, Python 3.11)             │
│  - pick simulator + scenario                             │
│  - dynamic config form (from adapter manifest)           │
│  - "Run baseline" / "Train (SB3)" buttons                │
│  - live charts (tails runs/<id>/metrics.jsonl)           │
└───────────────┬─────────────────────────────────────────┘
                │ conda run -n <sim-env> python -m rlhvac.runner --spec job.json
                ▼
┌─────────────────────────────────────────────────────────┐
│  Runner  (env: rlhvac-<sim>)                             │
│  - loads the adapter for the chosen sim                  │
│  - runs baseline episode OR SB3 training                 │
│  - writes status.json, metrics.jsonl, artifacts/         │
└───────────────┬─────────────────────────────────────────┘
                ▼
        ┌───────────────┐   ┌──────────────┐
        │ Adapter (gym) │──▶│ the simulator │
        └───────────────┘   └──────────────┘
```

## 4. Adapter Interface

Every simulator is wrapped behind one uniform interface. All four are normalized
to **Gymnasium** — Energym's custom API is shimmed to Gymnasium inside its
`make()` — so the runner is identical across simulators.

```python
class SimAdapter(Protocol):
    name: str  # "citylearn" | "sinergym" | "boptest" | "energym"

    @staticmethod
    def manifest() -> AdapterManifest:
        """Pure data, NO heavy imports. The UI env reads this even though it
           cannot import the sim. Contains:
           - scenarios/testcases list
           - config schema (typed fields → renders the form)
           - requirements (docker? energyplus? service URL?) + check() probe
           - native dashboard info (url/launch), if any"""

    def make(self, config: dict) -> gymnasium.Env:
        """Return a Gymnasium-compatible env (Energym shimmed here)."""

    def baseline_policy(self, env) -> Callable[[obs], action]:
        """Default controller — rule-based if the sim ships one, else random —
           so 'Run baseline' is meaningful."""

    def summarize(self, episode) -> dict:
        """Sim-specific KPIs: energy use, comfort violations, cost, etc."""
```

`AdapterManifest` is the contract that lets the UI render a per-simulator form and
a requirements/availability panel **without importing the simulator**. Adapter
modules that touch the simulator are imported only inside the runner subprocess.

## 5. Runner Protocol & Run Directory

A run is a directory both sides communicate through.

```
runs/<run_id>/
  job.json        # input: {sim, scenario, config, mode: baseline|train,
                  #         algo, timesteps, seed, visual: bool}
  status.json     # {state: queued|running|done|error, progress, pid, error?}
  metrics.jsonl   # one JSON line per step/episode (UI tails this live)
  artifacts/      # SB3 model.zip, plots, episode CSVs
  logs/runner.log # full stdout/stderr
```

- **Live visualization** = UI polls new lines in `metrics.jsonl` (~1 s) and
  redraws. The `visual` flag selects per-step (rich/heavy) vs per-episode (light)
  metric emission.
- **Robustness** = on crash, `status.json` → `error` with the traceback tail; the
  UI surfaces it instead of hanging.
- The runner is a single entry point: `python -m rlhvac.runner --spec <job.json>`,
  dispatching to `baseline` or `train` (SB3) mode.

## 6. UI Design (Streamlit)

Single-page app, localhost. Layout adapts per simulator from the manifest.

1. **Sidebar — Simulator picker.** Lists the four sims with an availability badge
   from each manifest's `check()` (green = ready, amber = needs Docker/EnergyPlus,
   with a one-line "how to enable" hint).
2. **Scenario + Config panel.** Dynamically rendered from the selected adapter's
   config schema (dropdowns for scenarios/testcases, number/bool inputs for params).
   This is how the UI "changes per simulator."
3. **Run controls.** `Run baseline` (Phase 1) and `Train (SB3)` (Phase 4) with
   algo/timesteps/seed and a `visual` toggle.
4. **Live panel.** Charts tailing `metrics.jsonl`: reward over time, key
   observations (indoor temp, energy, comfort), action trace; status + progress.
5. **Results panel.** KPIs from `summarize()`, links to artifacts, and — when a
   sim exposes one — a link/embed to its **native dashboard** (e.g. BOPTEST KPIs).

**Hosting:** a `host_ui.py` / `run.ps1` wrapper runs `streamlit run app.py` on
localhost so "host it" is one command.

## 7. Environment & Setup Strategy

- `envs/` holds one `environment-<sim>.yml` per simulator + `environment-ui.yml`.
- A `make setup` / `setup.ps1` creates the conda envs and runs each adapter's
  `check()` to report what's ready.
- Per-sim setup notes (EnergyPlus install + `PYTHONPATH`, Docker pulls for Energym,
  BOPTEST service URL config) live in `docs/setup/<sim>.md`.
- The UI degrades gracefully: unavailable sims are visible but disabled with the
  enablement hint, never a hard crash.

## 8. Repository Layout

```
RLHVAC/
  app.py                      # Streamlit entry
  host_ui.py / run.ps1        # launch on localhost
  rlhvac/
    runner.py                 # subprocess entry: baseline | train
    spec.py                   # job.json / status / metrics schemas
    run_store.py              # create/read run directories (UI side)
    launcher.py               # builds & spawns `conda run` commands
    adapters/
      base.py                 # SimAdapter Protocol + AdapterManifest
      citylearn.py
      boptest.py
      sinergym.py
      energym.py
    ui/                       # sidebar, config form, live panel, results
  envs/                       # environment-*.yml
  runs/                       # run directories (gitignored)
  docs/setup/                 # per-sim setup guides
```

## 9. Phased Build Plan

- **Phase 0 — Skeleton & contracts.** Repo layout, `SimAdapter`/`AdapterManifest`,
  run-directory schemas, `launcher` (conda-run spawn), a **mock adapter** + minimal
  Streamlit shell that runs a fake baseline end-to-end with live `metrics.jsonl`
  tailing. Proves the whole pipe with zero simulator deps.
- **Phase 1 — CityLearn + BOPTEST (baseline).** Real adapters for the two that run
  without Docker/EnergyPlus (BOPTEST via hosted service). Baseline episode + live
  charts + KPI summary in the UI.
- **Phase 2 — Sinergym (baseline).** EnergyPlus install + `PYTHONPATH`, adapter,
  setup guide.
- **Phase 3 — Energym (baseline).** Docker install, gym-shim adapter, setup guide.
- **Phase 4 — SB3 training mode.** `Train` path in the runner across all available
  adapters: live reward curves, checkpointing to `artifacts/`, model save/load.

Each phase ends with a working UI demoable to the mentor.

## 10. Testing Strategy

- **Contract tests** every adapter must pass: `manifest()` returns valid schema
  without heavy imports; `make()` yields a Gymnasium env; one `reset()`+`step()`
  succeeds; `summarize()` returns the declared KPI keys.
- **Mock adapter** drives runner + UI integration tests with no real simulator.
- **Runner protocol tests**: status transitions, crash → `error` + traceback,
  `metrics.jsonl` append/tail correctness.
- Real-simulator smoke tests are opt-in (require the env installed) and skipped
  when `check()` reports unavailable, so CI/dev without all deps still passes.

## 11. Open Questions / Future

- Embedding vs linking native dashboards (BOPTEST) — decide during Phase 1.
- Whether to graduate to Approach C (per-sim FastAPI microservices) if this becomes
  a shared multi-user lab tool.
- Experiment tracking (TensorBoard/W&B) — deferred past v1.
