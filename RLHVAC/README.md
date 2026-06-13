# RLHVAC

Upper-level Streamlit UI over building-energy RL simulators
(CityLearn, BOPTEST, Sinergym, Energym). Process-isolated adapters:
the UI never imports a simulator -- it spawns a runner in that simulator's
own conda env and reads results from a per-run directory.

## Phase 0 (this milestone): skeleton + mock simulator

### Setup
```bash
conda env create -f envs/environment-ui.yml
```

### Run the UI
```bash
conda run -n rlhvac-ui python host_ui.py
# open http://localhost:8501 -> pick "mock" -> Run baseline
```

### Test
```bash
conda run -n rlhvac-ui pytest -v
```

## Phase 1: CityLearn

CityLearn (multi-building demand response) runs in the `rlhvac-citylearn` env.
See `docs/setup/citylearn.md`. The UI spawns its runner in that env via conda run.

## Architecture
See `docs/superpowers/specs/2026-06-13-rlhvac-simulator-ui-design.md`.
Real simulators arrive in later phases (each its own conda env + adapter).
