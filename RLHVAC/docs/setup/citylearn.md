# CityLearn setup

CityLearn runs in its own conda env (dependency isolation from the UI env).

> **Important:** plain `conda env create -f envs/environment-citylearn.yml` is NOT
> sufficient — it installs CityLearn's dependencies only. CityLearn itself must be
> installed separately with `--no-deps` (see below).

## Canonical setup (PowerShell)

```powershell
pwsh envs/setup-citylearn.ps1
```

## Manual fallback (non-PowerShell / cross-platform)

```bash
conda env create -f envs/environment-citylearn.yml
conda run -n rlhvac-citylearn pip install "CityLearn==2.3.1" --no-deps
```

**Note:** CityLearn 2.3.1 hard-requires `openstudio<=3.3.0` which is unavailable on PyPI
(only 3.9.0+ exists). `openstudio` is only used in the `end_use_load_profiles` sub-module,
not in `CityLearnEnv` itself. Installing with `--no-deps` and providing the other runtime
deps via the yml is the workaround.

First baseline run downloads the selected dataset (cached afterwards).
In the UI, pick **citylearn** in the sidebar, choose a dataset scenario,
set the simulation-steps horizon (default 168 = one week), and Run baseline.
Mode is central single-agent (StableBaselines3Wrapper); baseline policy is
zero-action (no distributed-energy-resource control). KPIs come from
`CityLearnEnv.evaluate()` (district level).
