"""CityLearn adapter. Heavy `citylearn` imports happen ONLY inside make()/check()
so this module is importable in the rlhvac-ui env where citylearn is absent."""
from __future__ import annotations
import math
from typing import Any, Callable
from rlhvac.spec import AdapterManifest, ConfigField, CheckResult, SceneSchema, VarSpec

# Curated built-in datasets (static — must not import citylearn to list them).
# Thermal-capable datasets (have indoor_dry_bulb_temperature) come first.
_DATASETS = [
    "baeda_3dem",
    "citylearn_challenge_2023_phase_1",
    "citylearn_challenge_2022_phase_1",
    "citylearn_challenge_2022_phase_2",
]

# Superset of per-building observation keys across all supported datasets.
# read_scene() includes only keys actually present + non-None per dataset.
_BUILDING_VARS = [
    VarSpec("indoor_dry_bulb_temperature", "Indoor temp", "C", "temperature"),
    VarSpec("indoor_dry_bulb_temperature_cooling_set_point", "Cooling setpoint", "C", "setpoint"),
    VarSpec("indoor_dry_bulb_temperature_cooling_delta", "Comfort delta", "C", "comfort"),
    VarSpec("cooling_storage_soc", "Cooling storage SoC", "", "soc"),
    VarSpec("electrical_storage_soc", "Battery SoC", "", "soc"),
    VarSpec("non_shiftable_load", "Load", "kWh", "power"),
    VarSpec("solar_generation", "Solar", "kWh", "power"),
    VarSpec("net_electricity_consumption", "Net electricity", "kWh", "energy"),
    VarSpec("occupant_count", "Occupants", "", "count"),
    VarSpec("carbon_intensity", "Carbon intensity", "kgCO2/kWh", "carbon"),
    VarSpec("electricity_pricing", "Price", "$/kWh", "price"),
    VarSpec("outdoor_dry_bulb_temperature", "Outdoor temp", "C", "temperature"),
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
            requirements=["pip install CityLearn==2.3.1 --no-deps (in rlhvac-citylearn env)"],
            dashboard=None,
        )

    @staticmethod
    def check() -> CheckResult:
        try:
            import citylearn  # noqa: F401
            return CheckResult(available=True, hint="")
        except Exception:
            return CheckResult(available=False,
                               hint="Create the env: conda env create -f envs/environment-citylearn.yml"
                                    " then: conda run -n rlhvac-citylearn pip install 'CityLearn==2.3.1' --no-deps")

    def make(self, config: dict):
        from citylearn.citylearn import CityLearnEnv
        from citylearn.wrappers import StableBaselines3Wrapper

        scenario = config.get("scenario") or _DATASETS[0]
        steps = int(config.get("simulation_steps", 168))
        seed = int(config.get("seed", 0))
        base = CityLearnEnv(
            schema=scenario,
            central_agent=True,
            simulation_start_time_step=0,
            simulation_end_time_step=max(1, steps),
            random_seed=seed,
        )
        self._citylearn_env = base  # stashed for summarize()
        return StableBaselines3Wrapper(base)

    @staticmethod
    def scene_schema() -> SceneSchema:
        return SceneSchema(
            color_by="indoor_dry_bulb_temperature",
            color_range=(18.0, 32.0), layout="grid",
            variables=list(_BUILDING_VARS),
        )

    def read_scene(self, env) -> dict:
        base = getattr(self, "_citylearn_env", None) or env.unwrapped
        wanted = {v.name for v in _BUILDING_VARS}
        scene: dict = {}
        for b in base.buildings:
            try:
                obs = b.observations()
            except TypeError:
                obs = b.observations(normalize=False, periodic_normalization=False)
            scene[b.name] = {k: float(obs[k]) for k in wanted if k in obs and obs[k] is not None}
        return scene

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
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    out[str(row["cost_function"])] = float(val)
        except Exception as exc:  # evaluate() schema can vary by version
            out = {"evaluate_error": str(exc)[:200]}
        return out
