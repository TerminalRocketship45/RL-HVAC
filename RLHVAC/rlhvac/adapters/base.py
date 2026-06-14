from __future__ import annotations
from typing import Any, Callable, Protocol, runtime_checkable
import gymnasium as gym
from rlhvac.spec import AdapterManifest, CheckResult, SceneSchema, UnitSpec, VarSpec


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

    @staticmethod
    def scene_schema() -> "SceneSchema":
        """Static schema describing the per-simulator visual. No heavy import."""
        ...

    def read_scene(self, env) -> dict:
        """Per-step {unit_name: {var_name: value}} pulled from live env state."""
        ...


def default_scene_schema() -> SceneSchema:
    return SceneSchema(
        color_by="reward", color_range=(-10.0, 0.0), layout="grid",
        variables=[VarSpec(name="reward", label="Reward", kind="other")],
        units=[UnitSpec(name="system", label="System",
                        variables=[VarSpec(name="reward", label="Reward", kind="other")])],
    )


def default_read_scene(reward: float = 0.0) -> dict:
    return {"system": {"reward": float(reward)}}
