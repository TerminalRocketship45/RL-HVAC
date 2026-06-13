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
