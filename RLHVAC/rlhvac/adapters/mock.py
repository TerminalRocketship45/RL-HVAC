from __future__ import annotations
from typing import Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rlhvac.spec import AdapterManifest, ConfigField, CheckResult, SceneSchema, UnitSpec, VarSpec


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
        self._temp = 15.0  # fixed start -> deterministic
        return np.array([self._temp], dtype=np.float32), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).flat[0], -5.0, 5.0))
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

    @staticmethod
    def scene_schema() -> SceneSchema:
        vars_ = [
            VarSpec(name="temp", label="Temperature", unit="C", kind="temperature"),
            VarSpec(name="setpoint", label="Setpoint", unit="C", kind="setpoint"),
        ]
        return SceneSchema(color_by="temp", color_range=(10.0, 30.0), layout="grid",
                           variables=vars_,
                           units=[UnitSpec(name="zone", label="Zone", variables=vars_)])

    def read_scene(self, env) -> dict:
        u = env.unwrapped
        return {"zone": {"temp": float(u._temp), "setpoint": float(u.setpoint)}}
