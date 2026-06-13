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
