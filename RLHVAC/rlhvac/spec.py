from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

FieldType = Literal["number", "int", "bool", "select", "text"]
VarKind = Literal["temperature", "power", "energy", "soc", "price", "carbon",
                  "comfort", "count", "setpoint", "other"]


@dataclass
class VarSpec:
    name: str
    label: str
    unit: str = ""
    kind: VarKind = "other"


@dataclass
class UnitSpec:
    name: str
    label: str
    variables: list[VarSpec] = field(default_factory=list)


@dataclass
class SceneSchema:
    color_by: str
    color_range: tuple[float, float] = (0.0, 1.0)
    layout: Literal["grid", "row", "diagram"] = "grid"
    variables: list[VarSpec] = field(default_factory=list)
    units: list[UnitSpec] = field(default_factory=list)

    def variable_meta(self, name: str) -> "VarSpec | None":
        return next((v for v in self.variables if v.name == name), None)


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
    episodes: int = 1

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
    current_episode: int = 0
    episodes_total: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: dict) -> "RunStatus":
        return cls(**data)
