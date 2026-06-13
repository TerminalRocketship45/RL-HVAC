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
