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
    return run_dir


def write_status(run_dir: Path, status: RunStatus) -> None:
    (Path(run_dir) / "status.json").write_text(status.to_json())


def read_status(run_dir: Path) -> RunStatus:
    data = json.loads((Path(run_dir) / "status.json").read_text())
    return RunStatus.from_json(data)


def read_job(run_dir: Path) -> JobSpec:
    data = json.loads((Path(run_dir) / "job.json").read_text())
    return JobSpec.from_json(data)


def _ep_name(ep: int) -> str:
    return f"{ep:03d}"


def episode_dir(run_dir, ep: int) -> Path:
    return Path(run_dir) / "episodes" / _ep_name(ep)


def create_episode(run_dir, ep: int) -> Path:
    ep_dir = episode_dir(run_dir, ep)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "frames.jsonl").touch()
    return ep_dir


def list_episodes(run_dir) -> list[int]:
    base = Path(run_dir) / "episodes"
    if not base.exists():
        return []
    return sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit())


def append_frame(ep_dir, frame: dict) -> None:
    with open(Path(ep_dir) / "frames.jsonl", "a") as f:
        f.write(json.dumps(frame) + "\n")


def read_frames(ep_dir) -> list[dict]:
    path = Path(ep_dir) / "frames.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def write_episode_summary(ep_dir, summary: dict) -> None:
    (Path(ep_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


def read_episode_summary(ep_dir) -> dict:
    path = Path(ep_dir) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def append_rollup(run_dir, row: dict) -> None:
    with open(Path(run_dir) / "rollup.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def read_rollup(run_dir) -> list[dict]:
    path = Path(run_dir) / "rollup.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
