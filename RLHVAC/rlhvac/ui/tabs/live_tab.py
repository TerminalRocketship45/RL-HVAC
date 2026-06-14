from __future__ import annotations
from pathlib import Path
from rlhvac import run_store
from rlhvac.ui.scene_access import get_scene_schema
from rlhvac.ui import viz_plotly as viz


def list_run_dirs(base) -> list[Path]:
    base = Path(base)
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and (p / "job.json").exists()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def episode_frame_at(run_dir, episode: int, step_index: int) -> dict:
    frames = run_store.read_frames(run_store.episode_dir(run_dir, episode))
    if not frames:
        return {}
    step_index = max(0, min(step_index, len(frames) - 1))
    return frames[step_index]


def render(st, runs_dir="runs") -> None:
    st.subheader("Live simulation")
    runs = list_run_dirs(runs_dir)
    if not runs:
        st.info("No runs yet — launch one in the Run tab.")
        return
    default = st.session_state.get("active_run")
    names = [d.name for d in runs]
    idx = names.index(Path(default).name) if default and Path(default).name in names else 0
    run_name = st.selectbox("Run", names, index=idx)
    run_dir = next(d for d in runs if d.name == run_name)

    job = run_store.read_job(run_dir)
    status = run_store.read_status(run_dir)
    schema = get_scene_schema(job.sim)
    eps = run_store.list_episodes(run_dir)
    if not eps:
        st.info("Waiting for frames...")
        return
    ep = st.selectbox("Episode", eps, index=len(eps) - 1)
    frames = run_store.read_frames(run_store.episode_dir(run_dir, ep))
    st.caption(f"Status: {status.state} · episode {status.current_episode + 1}/{status.episodes_total}")
    if not frames:
        st.info("Waiting for frames...")
        return

    live = status.state == "running"
    step = len(frames) - 1 if live else st.slider("Step", 0, len(frames) - 1, len(frames) - 1)
    st.plotly_chart(viz.heatmap_figure(schema, frames[step]), use_container_width=True)

    summary = run_store.read_episode_summary(run_store.episode_dir(run_dir, ep))
    if summary:
        st.plotly_chart(viz.episode_bar_figure(summary), use_container_width=True)
    st.plotly_chart(viz.reward_timeseries(frames), use_container_width=True)
    temp_var = schema.color_by
    st.plotly_chart(viz.variable_timeseries(frames, schema, temp_var), use_container_width=True)
