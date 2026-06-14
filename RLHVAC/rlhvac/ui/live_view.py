from __future__ import annotations
from pathlib import Path
import pandas as pd
from rlhvac import run_store


def latest_episode_frames(run_dir) -> pd.DataFrame:
    eps = run_store.list_episodes(run_dir)
    if not eps:
        return pd.DataFrame()
    ep_dir = Path(run_dir) / "episodes" / f"{eps[-1]:03d}"
    return pd.DataFrame(run_store.read_frames(ep_dir))


def render_live(st, run_dir: Path) -> None:
    status = run_store.read_status(run_dir)
    st.write(f"**Status:** {status.state}  ·  episode {status.current_episode + 1}/{status.episodes_total}")
    if status.state == "error":
        st.error(status.error or "Run failed")
        return
    df = latest_episode_frames(run_dir)
    if df.empty:
        st.info("Waiting for frames...")
        return
    if "step" in df.columns and "reward" in df.columns:
        st.line_chart(df.set_index("step")["reward"])
    rollup = run_store.read_rollup(run_dir)
    if rollup:
        st.write("**Episode rewards**")
        st.bar_chart(pd.DataFrame(rollup).set_index("episode")["total_reward"])
