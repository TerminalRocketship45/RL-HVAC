from __future__ import annotations
from pathlib import Path
import pandas as pd
from rlhvac import run_store


def metrics_dataframe(run_dir) -> pd.DataFrame:
    rows = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "step"]
    return pd.DataFrame(rows)


def render_live(st, run_dir: Path) -> None:
    status = run_store.read_status(run_dir)
    st.write(f"**Status:** {status.state}")
    if status.state == "error":
        st.error(status.error or "Run failed")
        return
    df = metrics_dataframe(run_dir)
    if df.empty:
        st.info("Waiting for metrics...")
        return
    if "reward" in df:
        st.line_chart(df.set_index("step")["reward"])
    if "temp" in df:
        st.line_chart(df.set_index("step")["temp"])
    summary = [m for m in run_store.read_metrics(run_dir) if m.get("kind") == "summary"]
    if summary:
        st.json(summary[-1])
