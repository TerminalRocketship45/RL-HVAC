from __future__ import annotations
from pathlib import Path
from rlhvac import run_store
from rlhvac.ui.tabs.live_tab import list_run_dirs
from rlhvac.ui import viz_plotly as viz


def rollup_metric_names(run_dir) -> list[str]:
    rollup = run_store.read_rollup(run_dir)
    keys: list[str] = []
    for row in rollup:
        for k, v in row.items():
            if k != "episode" and isinstance(v, (int, float)) and k not in keys:
                keys.append(k)
    return keys


def render(st, runs_dir="runs") -> None:
    st.subheader("Metrics across episodes")
    runs = list_run_dirs(runs_dir)
    if not runs:
        st.info("No runs yet.")
        return
    default = st.session_state.get("active_run")
    names = [d.name for d in runs]
    idx = names.index(Path(default).name) if default and Path(default).name in names else 0
    run_name = st.selectbox("Run", names, index=idx, key="metrics_run")
    run_dir = next(d for d in runs if d.name == run_name)

    rollup = run_store.read_rollup(run_dir)
    if not rollup:
        st.info("No completed episodes yet.")
        return
    metrics = rollup_metric_names(run_dir)
    st.caption("Cross-episode trends (training curves slot in here once RL training is added).")
    chosen = st.multiselect("Metrics", metrics, default=[m for m in ["total_reward"] if m in metrics] or metrics[:1])
    for m in chosen:
        st.plotly_chart(viz.rollup_curve_figure(rollup, m), use_container_width=True)
