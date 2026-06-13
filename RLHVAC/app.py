"""RLHVAC Streamlit UI -- Phase 0 shell."""
from __future__ import annotations
import time
from pathlib import Path
import streamlit as st
from rlhvac import run_store, launcher
from rlhvac.spec import JobSpec
from rlhvac.adapters import available_sims, get_manifest
from rlhvac.ui.manifest_view import render_config_form
from rlhvac.ui.live_view import render_live

RUNS_DIR = Path("runs")

st.set_page_config(page_title="RLHVAC", layout="wide")
st.title("RLHVAC -- Simulator Control Panel")

with st.sidebar:
    st.header("Simulator")
    sim = st.selectbox("Choose a simulator", available_sims())
    manifest = get_manifest(sim)
    scenario = st.selectbox("Scenario", manifest.scenarios)
    st.caption(f"Runner env: `{manifest.runner_env}`")

st.subheader("Configuration")
config = render_config_form(st, manifest)
visual = st.checkbox("Stream per-step metrics (visual)", value=True)

if st.button("Run baseline", type="primary"):
    job = JobSpec(run_id=run_store.new_run_id(), sim=sim, scenario=scenario,
                  config=config, mode="baseline", algo=None, timesteps=0,
                  seed=7, visual=visual)
    run_dir = run_store.create_run(RUNS_DIR, job)
    runner_env = None if manifest.runner_env == launcher.UI_ENV else manifest.runner_env
    launcher.spawn(run_dir, runner_env)
    st.session_state["active_run"] = str(run_dir)

if "active_run" in st.session_state:
    st.subheader("Live")
    run_dir = Path(st.session_state["active_run"])
    placeholder = st.empty()
    for _ in range(600):  # ~10 min cap of 1s polls
        with placeholder.container():
            render_live(st, run_dir)
        if run_store.read_status(run_dir).state in ("done", "error"):
            break
        time.sleep(1)
