"""RLHVAC Streamlit UI — Run / Live / Metrics tabs."""
from __future__ import annotations
import time
import streamlit as st
from rlhvac import run_store
from rlhvac.ui.tabs import run_tab, live_tab, metrics_tab

st.set_page_config(page_title="RLHVAC", layout="wide")
st.title("RLHVAC — Simulator Control Panel")

run_t, live_t, metrics_t = st.tabs(["Run", "Live Simulation", "Metrics"])
with run_t:
    run_tab.render(st)
with live_t:
    live_tab.render(st)
with metrics_t:
    metrics_tab.render(st)

# auto-refresh while an active run is still in progress
active = st.session_state.get("active_run")
if active:
    status = run_store.read_status(active)
    if status.state in ("queued", "running"):
        time.sleep(1.0)
        st.rerun()
