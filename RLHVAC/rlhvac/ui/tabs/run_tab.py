from __future__ import annotations
from pathlib import Path
from rlhvac import run_store, launcher
from rlhvac.spec import JobSpec
from rlhvac.adapters import available_sims, get_manifest
from rlhvac.ui.manifest_view import render_config_form

RUNS_DIR = Path("runs")


def render(st) -> None:
    st.subheader("Configure a run")
    sim = st.selectbox("Simulator", available_sims())
    manifest = get_manifest(sim)
    scenario = st.selectbox("Scenario", manifest.scenarios)
    st.caption(f"Runner env: `{manifest.runner_env}`")
    if manifest.dashboard:
        st.markdown(f"[Open native dashboard ↗]({manifest.dashboard})")
    config = render_config_form(st, manifest)
    episodes = st.number_input("Episodes", min_value=1, max_value=50, value=1, step=1)
    visual = st.checkbox("Stream per-step frames (visual)", value=True)
    if st.button("Run baseline", type="primary"):
        job = JobSpec(run_id=run_store.new_run_id(), sim=sim, scenario=scenario,
                      config=config, mode="baseline", algo=None, timesteps=0, seed=7,
                      visual=visual, episodes=int(episodes))
        run_dir = run_store.create_run(RUNS_DIR, job)
        runner_env = None if manifest.runner_env == launcher.UI_ENV else manifest.runner_env
        launcher.spawn(run_dir, runner_env)
        st.session_state["active_run"] = str(run_dir)
        st.success(f"Launched run {job.run_id} — see the Live Simulation tab.")
