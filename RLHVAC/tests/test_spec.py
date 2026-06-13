import json
from rlhvac.spec import JobSpec, RunStatus, ConfigField, AdapterManifest, CheckResult


def test_jobspec_roundtrip():
    job = JobSpec(
        run_id="r1", sim="mock", scenario="sine-day",
        config={"episode_length": 24}, mode="baseline",
        algo=None, timesteps=0, seed=7, visual=True,
    )
    restored = JobSpec.from_json(json.loads(job.to_json()))
    assert restored == job


def test_runstatus_defaults_and_roundtrip():
    s = RunStatus(state="running", progress=0.5, pid=123)
    restored = RunStatus.from_json(json.loads(s.to_json()))
    assert restored.state == "running"
    assert restored.error is None
    assert restored == s


def test_manifest_holds_config_schema():
    m = AdapterManifest(
        name="mock",
        scenarios=["sine-day"],
        config_schema=[ConfigField(name="episode_length", type="int", label="Episode length", default=24)],
        runner_env="rlhvac-ui",
        requirements=[],
        dashboard=None,
    )
    assert m.config_schema[0].type == "int"


def test_checkresult_shape():
    c = CheckResult(available=False, hint="Install EnergyPlus")
    assert c.available is False and "EnergyPlus" in c.hint
