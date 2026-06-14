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


def test_scene_schema_dataclasses():
    from rlhvac.spec import VarSpec, UnitSpec, SceneSchema
    schema = SceneSchema(
        units=[UnitSpec(name="z0", label="Zone 0",
                        variables=[VarSpec(name="temp", label="Temp", unit="C", kind="temperature")])],
        color_by="temp", color_range=(10.0, 30.0), layout="grid",
    )
    assert schema.units[0].variables[0].kind == "temperature"
    assert schema.color_range == (10.0, 30.0)


def test_jobspec_has_episodes_default():
    from rlhvac.spec import JobSpec
    import json
    job = JobSpec(run_id="r", sim="mock", scenario="s", config={}, mode="baseline",
                  algo=None, timesteps=0, seed=0, visual=True)
    assert job.episodes == 1
    restored = JobSpec.from_json(json.loads(job.to_json()))
    assert restored.episodes == 1


def test_runstatus_has_episode_progress_fields():
    from rlhvac.spec import RunStatus
    s = RunStatus(state="running", current_episode=2, episodes_total=5)
    assert s.current_episode == 2 and s.episodes_total == 5
