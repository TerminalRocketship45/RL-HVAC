import sys
from rlhvac import launcher


def test_command_uses_current_interpreter_when_env_is_none():
    cmd = launcher.build_command("/runs/r1", runner_env=None)
    assert cmd[0] == sys.executable
    assert "rlhvac.runner" in cmd
    assert cmd[-1].endswith("job.json")


def test_command_uses_conda_run_for_named_env():
    cmd = launcher.build_command("/runs/r1", runner_env="rlhvac-sinergym")
    assert cmd[:4] == ["conda", "run", "-n", "rlhvac-sinergym"]
    assert "rlhvac.runner" in cmd
