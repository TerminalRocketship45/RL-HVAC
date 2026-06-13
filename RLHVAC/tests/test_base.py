from rlhvac.adapters.base import SimAdapter


def test_protocol_is_runtime_checkable():
    class Incomplete:
        name = "x"
    assert not isinstance(Incomplete(), SimAdapter)
