"""Adapter registry with lazy module loading.

REGISTRY maps a simulator name to the dotted module path of its adapter class.
Real-simulator modules import their heavy deps only inside `make()`, so calling
`get_manifest()`/`get_adapter()` is safe in the rlhvac-ui env.
"""
from __future__ import annotations
import importlib
from rlhvac.spec import AdapterManifest

REGISTRY: dict[str, tuple[str, str]] = {
    # name: (module, class)
    "mock": ("rlhvac.adapters.mock", "MockAdapter"),
}


def get_adapter(name: str):
    module_path, class_name = REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def get_manifest(name: str) -> AdapterManifest:
    return get_adapter(name).manifest()


def available_sims() -> list[str]:
    return list(REGISTRY.keys())
