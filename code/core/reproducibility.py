"""Portable paths and strict JSON helpers for reproducible experiments."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def _resolve_path(value: str, base_dir: Path, environ: Mapping[str, str]) -> str:
    expanded = os.path.expanduser(value)
    for name, replacement in environ.items():
        expanded = expanded.replace("$" + "{" + name + "}", replacement)
    if "$" + "{" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def resolve_runtime_paths(
    config: dict,
    config_path: Path | None,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Return a runtime copy with path fields resolved without mutating the manifest."""
    resolved = copy.deepcopy(config)
    base_dir = config_path.parent if config_path else Path.cwd()
    env = dict(os.environ if environ is None else environ)
    for section, key in (
        ("data", "xml_dir"),
        ("data", "feature_config"),
        ("output", "base_dir"),
    ):
        value = resolved.get(section, {}).get(key)
        if isinstance(value, str):
            resolved[section][key] = _resolve_path(value, base_dir, env)
    return resolved


def json_safe(value: Any) -> Any:
    """Convert non-finite numeric values to JSON null recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def strict_json_dump(value: Any, handle, *, indent: int = 2) -> None:
    """Write standards-compliant JSON; NaN and infinities become null."""
    json.dump(json_safe(value), handle, indent=indent, allow_nan=False, default=str)
    handle.write("\n")
