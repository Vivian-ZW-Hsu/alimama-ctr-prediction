"""Guard against running a formal stage in the wrong Python environment.

The repository pins the environment it was tested in through
`requirements-tested.txt`, and `results/sequence_selection_v1/environment.json`
records the interpreter that produced the frozen artifacts. A second interpreter
with older packages exists on this machine, and running a formal stage there
silently produces artifacts that are not comparable with the frozen ones.

Every formal data or model entry point calls `require_tested_environment()`, so
that mistake fails immediately instead of being discovered afterwards.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "requirements-tested.txt"

# Distribution name -> import name.
MODULES = {
    "polars": "polars",
    "pyarrow": "pyarrow",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "torch": "torch",
    "lightgbm": "lightgbm",
}


def tested_pins() -> dict[str, str]:
    pins = {}
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def installed_versions() -> dict[str, str]:
    versions = {}
    for distribution, module_name in MODULES.items():
        try:
            versions[distribution] = importlib.import_module(module_name).__version__
        except Exception:
            versions[distribution] = "missing"
    return versions


def describe() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": installed_versions(),
        "tested_pins": tested_pins(),
    }


def compare() -> dict[str, tuple[str, str]]:
    """Return {distribution: (pinned, installed)} for every mismatch."""
    pins = tested_pins()
    installed = installed_versions()
    return {
        name: (pinned, installed.get(name, "missing"))
        for name, pinned in pins.items()
        if name in MODULES and installed.get(name) != pinned
    }


def require_tested_environment(stage: str) -> dict[str, object]:
    """Fail unless every pinned package matches `requirements-tested.txt`."""
    differences = compare()
    if differences:
        lines = [
            f"{stage} refuses to run: this interpreter is not the tested project "
            "environment.",
            f"  interpreter: {sys.executable}",
            "  package mismatches (pinned -> installed):",
        ]
        for name, (pinned, installed) in sorted(differences.items()):
            lines.append(f"    {name}: {pinned} -> {installed}")
        lines += [
            "",
            "Frozen artifacts were produced by the interpreter recorded in",
            "results/sequence_selection_v1/environment.json. Running a formal stage",
            "elsewhere yields artifacts that are not comparable with them.",
        ]
        raise RuntimeError("\n".join(lines))
    return describe()
