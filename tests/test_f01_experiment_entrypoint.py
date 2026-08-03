"""Entrypoint regression for the vendored TERA simulation experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_f01_direct_entrypoint_resolves_vendored_tera_packages() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "experiments/f01_orbit_gp_sim.py", "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "F01 ORBIT GP-simulation experiment" in result.stdout
