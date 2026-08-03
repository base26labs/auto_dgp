from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from experiments.f02_internal_models import FrozenTERAParameters
from experiments.f02_same_m_diagnostic import _singular_spectra


def test_help_does_not_load_data_or_fit_a_model() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "experiments/f02_same_m_diagnostic.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "development-only" in result.stdout.lower()


def test_singular_spectrum_reports_current_numerical_rank() -> None:
    x_train = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    x_eval = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    parameters = FrozenTERAParameters(
        lengthscale=torch.ones(1),
        outputscale=1.0,
        sigma_f=0.1,
        sigma_g=0.1,
        kernel="rbf",
    )
    records = _singular_spectra(x_train, x_eval, parameters, m=3)
    assert len(records) == 1
    assert records[0]["algebraic_maximum_rank"] == 2
    assert records[0]["current_retained_rank"] == 1
