import copy

import pytest
import torch

from experiments import paper_nbody_precision_rank_dev as dev


def _source() -> dict:
    return {
        "learned_parameters": {
            "lengthscale": [0.75],
            "outputscale": 1.2,
            "value_noise_variance": 1e-3,
            "gradient_noise_variance": 2e-3,
            "kernel": "rbf",
            "gradient_noise_model": "iid",
        }
    }


def test_parameters_preserve_frozen_variance_semantics() -> None:
    parameters = dev._parameters_from_source(_source())
    assert parameters.lengthscale.dtype == torch.float64
    assert parameters.sigma_f == pytest.approx(1e-3)
    assert parameters.sigma_g == pytest.approx(2e-3)
    assert dev.SOURCE_RANK_EPSILON == torch.finfo(torch.float32).eps
    assert dev.MAXIMUM_DIRECTION_RANK == 16
    assert dev.TRUST_RADIUS_SIGMA == pytest.approx(0.025)


def test_parameters_reject_nonpositive_lengthscale() -> None:
    source = copy.deepcopy(_source())
    source["learned_parameters"]["lengthscale"] = [0.0]
    with pytest.raises(ValueError, match="lengthscale"):
        dev._parameters_from_source(source)
