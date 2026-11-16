import math

import jax.numpy as jnp
import pytest

from flow_model_training import (
    distance_loss,
    ent_loss,
    entropy,
    flow_l2_loss,
    obs_loss,
)


def test_obs_loss_hand_calculated(tiny_densities):
    pred_densities, true_densities = tiny_densities
    # (0.5-0.4)^2 + (0.5-0.6)^2 + (0.6-0.5)^2 + (0.4-0.5)^2 = 0.04
    result = obs_loss(pred_densities, None, true_densities, None)
    assert float(result) == pytest.approx(0.04)


def test_distance_loss_hand_calculated(tiny_flow):
    flows, d_matrices = tiny_flow
    # 0.2*1 + 0.3*2 + 0.1*3 + 0.4*4 = 2.7
    result = distance_loss(None, flows, None, d_matrices)
    assert float(result) == pytest.approx(2.7)


def test_entropy_hand_calculated():
    p = jnp.array([0.5, 0.5])
    expected = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert float(entropy(p)) == pytest.approx(expected)


def test_entropy_degenerate_distribution_is_nan():
    p = jnp.array([1.0, 0.0])
    # 0*log(0) is treated as 0 in the limit; jnp.log(0) is -inf but the
    # mass there is 0, giving 0*-inf = nan unless this is handled upstream.
    # This test documents current behavior rather than asserting a "fixed"
    # value, since fixing it is out of scope for this refactor.
    result = float(entropy(p))
    assert math.isnan(result)


def test_ent_loss_matches_manual_combination(tiny_densities, tiny_flow):
    pred_densities, _ = tiny_densities
    flows, _ = tiny_flow

    result = ent_loss(pred_densities, flows, None, None)

    expected = sum(float(entropy(f)) for f in flows) - sum(
        float(entropy(p)) for p in pred_densities
    )
    assert float(result) == pytest.approx(expected)


def test_flow_l2_loss_hand_calculated(tiny_flow):
    flows, _ = tiny_flow
    # 0.2^2 + 0.3^2 + 0.1^2 + 0.4^2 = 0.04 + 0.09 + 0.01 + 0.16 = 0.3
    result = flow_l2_loss(None, flows, None, None)
    assert float(result) == pytest.approx(0.3)
