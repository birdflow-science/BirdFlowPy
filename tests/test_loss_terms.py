import math

import jax.numpy as jnp
import pytest

from flow_model_training import (
    distance_loss,
    ent_loss,
    entropy,
    flow_l2_loss,
    obs_loss,
    smooth_loss,
)


def test_obs_loss_hand_calculated(tiny_densities):
    pred_densities, true_densities = tiny_densities
    # (0.5-0.4)^2 + (0.5-0.6)^2 + (0.6-0.5)^2 + (0.4-0.5)^2 = 0.04
    result = obs_loss(pred_densities, None, true_densities, None, None)
    assert float(result) == pytest.approx(0.04)


def test_distance_loss_hand_calculated(tiny_flow):
    flows, d_matrices = tiny_flow
    # 0.2*1 + 0.3*2 + 0.1*3 + 0.4*4 = 2.7
    result = distance_loss(None, flows, None, d_matrices, None)
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

    result = ent_loss(pred_densities, flows, None, None, None)

    expected = sum(float(entropy(f)) for f in flows) - sum(
        float(entropy(p)) for p in pred_densities
    )
    assert float(result) == pytest.approx(expected)


def test_flow_l2_loss_hand_calculated(tiny_flow):
    flows, _ = tiny_flow
    # 0.2^2 + 0.3^2 + 0.1^2 + 0.4^2 = 0.04 + 0.09 + 0.01 + 0.16 = 0.3
    result = flow_l2_loss(None, flows, None, None, None)
    assert float(result) == pytest.approx(0.3)


def test_smooth_loss_hand_calculated_no_flicker(tiny_smooth_no_flicker):
    """3 weeks, 2 cells, no mask flicker; isolates open-boundary behavior
    (only the single interior week, t=1, contributes a curvature term --
    weeks 0 and 2 are boundary weeks and are excluded from the sum, though
    their values are still used as t=1's neighbors).

    densities: week0=[0.5,0.5], week1=[0.6,0.4], week2=[0.4,0.6]
    curvature[1] = d[0] - 2*d[1] + d[2] = [0.5,0.5] - [1.2,0.8] + [0.4,0.6]
                 = [-0.3, 0.3]
    sum of squares = 0.09 + 0.09 = 0.18
    """
    pred_densities, masks = tiny_smooth_no_flicker
    result = smooth_loss(pred_densities, None, None, None, masks)
    assert float(result) == pytest.approx(0.18, abs=1e-6)


def test_smooth_loss_hand_calculated_with_flicker(tiny_smooth_with_flicker):
    """3 cells, 3 weeks; cell 1 drops out of week 1's dynamic mask, so its
    week-1 density must be embedded as true zero (not skipped) -- checks
    zero-embedding. Only the interior week (t=1) contributes, per the
    open-boundary behavior in test_smooth_loss_hand_calculated_no_flicker.

    Embedded (canonical 3-cell space) densities:
      week0 = [0.5, 0.2, 0.3]
      week1 = [0.4, 0.0, 0.6]   (cell 1 inactive -> true zero)
      week2 = [0.1, 0.5, 0.4]
    curvature[1] = d[0] - 2*d[1] + d[2] = [0.5,0.2,0.3] - [0.8,0.0,1.2] + [0.1,0.5,0.4]
                 = [-0.2, 0.7, -0.5]
    sum of squares = 0.04 + 0.49 + 0.25 = 0.78
    """
    pred_densities, masks = tiny_smooth_with_flicker
    result = smooth_loss(pred_densities, None, None, None, masks)
    assert float(result) == pytest.approx(0.78, abs=1e-6)


def test_smooth_loss_two_weeks_has_no_interior_week():
    """With only 2 weeks, there is no interior week at all (both are
    boundary weeks), so smooth_loss must return exactly 0 regardless of
    how different the two weeks' densities are -- a direct check of the
    open-boundary exclusion, independent of any particular values."""
    pred_densities = [jnp.array([0.9, 0.1]), jnp.array([0.1, 0.9])]
    masks = jnp.ones((2, 2), dtype=bool)
    result = smooth_loss(pred_densities, None, None, None, masks)
    assert float(result) == pytest.approx(0.0, abs=1e-6)
