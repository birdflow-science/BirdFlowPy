from flow_model_training import extra_weight_filename_suffix

ZERO_WEIGHTS = {'obs': 1.0, 'dist': 1e-2, 'ent': 1e-4, 'flow_l2': 0.0}


def test_nonhistoric_weight_at_zero_produces_no_suffix():
    """The historic case: flow_l2 (new, didn't exist pre-refactor) at
    zero -- must reproduce exactly the empty suffix the old hardcoded
    filename implied, regardless of what obs/dist/ent are set to."""
    assert extra_weight_filename_suffix(ZERO_WEIGHTS) == ''


def test_historic_weights_never_appear_in_suffix_even_when_nonzero():
    """obs/dist/ent are always embedded directly in update_hdf.py's
    f-string regardless of value (historic behavior); the suffix
    mechanism must never duplicate them."""
    weights = {'obs': 2.0, 'dist': 0.5, 'ent': 0.001, 'flow_l2': 0.0}
    assert extra_weight_filename_suffix(weights) == ''


def test_nonhistoric_weight_nonzero_is_included():
    weights = dict(ZERO_WEIGHTS, flow_l2=2.5)
    assert extra_weight_filename_suffix(weights) == '_flow_l22.5'


def test_multiple_nonhistoric_weights_are_sorted_by_name():
    weights = dict(ZERO_WEIGHTS, flow_l2=5.0, zz_term=3.0)
    assert extra_weight_filename_suffix(weights) == '_flow_l25.0_zz_term3.0'


def test_negative_weight_is_still_nonzero():
    weights = dict(ZERO_WEIGHTS, flow_l2=-1.0)
    assert extra_weight_filename_suffix(weights) == '_flow_l2-1.0'
