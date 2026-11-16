from flow_model_training import extra_weight_filename_suffix

DEFAULTS = {'obs': 1.0, 'dist': 1e-2, 'ent': 1e-4, 'flow_l2': 0.0}


def test_all_weights_at_default_produces_no_suffix():
    """The historic case: obs/dist/ent at their defaults, flow_l2 (new,
    didn't exist pre-refactor) also at its default -- must reproduce
    exactly the empty suffix the old hardcoded filename implied."""
    weights = dict(DEFAULTS)
    assert extra_weight_filename_suffix(weights, DEFAULTS) == ''


def test_historic_weights_never_appear_in_suffix_even_when_nondefault():
    """obs/dist/ent are always embedded directly in update_hdf.py's
    f-string regardless of value (historic behavior); the suffix
    mechanism must never duplicate them, even if they're set away from
    their defaults."""
    weights = {'obs': 2.0, 'dist': 0.5, 'ent': 0.001, 'flow_l2': 0.0}
    assert extra_weight_filename_suffix(weights, DEFAULTS) == ''


def test_nonhistoric_weight_at_default_is_omitted():
    weights = dict(DEFAULTS, flow_l2=0.0)
    assert extra_weight_filename_suffix(weights, DEFAULTS) == ''


def test_nonhistoric_weight_away_from_default_is_included():
    weights = dict(DEFAULTS, flow_l2=2.5)
    assert extra_weight_filename_suffix(weights, DEFAULTS) == '_flow_l22.5'


def test_multiple_nonhistoric_weights_are_sorted_by_name():
    defaults = dict(DEFAULTS, zz_term=0.0)
    weights = dict(defaults, flow_l2=5.0, zz_term=3.0)
    assert extra_weight_filename_suffix(weights, defaults) == '_flow_l25.0_zz_term3.0'


def test_missing_default_treated_as_zero():
    """If a weight name has no entry in `defaults` at all, it's treated
    as defaulting to 0.0 (matching loss_fn's own weights.get(name, 0.0)
    convention), so a nonzero value for it is still included."""
    weights = dict(DEFAULTS, brand_new_term=1.0)
    assert extra_weight_filename_suffix(weights, DEFAULTS) == '_brand_new_term1.0'
