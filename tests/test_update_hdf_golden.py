"""End-to-end golden-value regression test for update_hdf.py.

update_hdf.py is a script (argparse + top-level execution), not an
importable module, so it's exercised here via subprocess, the same way
it's actually invoked from the command line / from R.

The expected values below were originally captured by running this exact
fixture and invocation against the pre-refactor code (commit dce4646, the
last commit before the Step 3 loss-term registry refactor -- see
dev/workplan.md) and comparing key-by-key against a run of the current
code with equivalent weights (default flow_l2_weight=0.0). That
comparison found the two sets of output identical except for the two new
flow_l2-related keys the registry always computes (expected, since
flow_l2 is now part of LOSS_TERMS; see test_flow_l2_weight_zero_is_noop
in test_golden_loss_fn.py for the same property at the loss_fn level).

They were regenerated 2026-11-17 against `.venv-unity` (jax==0.8.0,
dm-haiku==0.0.15, optax==0.2.6, numpy==2.3.4 -- the versions installed in
BirdFlowContainer on Unity as of that date, see requirements.txt and
dev/known-issues.md), after confirming (the same way as
test_golden_loss_fn.py) that the original values no longer reproduced
under the newer stack due to RNG/dependency drift rather than any code
regression.
"""

import glob
import os
import subprocess
import sys

import h5py
import pytest

TRAINING_STEPS = 5


def _run_update_hdf(root, species, resolution, ebirdst_year, extra_args=()):
    subprocess.run(
        [
            sys.executable,
            "update_hdf.py",
            root,
            species,
            str(resolution),
            "--training_steps",
            str(TRAINING_STEPS),
            "--ebirdst_year",
            str(ebirdst_year),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = glob.glob(f"{root}/{species}_{ebirdst_year}_{resolution}km_obs*.hdf5")
    # Pick the just-written file by mtime, in case the fixture directory
    # already holds an earlier run's differently-named output (e.g. the
    # collision-avoidance test runs update_hdf.py twice against one root).
    return max(candidates, key=os.path.getmtime)


def test_update_hdf_golden_output(synthetic_hdf5_fixture):
    root, species, resolution, ebirdst_year = synthetic_hdf5_fixture
    out_path = _run_update_hdf(root, species, resolution, ebirdst_year)

    assert os.path.basename(out_path) == (
        f"{species}_{ebirdst_year}_{resolution}km_obs1.0_ent0.0001_dist0.01_pow0.4.hdf5"
    )

    with h5py.File(out_path, "r") as f:
        loss_values = {k: f[f"metadata/loss_values/{k}"][()].tolist() for k in f["metadata/loss_values"]}
        hyperparameters = {k: f[f"metadata/hyperparameters/{k}"][()] for k in f["metadata/hyperparameters"]}
        week1_to_2 = f["marginals/Week1_to_2"][()]

    expected_loss_values = {
        "dist": [
            0.2947274446487427,
            0.29587891697883606,
            0.297288179397583,
            0.29854393005371094,
            0.2995595633983612,
        ],
        "ent": [
            2.6726341247558594,
            2.709975242614746,
            2.727104663848877,
            2.7285006046295166,
            2.7205650806427,
        ],
        "flow_l2": [
            0.1875382661819458,
            0.17206059396266937,
            0.16678723692893982,
            0.1688559502363205,
            0.17466774582862854,
        ],
        "obs": [
            0.5620908141136169,
            0.4655943214893341,
            0.3895336091518402,
            0.3302020728588104,
            0.2818446755409241,
        ],
        "smooth": [
            0.2979072332382202,
            0.2507886290550232,
            0.27094918489456177,
            0.34149613976478577,
            0.4368355870246887,
        ],
        "total": [
            0.5647708177566528,
            0.4682821035385132,
            0.3922337591648102,
            0.33291468024253845,
            0.2845682203769684,
        ],
    }

    assert loss_values.keys() == expected_loss_values.keys()
    for key, expected in expected_loss_values.items():
        actual = loss_values[key]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == pytest.approx(e, rel=1e-4)

    assert hyperparameters["flow_l2_weight"] == pytest.approx(0.0)
    assert hyperparameters["smooth_weight"] == pytest.approx(0.0)
    assert hyperparameters["dist_weight"] == pytest.approx(0.01)
    assert hyperparameters["ent_weight"] == pytest.approx(0.0001)
    assert hyperparameters["obs_weight"] == pytest.approx(1.0)

    assert week1_to_2.shape == (6, 6)
    assert week1_to_2[0, 0] == pytest.approx(0.008624911308288574, rel=1e-4)


def test_update_hdf_nonzero_flow_l2_weight_gets_distinct_filename(synthetic_hdf5_fixture):
    """A run with a nonzero flow_l2_weight must not collide with (or
    overwrite) the default run's output file -- this is the bug fixed
    after the original registry refactor: before the fix, both runs
    produced the identical obs/ent/dist/pow filename regardless of
    flow_l2_weight."""
    root, species, resolution, ebirdst_year = synthetic_hdf5_fixture

    default_path = _run_update_hdf(root, species, resolution, ebirdst_year)
    nonzero_path = _run_update_hdf(
        root, species, resolution, ebirdst_year, extra_args=["--flow_l2_weight", "2.5"]
    )

    assert default_path != nonzero_path
    assert os.path.exists(default_path)
    assert os.path.exists(nonzero_path)
    assert os.path.basename(nonzero_path) == (
        f"{species}_{ebirdst_year}_{resolution}km_obs1.0_ent0.0001_dist0.01_flow_l22.5_pow0.4.hdf5"
    )

    with h5py.File(nonzero_path, "r") as f:
        assert f["metadata/hyperparameters/flow_l2_weight"][()] == pytest.approx(2.5)


def test_update_hdf_nonzero_smooth_weight_gets_distinct_filename(synthetic_hdf5_fixture):
    """Same collision-avoidance property as flow_l2 above, for the new
    smooth_weight flag."""
    root, species, resolution, ebirdst_year = synthetic_hdf5_fixture

    default_path = _run_update_hdf(root, species, resolution, ebirdst_year)
    nonzero_path = _run_update_hdf(
        root, species, resolution, ebirdst_year, extra_args=["--smooth_weight", "3.0"]
    )

    assert default_path != nonzero_path
    assert os.path.exists(default_path)
    assert os.path.exists(nonzero_path)
    assert os.path.basename(nonzero_path) == (
        f"{species}_{ebirdst_year}_{resolution}km_obs1.0_ent0.0001_dist0.01_smooth3.0_pow0.4.hdf5"
    )

    with h5py.File(nonzero_path, "r") as f:
        assert f["metadata/hyperparameters/smooth_weight"][()] == pytest.approx(3.0)
