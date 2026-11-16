"""End-to-end golden-value regression test for update_hdf.py.

update_hdf.py is a script (argparse + top-level execution), not an
importable module, so it's exercised here via subprocess, the same way
it's actually invoked from the command line / from R.

The expected values below were captured by running this exact fixture and
invocation against the pre-refactor code (commit dce4646, the last commit
before the Step 3 loss-term registry refactor -- see dev/workplan.md) and
comparing key-by-key against a run of the current code with equivalent
weights (default flow_l2_weight=0.0). That comparison found the two sets
of output identical except for the two new flow_l2-related keys the
registry always computes (expected, since flow_l2 is now part of
LOSS_TERMS; see test_flow_l2_weight_zero_is_noop in
test_golden_loss_fn.py for the same property at the loss_fn level).
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
            0.2997113764286041,
            0.29574722051620483,
            0.2914019823074341,
            0.2869254946708679,
            0.2825625538825989,
        ],
        "ent": [
            2.3626461029052734,
            2.4014925956726074,
            2.4148306846618652,
            2.409897804260254,
            2.3965017795562744,
        ],
        "flow_l2": [
            0.27607813477516174,
            0.24342674016952515,
            0.22658443450927734,
            0.2208499163389206,
            0.22131529450416565,
        ],
        "obs": [
            0.6907318830490112,
            0.5573172569274902,
            0.4501211643218994,
            0.3656962513923645,
            0.29921233654022217,
        ],
        "total": [
            0.6934927105903625,
            0.5600345730781555,
            0.452793687582016,
            0.36832451820373535,
            0.3017983138561249,
        ],
    }

    assert loss_values.keys() == expected_loss_values.keys()
    for key, expected in expected_loss_values.items():
        actual = loss_values[key]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == pytest.approx(e, rel=1e-4)

    assert hyperparameters["flow_l2_weight"] == pytest.approx(0.0)
    assert hyperparameters["dist_weight"] == pytest.approx(0.01)
    assert hyperparameters["ent_weight"] == pytest.approx(0.0001)
    assert hyperparameters["obs_weight"] == pytest.approx(1.0)

    assert week1_to_2.shape == (6, 6)
    assert week1_to_2[0, 0] == pytest.approx(0.0005380368093028665, rel=1e-4)


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
