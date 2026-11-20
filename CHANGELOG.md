# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Pluggable loss-term registry (`LOSS_TERMS` in `flow_model_training.py`):
  loss components are no longer hardcoded into `loss_fn`'s signature and
  return value, so new additive terms or CLI-selectable variants can be
  added without editing `loss_fn`, `train_model`, and `update_hdf.py`'s
  HDF5-writing code separately.
- `flow_l2` loss term (L2 penalty on the flow matrices) and a
  `--flow_l2_weight` CLI flag in `update_hdf.py`, default `0.0`, as a
  worked example of the registry mechanism.
- `tests/` (pytest) covering `flow_model_training.py` and
  `update_hdf.py`, including golden-value regression tests and an
  end-to-end test of `update_hdf.py` against a synthetic HDF5 fixture.
  See `requirements.txt` for the pinned dependency versions (matching
  the production container on Unity) to run them against.
- `dev/architecture.md`, `dev/known-issues.md`, and the `dev/`
  workplan/worklog convention.

### Fixed

- `ent_loss`'s parameters were named backwards from what they actually
  received (`ent_loss(probs, flows)`, called as `ent_loss(flows,
  pred_densities)`); renamed to match actual usage. No change in
  computed values.

## [1.1.0] - 2023-12-12

- Switched to per-week dynamic cell masks (`geom/dynamic_mask` in the
  HDF5 file) in place of a single static NaN mask, matching
  BirdFlowR's per-week active-cell concept.
- `update_hdf.py` introduced as the HDF5-based interface to the R
  package: reads `distr`/`distances`/`geom/dynamic_mask` from an
  existing HDF5 file and writes marginals, hyperparameters, and loss
  history back into it, in place of the original raster/`.npy`-based
  workflow.
- Model hyperparameters (`obs_weight`, `dist_weight`, `ent_weight`,
  `dist_pow`, etc.) exposed as `update_hdf.py` CLI arguments instead of
  being hardcoded.
- Output HDF5 filename now encodes the hyperparameters used; output
  structure updated with additional metadata fields.
- Distance-power normalization option added (`--dont_normalize`).
- `--save_pkl` flag added (parameter pickling now opt-in, previously
  always on).
- Removed now-deprecated `jax.ops.index`/`jax.ops.index_update` API
  usage in `flow_model.py`.
- Added `--ebirdst_year` argument for constructing the input/output
  filenames against a specific eBird Status & Trends version year.

## [1.0.0] - 2022-09-08

Initial public release, tied to the Zenodo archive and the published
paper (Fuentes et al., [besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14052](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14052)):
model definition (`flow_model.py`), training loop
(`flow_model_training.py`), and the original raster/`.npy`-based
training entry point (`run_model.py`).

[Unreleased]: https://github.com/birdflow-science/BirdFlowPy/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/birdflow-science/BirdFlowPy/compare/zenodo_release...v1.1.0
[1.0.0]: https://github.com/birdflow-science/BirdFlowPy/releases/tag/zenodo_release
