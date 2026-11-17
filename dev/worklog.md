# Worklog

Running log of changes made in this repo: code edits, doc edits,
config tweaks, exploratory findings, memory/settings updates, build
fiddling. More detailed than any user-facing changelog — include
enough context that a reader can reconstruct what happened without
diffing.

Newest entries on top. Group by `## YYYY-MM-DD — branch <name>`
headers, then `### <short scope>` sub-headers under them. If today's
date and current branch already have a heading, append a new
sub-heading rather than starting a new date block.

Older entries get moved to `worklog-archive.md` once this file gets
long (~500 lines) or the oldest entry is more than ~30 days old. Move
whole date blocks, never split one. Read this file first when
recalling history; consult the archive only if the answer isn't here.

---

## 2026-11-17 — branch pluggable-loss-terms

### Matched local dev environment to the Unity container; regenerated golden tests; dropped the old environment

Installed Python 3.13.18 (Homebrew) and created `.venv-unity`, pinned
to `requirements.txt` (the versions captured from the Unity container
earlier this session). Every package resolved to the exact pinned
version with no conflicts. Ran `tests/` under `.venv-unity`: no
import/API errors, but the three golden-value/snapshot tests failed
with large numeric divergences (including a sign flip in a fitted
parameter). Cross-checked via a `git worktree` at commit `dce4646`
(pre-dating this branch's registry refactor): identical failure
pattern under `.venv-unity`, confirming this is jax 0.4.18→0.8.0
RNG/dependency drift, not a regression introduced by this branch.
Regenerated the frozen expected values in `tests/test_golden_loss_fn.py`
and `tests/test_update_hdf_golden.py` against `.venv-unity`; full
17-test suite now passes under it.

Per the project owner's decision, dropped the old local dev
environment entirely (it no longer has a purpose now that
`.venv-unity` matches production): deleted `requirements-dev.txt` and
`.venv/`, updated `dev/architecture.md`/`CHANGELOG.md` to point at
`.venv-unity`/`requirements.txt` as the one documented dev/test
environment going forward. Full details in
`dev/known-issues.md`'s issue #5 entry ("Update 2026-11-17").

### Clarified BirdFlowPipeline test suite's coverage and merge bar

Project owner confirmed `BirdFlowPipeline`'s suite doesn't exercise
this branch's new features (registry/`flow_l2`/new HDF5 fields), does
run full fits so would likely catch a hard dependency break, but isn't
clearly rigorous about snapshotting fitted output for regressions.
Also clarified merge priority: don't break the existing workflow while
opening room for new loss terms — the owner is currently the sole
user of the new features and is fine finding bugs in those via normal
development, so `BirdFlowPipeline` coverage isn't a blocker for this
branch. Details and reasoning in `dev/known-issues.md`'s issue #5
entry.

### Discussed dependency-pinning strategy; added requirements.txt

Discussed with the project owner how BirdFlowPy should manage
dependency versions long-term, given it lacks CI and differs from
BirdFlowR's continuously-validated approach. Full discussion and
proposed direction captured in `dev/known-issues.md`'s "No
pinned/tracked dependency versions for the production environment
(issue #5)" entry. Pragmatically, added `requirements.txt` recording
the versions currently installed in the Unity container (not yet
wired into `BirdFlowContainer.def`; no container rebuild performed).

### Compared local vs. Unity container dependency versions (issue #5)

Investigated [issue #5](https://github.com/birdflow-science/BirdFlowPy/issues/5),
a user-reported `module 'jax.extend.core' has no attribute 'JaxprEqn'`
error asking for the specific `jax`/`haiku`/`optax`/`scipy` versions
this project needs. Found that `BirdFlowContainer.def` installs these
via plain `pip install --upgrade` with no version pins, so there was
no tracked answer anywhere.

Wrote `dev/print_container_versions.sh`, a standalone script (prints
Python + `jax`/`jaxlib`/`haiku`/`optax`/`scipy`/`h5py`/`numpy`
versions from whatever environment it's run in) intended to be copied
to Unity and run inside the container via `apptainer exec
BirdFlowContainer.sif bash dev/print_container_versions.sh` or
`reticulate::repl_python()` in an RStudio session. Verified it against
the local `.venv` first (matched `requirements-dev.txt`'s pins
exactly). User ran it on Unity inside the current container; results
diverged sharply from local: Python 3.13.9 (vs. 3.9 locally), jax/jaxlib
0.8.0 (vs. 0.4.18), haiku 0.0.15 (vs. 0.0.10), optax 0.2.6 (vs. 0.1.7),
scipy 1.16.2 (vs. 1.10.1), h5py 3.15.1 (vs. 3.14.0), and — most
notably — numpy 2.3.4 (vs. 1.26.4; `requirements-dev.txt` deliberately
pins `numpy<2`).

Logged this as a known issue (`dev/known-issues.md`, "No
pinned/tracked dependency versions for the production environment
(issue #5)") rather than fixing anything, since the fix needs a
decision: where to pin (most likely `BirdFlowContainer.def`, since it
actually controls what ships) and how/whether to reconcile
`requirements-dev.txt` with those pins. Also flagged two downstream
consequences worth deciding alongside it: (1) this branch's golden-value
regression tests have only run against the old numpy-1.x/jax-0.4/
Python-3.9 stack locally, never the numpy-2.x/jax-0.8/Python-3.13 stack
production actually runs, so they may need regenerating; (2) issue #5's
specific error looks like a plausible haiku/jax internal-API
incompatibility at the container's *current* versions rather than a
user mistake, meaning the container as currently built may already be
broken for model fitting independent of anything in this branch.

Also added `run_model.py`/`birdflow_demo.ipynb` as a known issue
(`dev/known-issues.md`) at the user's request: both are broken against
current code (`Datatuple` arity mismatch, removed `gen_d_matrix`,
deprecated JAX indexing API in the notebook — all pre-existing,
already described in `dev/architecture.md`'s "Stale vs. current
paths"), framed as a fix-both-or-delete-both decision rather than
picked for them.

## 2026-11-16 — branch pluggable-loss-terms

### Documented testing an unmerged branch on Unity

Prompted by planning how to test this branch (or issue #7's eventual
temporal-smoothing work) on Unity without affecting other users or
rebuilding `BirdFlowContainer`. Established that the container
(`../BirdFlowContainer/BirdFlowContainer.def`, a separate repo) only
supplies the shared environment (conda/Python, R, pinned
`jax`/`dm-haiku`/`optax`/`h5py`, etc.) — it never bakes in a copy of
this repo's code. Per `BirdFlowContainer`'s README, a user launches
the shared `.sif` via Unity OnDemand's RStudio IDE and then opens
whichever local BirdFlowPy clone they want as the RStudio project, so
code and environment are already decoupled per user.

Added a new "Running on Unity, and testing an unmerged branch there"
section to `dev/architecture.md` (between "Stale vs. current paths"
and "Tests") documenting the consequence: testing a feature branch is
just `git worktree add ../BirdFlowPy-<branch-name> <branch-name>` plus
pointing an RStudio session at that directory — a worktree rather than
a second clone, since it shares the same repo/object store with no
separate remote to sync. Flagged the one case where this doesn't
suffice: a branch that needs a dependency the shared image doesn't
already have would require an actual container rebuild.

Also noted that when `update_hdf.py` is instead invoked via
`BirdFlowPipeline` (the R package driving batch
preprocessing/fits/downstream processing on the cluster) rather than
run from an open RStudio project directly, the directory it uses is
controlled by `BirdFlowPipeline::the$python_repo_path`, not by
whatever RStudio project is open — so pointing a pipeline-driven run
at a branch worktree means updating that setting instead.

No code changes; `dev/architecture.md` only.

### Added version tracking and a changelog

No version number or changelog existed anywhere in the repo (no
`setup.py`/`pyproject.toml` either). Added `_version.py` — a
standalone `__version__` string (PEP 440/semver convention, the
standard approach when there's no package `__init__.py` to attach a
version attribute to) — and `CHANGELOG.md` in [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/) format.

Backfilled history using the existing `zenodo_release` git tag
(2022-09-08, tied to the published paper) as `v1.0.0`, and main's
current state (2023-12-12, i.e. everything accumulated since the tag:
dynamic masks, `update_hdf.py`'s introduction, hyperparameter CLI
args, `--ebirdst_year`, etc.) as `v1.1.0`. This branch's own changes
(loss-term registry, `flow_l2`, the filename fix) are the
`[Unreleased]` section.

`_version.py` is kept at `1.1.0` (the last *released* version) rather
than bumped to `1.2.0` — per instruction, it should only move to
`1.2.0` when this branch's changes actually ship, at the same time the
changelog's `[Unreleased]` section gets a version number and date.

Also logged a known issue at the user's request (not a review
finding): `update_hdf.py`'s output HDF5 doesn't record which codebase
version produced it, which would help trace an old model run back to
the code that generated it once loss terms/CLI flags start changing
between releases. See `dev/known-issues.md`'s "Record the package
version in HDF5 output metadata" entry for the concrete fix sketch
(write `_version.py`'s `__version__` to a new `metadata/birdflow_py_version`
dataset, alongside the existing `birdflow_model_date` write).

Pointed `README.md` and `dev/architecture.md` at the new files; both
updated, no test changes needed (versioning/changelog additions don't
touch runtime code). Suite still at 17 passing.

### Critical review of the branch + output-filename collision fix

Ran a critical review of the full `pluggable-loss-terms` diff
(`main..pluggable-loss-terms`) looking for problems introduced by the
refactor, re-running scenarios against the actual code rather than
just reading the diff. Found three issues; fixed the most significant
one, logged the other two for later triage (see
`dev/known-issues.md`).

Fixed: `update_hdf.py`'s output filename (and `--save_pkl` filename)
only ever encoded `obs`/`ent`/`dist`/`dist_pow`, so once `flow_l2_weight`
existed, two runs differing only in that weight produced the identical
destination filename and silently overwrote each other via
`shutil.copyfile` — no error, no warning. Added
`flow_model_training.extra_weight_filename_suffix(weights)`: any
registered loss weight other than the historic `obs`/`dist`/`ent` trio
gets appended to the filename as `_<name><value>` whenever it's
nonzero (matching `loss_fn`'s own `weights.get(name, 0.0)` convention
— a zero weight has no effect on the loss, so it doesn't need to be in
the filename). The historic three stay unconditionally embedded
regardless of value, exactly as before, so default/historic runs
reproduce byte-identical filenames to before this branch.

Went through two iterations on the condition: first "differs from its
CLI default" (required threading a `defaults` dict through
`update_hdf.py`), then simplified to "nonzero" per feedback, since
that's the more fundamental rule (matches `loss_fn`'s own treatment of
weights) and happens to coincide with "differs from default" for every
term today, since every non-historic term's CLI default is `0.0`.

Added `tests/test_filename_suffix.py` (unit tests on the suffix logic:
historic terms always excluded even when nonzero, non-historic terms
included only when nonzero, multiple extras sorted by name, negative
weights still count as nonzero) and two tests in
`tests/test_update_hdf_golden.py`: one confirming the default run's
filename matches the pre-existing historic pattern exactly, one
confirming a `--flow_l2_weight 2.5` run produces a *different*
filename from the default run rather than colliding with it (directly
reproducing, then disproving, the original bug). Final suite: 17
tests passing.

### Pluggable loss-term registry + flow_l2 example term

Goal: make it possible to add/swap loss-function components for
`update_hdf.py` without changing default behavior, backed by
regression tests proving the refactor alone doesn't change numeric
output. Full plan was tracked in `dev/workplan.md` (now reset to
placeholder) across six steps; details below.

- **Environment**: no local Python/jax/haiku/optax existed on this
  machine and the repo had no requirements file (production fitting
  happens in `BirdFlowContainer` on Unity). Created `.venv/`
  (gitignored) pinned via `requirements-dev.txt`:
  `numpy<2`, `jax==0.4.18`, `jaxlib==0.4.18`, `dm-haiku==0.0.10`,
  `optax==0.1.7`, `scipy==1.10.1`, `h5py`, `pytest` — the newest
  mutually-compatible set for Python 3.9 (this Mac's system Python;
  newer haiku/jax/optax require 3.10+). This is local-dev-only and
  has **not** been verified against the Unity/`BirdFlowContainer`
  production environment — that verification is planned separately,
  after this branch, before merging.
- **Step 1 (regression tests against pre-refactor code)**: added
  `tests/` (pytest): `tests/conftest.py` fixtures for a small
  synthetic dataset; `tests/test_loss_terms.py` hand-computed unit
  tests for `obs_loss`/`distance_loss`/`entropy`/`ent_loss` (also
  documents that `entropy` on a degenerate `[1.0, 0.0]` distribution
  currently returns NaN, not 0 — left as-is, out of scope); and
  `tests/test_golden_loss_fn.py`, a golden-value regression test
  freezing `loss_fn`/`train_model`'s exact loss history and fitted
  parameters for 5 training steps at a fixed RNG seed.
- **Step 2 (behavior-preserving cleanup)**: `ent_loss` was defined
  `ent_loss(probs, flows)` but called as `ent_loss(flows,
  pred_densities)` — the parameter names were swapped from what they
  actually held. Renamed to `ent_loss(flows, densities)` to match the
  call site; math and output unchanged (confirmed by the golden
  tests). `run_model.py`/`birdflow_demo.ipynb` were left untouched —
  both already broken independent of this work (see
  `dev/architecture.md`'s "Stale vs. current paths").
- **Step 3 (registry refactor)**: gave every loss term the shared
  signature `term_fn(pred_densities, flows, true_densities,
  d_matrices)`, introduced a `LOSS_TERMS` registry
  (`{name: (sign, term_fn)}`) in `flow_model_training.py`, changed
  `loss_fn` to take a single `weights` dict and return `(total,
  components_dict)` instead of a fixed 3-tuple, and made
  `train_model`'s `loss_dict` and `update_hdf.py`'s
  `metadata/hyperparameters`/`metadata/loss_values` HDF5 groups build
  themselves generically from whatever keys are present, instead of
  three hardcoded fields each. Confirmed behavior-preserving via the
  Step 1 golden tests (identical values) and a manual end-to-end run
  of `update_hdf.py` against a synthetic HDF5 fixture.
- **Step 4 (CLI mechanism + example term)**: added `flow_l2_loss` (L2
  penalty on the flow matrices) to `LOSS_TERMS` and a
  `--flow_l2_weight` CLI flag in `update_hdf.py`, default `0.0`, as a
  concrete smoke test that the registry + CLI mechanism works
  end-to-end for a genuinely new additive term. Covered by a unit
  test, golden-test additions proving weight `0.0` is a true no-op
  and a nonzero weight changes the loss, and a manual CLI run
  confirming `flow_l2_weight`/`flow_l2` show up in the output HDF5
  automatically.
- **Step 5 (validate nothing changed)**: re-ran the Step 1 golden
  suite against the fully refactored code — identical values. Also
  closed a gap: the Step 4 `update_hdf.py` end-to-end check had only
  been done manually in a scratch directory, not committed. Used a
  `git worktree` at commit `dce4646` (last commit before the registry
  refactor) to run the actual pre-refactor `update_hdf.py` against an
  identical synthetic HDF5 fixture, diffed every output key against a
  post-refactor run (identical except for the two new `flow_l2` keys,
  as expected), then wrote that as a permanent, committed
  subprocess-based pytest test
  (`tests/test_update_hdf_golden.py` + `synthetic_hdf5_fixture` in
  `tests/conftest.py`) so the check runs automatically going forward.
- **Step 6 (docs)**: rewrote `dev/architecture.md`'s loss-function
  section to describe the `LOSS_TERMS` registry mechanism instead of
  three hardcoded terms, added a "Three kinds of configurable
  parameter" section distinguishing registry-tracked loss weights
  (new) from other numeric parameters that aren't loss weights (e.g.
  `dist_pow`, which transforms data upstream of `loss_fn` and was
  never part of this refactor) from hypothetical switch-style variant
  selectors (not yet implemented), and added a "Tests" section
  describing the new test suite.

Also: scaffolded the `dev/` convention itself this session
(`dev/workplan.md`, `dev/worklog.md`, plus the pre-existing
`dev/architecture.md`), added `.DS_Store` to `.gitignore`, and
created this branch from `main`.

Final test count: 11 passing (`tests/test_loss_terms.py`,
`tests/test_golden_loss_fn.py`, `tests/test_update_hdf_golden.py`),
run via `.venv/bin/python -m pytest tests/`.

Outstanding before merge (not part of this branch's own work): running
this same regression suite inside `BirdFlowContainer` on Unity, per
the user's plan.
