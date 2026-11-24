# Known issues

Running list of issues identified during development but deliberately
deferred rather than fixed immediately — either low severity, or
needing a decision this doc hasn't made yet. Not a durable changelog;
when an issue is fixed, remove its entry (the commit/worklog is the
record of the fix) rather than marking it done here.

## `tests/test_update_hdf_golden.py` subprocess call is cwd-dependent

**Found:** 2026-11-16, reviewing branch `pluggable-loss-terms`.

`_run_update_hdf` in `tests/test_update_hdf_golden.py` invokes
`update_hdf.py` by a path relative to the current working directory:

```python
subprocess.run([sys.executable, "update_hdf.py", ...], ...)
```

This only resolves correctly when pytest is run with the repo root as
cwd (as documented: `.venv-unity/bin/python -m pytest tests/` from the
repo root). Running pytest from any other directory — a subdirectory, a
CI runner that `cd`s elsewhere first — makes this fail with
`FileNotFoundError`. The same underlying issue affects `conftest.py`'s
`from flow_model_training import ...` at collection time, confirmed by
running `pytest` from `/tmp`: `ModuleNotFoundError`.

**Fix (not yet applied):** resolve the script path relative to the
test file's own location, e.g. `Path(__file__).parent.parent /
"update_hdf.py"`, and/or add a `tests/conftest.py`
`sys.path`/cwd fixture so the suite doesn't depend on the invoker's
cwd. Low priority: doesn't affect the documented invocation, only a
less-common way of running it.

## Silent typo risk in `loss_fn`'s `weights.get(name, 0.0)`

**Found:** 2026-11-16, reviewing branch `pluggable-loss-terms`.

`loss_fn` (`flow_model_training.py`) looks up each `LOSS_TERMS` name in
the caller-supplied `weights` dict via `weights.get(name, 0.0)` — any
name not present defaults to `0.0` silently. A misspelled key (e.g.
`'entropy_weight'` instead of `'ent'`, verified concretely: produces
identical output to explicitly setting `ent: 0.0`, no exception or
warning) silently drops that term from the loss with no feedback.

For a model-fitting pipeline where a regularizer silently dropping out
changes the fitted model's qualitative behavior, this fails silently
rather than loudly. Mitigated today by `update_hdf.py` always printing
`str(args)` (so a CLI flag typo would show up as an unrecognized
argparse flag, not a silent weights-dict typo) — this is only a live
risk if `loss_fn`/`weights` is constructed some other way (e.g.
directly from Python, or if `update_hdf.py`'s weight-dict construction
is refactored later).

**Fix (not yet applied, needs a decision):** either (a) make
`loss_fn` validate `weights.keys() <= LOSS_TERMS.keys()` and raise on
an unrecognized key, or (b) leave as-is given the CLI already
surfaces typos via `argparse`'s own flag validation and the number of
weight names is small. Worth revisiting if the registry grows to many
more terms, or if `weights` is ever constructed somewhere other than
`update_hdf.py`'s explicit per-flag dict literal.

## `run_model.py` and `birdflow_demo.ipynb` are broken entry points

**Found:** pre-existing, documented in `dev/architecture.md`'s "Stale
vs. current paths" while writing that doc 2026-11-16.

Neither of these two entry points runs against the current code:

- `run_model.py` calls `process_data`, which constructs `Datatuple`
  with 5 positional values (`weeks, x_dim, y_dim, cells, nans`) against
  a 4-field namedtuple (`weeks, cells, distances, masks`) — raises
  `TypeError`. Both `run_model.py` and the notebook also import
  `gen_d_matrix` from `flow_model_training.py`, which no longer
  exists (removed in the same commit that introduced `mask_input`).
- `birdflow_demo.ipynb` additionally still calls the deprecated
  `jax.ops.index`/`jax.ops.index_update` API, removed from
  `flow_model.py` itself in an earlier commit (`43a7840`) but never
  updated in the notebook.

Both predate the dynamic-mask change (`64e4a65`) and represent the
older single-static-mask, raster/`.npy`-based workflow that
`update_hdf.py` superseded as the maintained entry point. They've been
left broken rather than fixed so far because nothing currently
depends on them — `update_hdf.py` is the only entry point the R
pipeline/Unity production path actually calls.

**Fix (not yet applied, needs a decision):** either (a) repair both —
fix `Datatuple` arity, regenerate a per-week `gen_d_matrix`-equivalent
compatible with `mask_input`, and update the notebook's JAX indexing
API — so the raster/`.npy`-based workflow and the grid-search use case
`run_model.py` was written for work again, or (b) delete both outright
if that workflow is no longer wanted now that `update_hdf.py`/HDF5 is
the live path. Worth deciding before the loss-term registry grows
further (e.g. the temporal-smoothing term from issue #7), since every
new `LOSS_TERMS` entry widens the gap between what these two stale
entry points assume and what the current code actually looks like.

## No pinned/tracked dependency versions for the production environment (issue #5)

**Found:** 2026-11-17, investigating [issue #5](https://github.com/birdflow-science/BirdFlowPy/issues/5)
(user-reported `module 'jax.extend.core' has no attribute 'JaxprEqn'`,
asking for the specific versions of `jax`/`haiku`/`optax`/`scipy` this
project needs).

**How the container works today:** `BirdFlowContainer.def` (the
separate repo that builds the production image used on Unity)
installs `jax[cuda12]`, `flax`, `dm-haiku`, `optax`, and `h5py` via
plain `pip install --upgrade`, with no version pins at all. There is
currently no tracked record anywhere of which dependency versions
BirdFlowPy actually needs or has been tested against in production —
"whatever pip resolved at the last container build" is the only
answer, and it silently changes on every rebuild. `BirdFlowContainer`
only supplies the shared environment (conda/Python, R, pinned
libraries) via `BirdFlowContainer.sif`; it does not bake in a copy of
this repo's code, so code and environment are decoupled per user —
see `dev/architecture.md`'s "Running on Unity" section.

**How dependencies are now listed:** `requirements.txt` at the repo
root records the exact versions confirmed installed in the container
as of 2026-11-17 (captured via `dev/print_container_versions.sh`), as
the single source of truth for BirdFlowPy's own pinned versions,
rather than a second, independently drifting file. It's also what
`.venv-unity` (the documented local dev/test environment — see
`dev/architecture.md`'s "Tests" section) installs from.

**Envisioned change to the container (not yet done):** actually
repoint `BirdFlowContainer.def` to install Python dependencies *from*
`requirements.txt` instead of its own inline `pip install --upgrade`
lines, so the container build itself pins to the same versions this
repo tracks. Deliberately not done yet, since that requires an actual
container rebuild, which is out of scope for now.

**How testing is done with `BirdFlowPipeline`:** `BirdFlowPipeline` (a
separate R package, not covered by this repo) is what actually drives
model fitting/evaluation *on Unity specifically*, including launching
parallel jobs, and it has its own test suite. Per
`BirdFlowContainer`'s README ("Rebuilding the BirdFlowContainer
apptainer" step 3), running `devtools::test()` from the
`BirdFlowPipeline` RStudio project against a freshly built container
is already the team's standing validation step before trusting a
rebuild — the closest thing to CI this stack has today, though it's
manual, Unity-specific, and runs only at rebuild time rather than
continuously. The project owner confirmed `BirdFlowPipeline`'s suite
runs full model fits (so it would likely surface a hard dependency
break mid-fit) but doesn't exercise every BirdFlowPy code path, and
isn't rigorous about snapshotting fitted output for regressions — so
it validates dependency bumps/container rebuilds reasonably well, but
isn't a substitute for this repo's own `tests/` when it comes to
verifying BirdFlowPy's own changes.

**Fix (not yet applied, needs a decision):** use `BirdFlowPipeline`'s
test suite as the deliberate gate for any pin bump, not just container
rebuilds for R's sake — run `devtools::test()` against the new
container before adopting a new pin set, and treat a failure as "don't
ship this pin." Also worth deciding: whether to record the resolved
version set every time the container is rebuilt (e.g. commit
`dev/print_container_versions.sh`'s output alongside
`BirdFlowContainer.def`'s change), so "what did we actually get this
time" has a durable record instead of living only in whoever did the
rebuild's memory.

## Record the package version in HDF5 output metadata

**Requested:** 2026-11-16, by the project owner (not a review finding).

`update_hdf.py` writes a `metadata/hyperparameters` group and a
`metadata/birdflow_model_date` field to the output HDF5, but nothing
records which version of this codebase (`_version.py`'s `__version__`,
added alongside `CHANGELOG.md`) produced the file. For reproducing or
debugging an old model run — especially once loss terms/CLI flags
start changing between versions — there's currently no way to tell
from the HDF5 file alone which code version generated it.

**Fix (not yet applied):** in `update_hdf.py`, `from _version import
__version__` and write it as a dataset under `metadata/` (e.g.
`metadata/birdflow_py_version`), alongside the existing
`birdflow_model_date` write. Should also be covered by
`tests/test_update_hdf_golden.py`'s expected-output assertions once
added.

## `LOSS_TERMS` scale inconsistently with active-cell count, which weakens weight transfer across species

**Found:** 2026-11-20, planning the issue #7 temporal-smoothing term
(`dev/workplan.md`). **Updated:** 2026-11-23, to add the second
(previously missing) scaling effect below, per the project owner.
**Updated:** 2026-11-23 (later), to fold `smooth_loss` into the main
discussion now that it's implemented rather than prospective, and to
note `flow_l2_loss`'s lower practical priority as a dummy term.

`obs_loss`, `distance_loss`, `ent_loss`, `flow_l2_loss`, and
`smooth_loss` (`flow_model_training.py`) are all unnormalized
`jnp.sum`s, but they don't scale the same way with a species'
active-cell count `C` (which at this project's standard fixed 100km
resolution varies enormously species to species). Two effects pull in
opposite directions as `C` grows, and this entry previously only
covered the first:

1. **More cells, more summed terms** — a bigger `C` means more entries
   get added into the sum, which on its own would push a term's total
   up.
2. **Smaller values per cell/entry** — `obs_loss`, `flow_l2_loss`, and
   `smooth_loss` all operate on *relative* abundance: each week's
   density sums to ~1 across `C` cells, so *mean* per-cell abundance
   is `1/C` by construction (not an empirical tendency — a direct
   consequence of fitting to relative abundance). `flow_l2_loss`'s
   flow matrix is similarly constrained (its entries sum to ~1 in
   total across ~`C`×`C` cells), so its mean entry shrinks faster
   still, ~`1/C²`. This pulls each squared term down as `C` grows,
   which on its own would push a term's total down. (`ent_loss` also
   operates on relative abundance, but entropy's `-p·log(p)` doesn't
   shrink the same way a squared term does — its own distinct
   cell-count behavior is covered in its bullet below.)

Which effect wins depends on the term:

- **`distance_loss`** — already addressed; this one's scaling was
  fixed previously (not re-verified here, per the project owner).
- **`ent_loss`** — bounded by `log(C)`, since entropy of a
  distribution over `C` cells can't exceed `log(C)`. Logarithmic, not
  linear — the mildest of the terms below, already close to
  self-normalizing in practice.
- **`obs_loss`**, **`flow_l2_loss`**, and **`smooth_loss`** — all sums
  of squared entries from a vector/matrix that's itself constrained to
  sum to ~1 (density, flow mass, or — for `smooth_loss` — a second
  difference of density, which inherits the same `1/C`-scale from the
  densities it's built from). If a species' residuals/flow mass/
  curvature were spread evenly across all `C` cells, the `1/C` (or
  `1/C²`) mean-value effect above would dominate: each squared term
  shrinks quadratically while only `C` (or, for `smooth_loss`, `C`
  times the number of interior weeks) terms are added, so the total
  would shrink roughly like `1/C` for `obs_loss`/`smooth_loss` and
  faster for `flow_l2_loss`. At the other extreme, a species whose
  real density concentrates into a roughly fixed number of hotspot
  cells regardless of total range size would see the opposite:
  growing `C` mostly adds near-zero cells that barely move the sum,
  so the total stays roughly constant in `C`. Not determinable from
  the formulas alone — real BirdFlow densities are probably somewhere
  between the two extremes, and which regime dominates is an open
  empirical question (see "Note on empirically estimating scaling"
  below). Of these three, `flow_l2_loss` is lowest priority to
  actually resolve: it's a dummy term added only to exercise the
  `LOSS_TERMS` registry mechanism (see `dev/architecture.md`), not one
  in scientific use, so its scaling behavior has no practical
  consequence today.

This matters because weights aren't always refit per species: the
project owner confirmed a substantial chunk of species get their loss
weights *transferred* from a similar species rather than independently
tuned via grid search. A transferred weight implicitly assumes the
term it multiplies means roughly the same thing (same effective
strength) on both species — which isn't guaranteed when a term's raw
magnitude depends on `C` in an uncontrolled way.

**Fix (not yet applied, needs a decision, and needs the empirical
question above answered first):** once the actual scaling behavior of
`obs_loss`/`ent_loss`/`smooth_loss` (and, lower priority,
`flow_l2_loss`) is known, decide whether to normalize some or all of
them together for consistency and better transfer behavior. Doing
this for any one of them in isolation risks making things worse — for
example, `smooth_loss` is in the same family as `obs_loss` and
competes against it directly in the objective, so normalizing
`smooth_loss` alone while leaving `obs_loss` raw could decouple their
relative scaling and make the obs/smooth *ratio* (which is what
actually governs the fitted model's smoothness-vs-fidelity tradeoff)
transfer worse across species, not better — so if a fix happens, it
should happen as one coordinated change across all the same-family
terms, not piecemeal. This would also be a breaking change to the
meaning of existing `obs_weight`/`ent_weight`/`flow_l2_weight`/
`smooth_weight` values and would require re-deriving or re-fitting
weights for every species currently using transferred values. (This
is why `smooth_loss` was deliberately left unnormalized rather than
implementing an earlier plan to normalize it by `T * total_cells` —
see `dev/workplan.md` for that fuller discussion.)

**Note on empirically estimating scaling:** the project has a large
trove of existing grid-search output with performance metrics per
fitted model, spanning many species (and therefore many active-cell
counts) at the fixed 100km resolution. That data could be mined
directly to estimate how `obs_loss`/`ent_loss`/`flow_l2_loss` actually
scale with `C` in practice, rather than reasoning about it from the
loss formulas alone — turning the open empirical question above into
a concrete analysis rather than a guess, before committing to any
normalization change. `smooth_loss` didn't exist when that historical
grid-search data was generated, so its own scaling would need to be
estimated separately (e.g. by computing it post hoc from those same
fitted models' marginals, or from new runs).

## `smooth_loss` treats absence and insufficient data as the same "true zero"

**Found:** 2026-11-20, implementing the issue #7 temporal-smoothing
term (`dev/workplan.md`, `dev/architecture.md`).

`smooth_loss` (`flow_model_training.py`) embeds each week's compact
marginal into the full canonical cell space via that week's
`dynamic_mask`, treating any cell outside the mask as **true zero**
density for that week. Per the project owner, a cell can land outside
a week's `dynamic_mask` for three different reasons that the mask alone
doesn't distinguish:

1. The underlying eBird S&T model predicted (near-)zero density there
   — a known true zero.
2. eBird S&T didn't have enough data to make a prediction at all (the
   owner's example: much of the Amazon basin) — unknown, not zero,
   even though it's recorded the same way as case 1 today.
3. The location is simply unsuitable for the species (e.g. ocean for
   a forest bird) — genuinely zero, but conflated with case 2 in the
   eBird output.

The project is aware of this distinction (it now tracks whether a
model made a prediction at all, separately from the prediction's
value), which allows identifying case 1 zeros, but doesn't resolve
whether a missing prediction is case 2 or case 3.

`smooth_loss` doesn't use that finer-grained information —
it was deliberately built simple first, per the owner's explicit
direction, with true-zero for all three cases as the starting point.

**Fix (not yet applied, needs a decision):** the owner floated that
`smooth_loss` could instead help *impute* the no-prediction cells
(cases 2 and 3) — e.g. by treating their curvature as unconstrained
(no penalty either way) rather than penalizing it against an assumed
zero, or by letting the smoothness term itself interpolate a plausible
value for a cell that's missing one week but present on either side.
Complicated by the fact that cases 2 and 3 aren't currently
distinguishable from each other (both just show up as "no prediction
made"), and only case 3 is actually a candidate for imputation —
case-2 cells are genuinely unknown, not quantities to be inferred. The
owner's intuition is that `smooth_loss` itself might still have useful
information to bring to bear here despite that ambiguity, but working
out how isn't being pursued now. Any fix would need the finer-grained
"did the model run here at all" signal plumbed through from the HDF5
into `smooth_loss` (currently only the single boolean `dynamic_mask`
is passed in) at minimum, on top of resolving the above.
