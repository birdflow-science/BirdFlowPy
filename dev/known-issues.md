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

`BirdFlowContainer.def` (the separate repo that builds the production
image used on Unity) installs `jax[cuda12]`, `flax`, `dm-haiku`,
`optax`, and `h5py` via plain `pip install --upgrade`, with no version
pins at all. There is currently no tracked record anywhere of which
dependency versions BirdFlowPy actually needs or has been tested
against in production — "whatever pip resolved at the last container
build" is the only answer, and it silently changes on every rebuild.

This repo's original local test environment (`requirements-dev.txt`,
since removed — see "Update 2026-11-17" below) was pinned, but
deliberately to an *older*, Python 3.9-compatible stack for local
development — not verified against, and found to diverge sharply
from, the actual production environment. Running
`dev/print_container_versions.sh` (added this session) inside the
current container on Unity vs. that old local environment gave:

| Package | Old local env (Python 3.9) | Unity container (Python 3.13.9) |
|---|---|---|
| jax / jaxlib | 0.4.18 | 0.8.0 |
| haiku | 0.0.10 | 0.0.15 |
| optax | 0.1.7 | 0.2.6 |
| scipy | 1.10.1 | 1.16.2 |
| h5py | 3.14.0 | 3.15.1 |
| numpy | 1.26.4 | 2.3.4 |

The numpy divergence was the sharpest: the old environment pinned
`numpy<2` specifically for Python-3.9 compatibility, while production
is on numpy 2.3.4. This meant `pluggable-loss-terms`'s golden-value
regression tests (`tests/test_golden_loss_fn.py`,
`tests/test_update_hdf_golden.py`) had only ever run against the
numpy-1.x/jax-0.4/Python-3.9 combination locally, never the
numpy-2.x/jax-0.8/Python-3.13 combination production actually runs.
See "Update 2026-11-17" below for how this was resolved.

Separately, issue #5's reported error
(`jax.extend.core.JaxprEqn` missing) looks like a plausible
haiku/jax internal-API incompatibility at the *current* container's
versions (haiku 0.0.15 against jax 0.8.0) rather than a user
environment-setup mistake — worth checking by actually running
BirdFlowPy's training/fitting path inside the current container,
since the container as currently built may be broken for model
fitting regardless of this branch.

### Broader context: why this differs from BirdFlowR, and the existing validation path

Discussed 2026-11-17 with the project owner. BirdFlowR tracks
latest R/CRAN continuously, validated by its own CI on every PR and
merge to main — that's viable because CRAN runs reverse-dependency
checks before a package release ships, catching most breakage before
it reaches consumers. PyPI has no equivalent gatekeeper: any package
(jax very much included) can break its own API in a minor version
with nothing catching it centrally, which is exactly the shape of
issue #5. BirdFlowPy is also public but far less supported than
BirdFlowR — the team's actual production use case is fitting and
releasing models once, which users then consume through BirdFlowR,
not continuous external development on BirdFlowPy itself. Floating
dependency versions with no CI (today's state: `BirdFlowContainer.def`
runs unpinned `pip install --upgrade`) is the one combination that
doesn't work: no gatekeeper and no validation.

`BirdFlowContainer` is rebuilt rarely (every couple of years, last
triggered by R falling behind) and shared identically by the whole
team, so a silent Python dependency change inside a routine rebuild is
a real risk, not just a tidiness concern.

Important existing piece: **`BirdFlowPipeline`** (a separate R
package, not covered by this repo) is what actually drives model
fitting/evaluation *on Unity specifically*, including launching
parallel jobs, and it has its own test suite. Per `BirdFlowContainer`'s
README ("Rebuilding the BirdFlowContainer apptainer" step 3), running
`devtools::test()` from the `BirdFlowPipeline` RStudio project against
a freshly built container is already the team's standing validation
step before trusting a rebuild. That makes it the closest thing to CI
this stack has today — it's just manual, Unity-specific, and runs only
at rebuild time rather than continuously.

Confirmed 2026-11-17 by the project owner: (1) `BirdFlowPipeline`'s
suite does **not** exercise this branch's new features (the
`LOSS_TERMS` registry, `flow_l2`, the new HDF5 output fields) — only
the pre-registry path. (2) It does run full model fits, so it would
likely surface a hard dependency break (e.g. issue #5's error) that
occurs mid-fit; less clear is how rigorously it snapshots/compares
*fitted model output* across runs, so it's a better bet for "does
fitting still run at all" than "did the fitted result subtly change."
This means it validates dependency bumps/container rebuilds
reasonably well, but isn't a substitute for this branch's own test
suite (`tests/`) when it comes to verifying the loss-term-registry
refactor itself.

Given that, and that the project owner is currently the only user of
this branch's new features (in active development on them, so bugs
there are expected and tolerable), the stated priority for merging
`pluggable-loss-terms` is narrower than "fully verified in
production": don't break the existing (pre-registry) workflow, while
opening room to develop new loss terms. That's a materially lower bar
than requiring `BirdFlowPipeline`'s suite to validate the new
features before merge — it can't, by design, so merge-readiness should
be judged by `tests/`'s own golden-value regression tests (proving the
refactor doesn't change pre-existing behavior) rather than waiting on
`BirdFlowPipeline` coverage that was never going to arrive for this
branch.

**Fix (not yet applied, needs a decision) — proposed direction:**

1. **Make BirdFlowPy's own pinned versions the single source of
   truth**, rather than two independently drifting files. Added
   `requirements.txt` at the repo root (2026-11-17) recording the
   table above as explicit pins. Still open: actually repointing
   `BirdFlowContainer.def` to install Python dependencies *from*
   `requirements.txt` instead of its own inline `pip install --upgrade`
   lines — deliberately not done yet, since that requires an actual
   container rebuild, which is out of scope for now.
2. **Freeze, don't float** — given there's no CI continuously
   validating BirdFlowPy against new dependency releases, floating
   buys nothing but risk; pin explicitly and bump only on a concrete
   trigger (a specific feature need, Unity/CUDA forcing a bump, PyPI
   dropping wheels for the current Python/CUDA combo) rather than on
   a schedule. This mirrors how rarely the container is rebuilt today,
   just makes the version set stable *between* rebuilds too, not only
   at rebuild time.
3. **Use `BirdFlowPipeline`'s test suite as the deliberate gate for
   any pin bump**, not just container rebuilds for R's sake — run
   `devtools::test()` against the new container before adopting a new
   pin set, and treat a failure as "don't ship this pin," the same way
   a BirdFlowR CI failure blocks a merge. Confirm first whether that
   suite actually covers the loss-term-registry/HDF5 changes on this
   branch; if not, extending it (or adding a narrower Python-side smoke
   test) is worth a separate decision.
4. **Record the resolved version set every time the container is
   rebuilt** (e.g. commit `dev/print_container_versions.sh`'s output
   alongside `BirdFlowContainer.def`'s change), so "what did we
   actually get this time" has a durable record instead of living only
   in whoever did the rebuild's memory — directly answering requests
   like issue #5's.

Should also settle whether `pluggable-loss-terms`'s own golden tests
need regenerating against whatever version set gets pinned
(floating-point results can shift across jax/numpy versions even
absent an API break) before this branch is considered verified for
merge.

### Update 2026-11-17: local environment now matches the container; golden tests regenerated

Installed Python 3.13.18 locally (Homebrew) and created `.venv-unity`,
`pip install -r requirements.txt pytest` — every package resolved to
exactly the version pinned above, i.e. nothing in `requirements.txt`'s
pin set conflicts with what's actually installable today. Ran
`tests/` under `.venv-unity`: no import or API error (unlike issue
#5's report) — `tests/test_loss_terms.py`'s hand-computed unit tests
(no RNG) passed unchanged, but the three golden-value/snapshot tests
(`test_golden_loss_dict`, `test_golden_fitted_params`,
`test_update_hdf_golden_output`) failed, with numeric divergences too
large to be rounding (e.g. a sign flip in a fitted parameter).

Cross-checked via a `git worktree` at commit `dce4646` (the last
commit before the `LOSS_TERMS` registry refactor, predating this
branch entirely) running the same golden-value tests under
`.venv-unity`: identical failure pattern. This confirms the
divergence is RNG/dependency drift between jax 0.4.18 and 0.8.0 (most
likely jax's PRNG stream implementation, used by
`hk.initializers.RandomNormal()`), not anything introduced by the
registry refactor. Regenerated the golden values in
`tests/test_golden_loss_fn.py` and `tests/test_update_hdf_golden.py`
against `.venv-unity`; full suite (17 tests) now passes under
`.venv-unity`.

Per the project owner's decision, dropped the old local dev
environment entirely rather than keeping both: deleted
`requirements-dev.txt` and `.venv/`, added `.venv-unity` to
`.gitignore`. `.venv-unity`/`requirements.txt` is now the only
documented local dev/test environment (see `dev/architecture.md`'s
"Tests" section) — it can no longer run the now-stale old golden
values (that environment doesn't exist anymore), which is fine since
it was superseded rather than kept as an alternative.

### Update 2026-11-17 (later): confirmed directly inside BirdFlowContainer on Unity

Project owner ran `tests/` from an RStudio OnDemand terminal, i.e.
directly inside `BirdFlowContainer.sif` rather than the local
`.venv-unity` approximation: all 17 tests passed. This confirms
`requirements.txt`'s pins genuinely match the container (not just a
best-effort guess from `print_container_versions.sh`'s output) and
that the regenerated golden values hold in the real production
environment. Also ran `BirdFlowPipeline`'s suite against this branch
(point 3 of the proposed direction below): passed. See
`dev/worklog.md`'s "Verified on Unity" entry for the full writeup.

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
(`dev/workplan.md`).

`obs_loss`, `distance_loss`, `ent_loss`, and `flow_l2_loss`
(`flow_model_training.py`) are all unnormalized `jnp.sum`s, but they
don't scale the same way with a species' active-cell count `C` (which
at this project's standard fixed 100km resolution varies enormously
species to species):

- **`distance_loss`** — already addressed; this one's scaling was
  fixed previously (not re-verified here, per the project owner).
- **`ent_loss`** — bounded by `log(C)`, since entropy of a
  distribution over `C` cells can't exceed `log(C)`. Logarithmic, not
  linear — the mildest of the four, already close to self-normalizing
  in practice.
- **`obs_loss`** and **`flow_l2_loss`** — both sums of squared entries
  from a vector/matrix that's itself constrained to sum to ~1
  (density, flow mass). How these scale with `C` depends on whether a
  species' density stays equally "spread out" relative to its own
  range as `C` grows (in which case these shrink roughly like `1/C`)
  or concentrates into a roughly fixed number of hotspot cells
  regardless of total range size (in which case they stay roughly
  constant in `C`). Not determinable from the formulas alone — real
  BirdFlow densities are probably somewhere between the two extremes,
  and which regime dominates is an open empirical question (see "Note
  on empirically estimating scaling" below).

This matters because weights aren't always refit per species: the
project owner confirmed a substantial chunk of species get their loss
weights *transferred* from a similar species rather than independently
tuned via grid search. A transferred weight implicitly assumes the
term it multiplies means roughly the same thing (same effective
strength) on both species — which isn't guaranteed when a term's raw
magnitude depends on `C` in an uncontrolled way.

The new `smooth_loss` term being added for issue #7 (a sum of squared
second differences on a density-shaped vector — the same mathematical
family as `obs_loss`/`flow_l2_loss`) is, for now, **also left
unnormalized** (raw `jnp.sum`), explicitly deferring the earlier plan
to normalize it by `T * total_cells`. Reasoning: since `smooth_loss`
is in the same family as `obs_loss`, it likely scales with `C`
similarly to `obs_loss` — and `obs_loss` is the term `smooth_loss`
most directly competes against in the objective. Normalizing only
`smooth_loss` while leaving `obs_loss` raw could decouple their
relative scaling and make the obs/smooth *ratio* (which is what
actually governs the fitted model's smoothness-vs-fidelity tradeoff)
transfer worse across species, not better. What matters for transfer
is that ratio staying stable across species, not any single term
being independently cell-count-invariant. See
`dev/workplan.md` for the fuller discussion.

**Fix (not yet applied, needs a decision, and needs the empirical
question above answered first):** once the actual scaling behavior of
`obs_loss`/`ent_loss`/`flow_l2_loss` (and prospectively `smooth_loss`)
is known, decide whether to normalize some or all of them together
for consistency and better transfer behavior. Doing this for any one
of them in isolation risks making things worse, per the reasoning
above — if it happens, it should happen as one coordinated change
across all the same-family terms. This would also be a breaking
change to the meaning of existing `obs_weight`/`ent_weight`/
`flow_l2_weight` values and would require re-deriving or re-fitting
weights for every species currently using transferred values.

**Note on empirically estimating scaling:** the project has a large
trove of existing grid-search output with performance metrics per
fitted model, spanning many species (and therefore many active-cell
counts) at the fixed 100km resolution. That data could be mined
directly to estimate how `obs_loss`/`ent_loss`/`flow_l2_loss` actually
scale with `C` in practice, rather than reasoning about it from the
loss formulas alone — turning the open empirical question above into
a concrete analysis rather than a guess, before committing to any
normalization change.

## `smooth_loss` treats absence and insufficient data as the same "true zero"

**Found:** 2026-11-20, implementing the issue #7 temporal-smoothing
term (`dev/workplan.md`, `dev/architecture.md`).

`smooth_loss` (`flow_model_training.py`) embeds each week's compact
marginal into the full canonical cell space via that week's
`dynamic_mask`, treating any cell outside the mask as **true zero**
density for that week. Per the project owner, a cell can land outside
a week's `dynamic_mask` for two different reasons that the mask alone
doesn't distinguish:

1. The underlying eBird S&T model predicted (near-)zero density there,
   or the location is simply unsuitable for the species (e.g. ocean for
   a forest bird) — genuinely zero, confidently.
2. eBird S&T didn't have enough data to make a prediction at all (the
   owner's example: much of the Amazon basin) — unknown, not zero,
   even though it's recorded the same way as case 1 today.

The project is aware of this distinction (it now tracks whether a
model made a prediction at all, separately from the prediction's
value) but `smooth_loss` doesn't use that finer-grained information —
it was deliberately built simple first, per the owner's explicit
direction, with true-zero for both cases as the starting point.

**Fix (not yet applied, needs a decision):** the owner floated that
`smooth_loss` could instead help *impute* case-2 gaps — e.g. by
treating unknown-cell curvature as unconstrained (no penalty either
way) rather than penalizing it against an assumed zero, or by letting
the smoothness term itself interpolate a plausible value for a
cell that's missing one week but present on either side. This would
need the finer-grained "did the model run here at all" signal plumbed
through from the HDF5 into `smooth_loss` (currently only the single
boolean `dynamic_mask` is passed in), and a decision on the specific
mechanism, before implementing.
