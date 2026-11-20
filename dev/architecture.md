# BirdFlowPy architecture

This is a from-scratch summary of how the top-level scripts fit together,
written for someone who knows the BirdFlowR model/data structures but not JAX
or Haiku. It was initially based on the state of the code as of commit
`de0d77d` (2026-11-16). Subsequent updates:

* Loss-term registry refactor on branch `pluggable-loss-terms` (2026-11-17)
* `smooth` (temporal-smoothness) loss term added on branch
  `7-temporal-smoothing` (edited 2026-11-21)

Detailed plans are stored within each branch in `dev/workplan.md` and later
moved to the top of `dev/worklog.md` before merging.

## The four files and how they relate

| File                     | Role                                                                 |
|---------------------------|----------------------------------------------------------------------|
| `flow_model.py`           | Model definition (the Markov chain) + inference/sampling utilities  |
| `flow_model_training.py`  | Data prep, loss function, training loop                             |
| `update_hdf.py`            | **Current** entry point: reads/writes the HDF5 file shared with BirdFlowR |
| `run_model.py`             | **Stale** entry point: raster/`.npy`-based workflow, kept for the grid-search use case it was written for |

`birdflow_demo.ipynb` is also stale — see "Stale vs. current paths" below.

## Conceptual model (same math as BirdFlowR)

Each bird's location is a cell index in a weekly, time-inhomogeneous discrete
Markov chain: an initial distribution `d0` over cells in week 1, and a
transition distribution for week *t* → *t+1* for every week. That's exactly
BirdFlowR's `distr` + per-transition probability structure. The thing being
fit is the full set of transition probabilities (the `θ` in the schematic),
trained to reproduce eBird S&T's weekly abundance ("distr") while penalizing
biologically implausible long-distance flow and encouraging smooth/non-degenerate
distributions.

The novelty in this repo, relative to a hand-rolled optimizer, is that the
whole forward simulation (initial distribution → week-by-week flow) is
written as a differentiable JAX computation, so standard gradient-based
optimizers (Adam, via `optax`) can fit all the transition matrices jointly
against a single scalar loss.

## Weeks are a linear chain, not a circular one

Though BirdFlow models an annual migratory cycle, nothing in this codebase
treats the sequence of weeks it receives as a true ring — `BirdFlowR::
preprocess_species()` already linearizes the annual cycle before this code
ever sees the data. For a full-year model, it duplicates week 1's target
distribution as an extra final week, so what arrives here is a linear
chain whose real annual wrap is encoded entirely in that duplicated final
week, not something any function in this repo needs to reconstruct by
wrapping indices itself. The rarer non-full-year, subset-of-weeks case is
likewise a genuine, shorter linear span with no wrap either way.

This isn't just a convenience for any one function — it's what lets
`FlowModel` itself (`flow_model.py`) be a plain linear chain, with no
circular-transition code anywhere in this repo. `FlowModel.__call__`
builds exactly `num_weeks - 1` `FlowBlock` transitions, walking forward
from `d0` (week 1) to the last week and never back; there's no block
anywhere computing a last-week-to-week-1 transition. If the annual wrap
weren't already linearized upstream, that transition would have to be
modeled explicitly — an extra `FlowBlock` (or equivalent) closing the
loop back to week 1 — rather than only showing up implicitly through
loss terms that compare across weeks.

Within that already-linear model, cross-week *loss* terms still have
their own boundary question: does a term that compares multiple weeks'
values treat week 0 and the last week as adjacent (circular) or not
(open)? Currently `smooth_loss` (see "Loss components" below) is the
only loss term that looks across weeks rather than operating on each one
independently, so it's the only place this choice shows up today. No
cross-week term is computed at week 0 or the last week. Any future loss
term (or other code) that looks across adjacent weeks should follow this
same open-boundary convention rather than reintroduce circularity.

## `flow_model.py` — model definition

### Haiku, `hk.Module`, and parameters

[Haiku](https://github.com/google-deepmind/dm-haiku) is DeepMind's thin
neural-network library on top of JAX. The relevant mental model, if you're
coming from R:

- JAX functions are normally **pure** (no mutable state). Haiku's
  `hk.Module` classes look like stateful objects (`self.cells`, etc.) but
  `hk.get_parameter(...)` doesn't store the parameter on the object — it
  registers a named slot in an external, nested dict that Haiku builds for
  you. That dict *is* the model's parameter set — conceptually equivalent to
  a flat list of the transition-matrix logits BirdFlowR would store.
- `hk.transform(predict)` takes an ordinary function `predict` that calls
  modules and returns two pure functions: `.init(rng, *args)` (allocate and
  randomly initialize all parameters, returning that nested dict) and
  `.apply(params, rng, *args)` (run the forward computation given a
  concrete params dict). `model_forward` at the bottom of `flow_model.py`
  is this transformed pair.
- Parameter dict keys are built from the `name=` strings passed to
  `super().__init__(name=...)` in each module, joined with `/`. That's why
  you see lookups like `params['Flow_Model/Week_7']['z']` elsewhere in the
  code — it's indexing into that nested dict by module path.

### The modules

- **`InitialLoc`** — one learned vector `z0` of length `cells[0]` (raw,
  unconstrained reals). `softmax(z0)` turns it into the week-1 distribution
  `d0`. Softmax is the reparameterization trick that keeps the distribution
  valid (non-negative, sums to 1) while leaving the underlying parameter
  unconstrained, which is what gradient descent needs.
- **`FlowBlock`** — one learned matrix `z` of shape `(cells_t, cells_{t+1})`
  per week-to-week transition. `softmax(z, axis=1)` makes each row a valid
  categorical distribution (row *i* = probability of going from cell *i* in
  week *t* to each cell in week *t+1*). Multiplying by the incoming mass
  (`last_week.reshape(-1, 1)`) converts "transition probabilities" into
  "flow amounts" — an unnormalized cells_t × cells_{t+1} matrix of mass
  moved. Each block is named `Week_{n}` so its parameters land at a
  predictable key.
- **`FlowModel`** — wires these together: get `d0` from `InitialLoc`, then
  for each week run a `FlowBlock`, summing each flow matrix's columns
  (`flow.sum(axis=0)`) to get the next week's marginal distribution, which
  feeds the next block. Returns `(d0, flow_amounts)` — the initial
  distribution plus the list of per-week flow matrices. Note `self.cells` is
  a **list**, one entry per week, not a single scalar — this is what lets
  the number of active (non-NaN / in-range) cells differ week to week
  (BirdFlowR's "dynamic mask" concept).
- **`predict` / `model_forward`** — the `hk.transform`-ed entry point used
  everywhere else (`model_forward.init(...)`, `model_forward.apply(...)`).

### Post-fit utilities (bottom half of the file)

These all operate on an already-fit `params` dict, not on the Haiku modules
directly:

- `get_prob` — look up a single transition probability `P(cell p2 in week+1
  | cell p1 in week)`.
- `sample_trajectory` / `gen_samples` / `aggregate_samples` — Monte Carlo
  sample individual bird tracks from the fitted chain (categorical draws
  week by week), the stochastic-trajectory analogue of BirdFlowR's route
  simulation.
- `project_density` — same forward recursion as `FlowModel.__call__`, but
  starting from an arbitrary distribution rather than the fitted `d0` — used
  for forecasting from a known/assumed current location.
- `gen_shift_list` / `shift_density` — bookkeeping to go from "masked,
  NaN-stripped cell index" back to "full raster cell index" for plotting.

## `flow_model_training.py` — loss and optimization

### Data prep: two parallel, inconsistent code paths

- `process_data(data_array)` — takes a raw `(weeks, y, x)` raster array,
  flattens it, drops NaN cells (using a single **static** mask from week 1
  only), and returns normalized weekly densities + a `Datatuple`. This is
  the older, single-static-mask workflow.
- `mask_input(true_densities, dtuple)` — takes densities that are already
  in `Datatuple` form with a **per-week** `masks` array (`dynamic_mask` from
  the HDF5 file) and a flat pairwise-distance vector, and produces, for each
  week transition, a masked distance sub-matrix and masked density vector
  sized to that week's active cells. This is the current, dynamic-mask
  workflow that matches `FlowModel`'s per-week cell counts.

  ⚠️ These two paths are no longer consistent with each other: `Datatuple`
  is currently defined with 4 fields (`weeks, cells, distances, masks`), but
  `process_data` still constructs it with 5 positional values
  (`weeks, x_dim, y_dim, cells, nans`) — left over from before the dynamic-mask
  change. Calling `process_data` as-is will raise a `TypeError`. `run_model.py`
  and `birdflow_demo.ipynb` both depend on this broken path (see below).

### Loss components: a pluggable registry

Loss terms are no longer hardcoded into `loss_fn`'s signature and return
value. Every term function shares one signature —
`term_fn(pred_densities, flows, true_densities, d_matrices, masks) ->
scalar` — even where it ignores most of the arguments, and a
module-level dict, `LOSS_TERMS`, maps a term's name to `(sign, term_fn)`:

```python
LOSS_TERMS = {
    'obs': (1.0, obs_loss),
    'dist': (1.0, distance_loss),
    'ent': (-1.0, ent_loss),
    'flow_l2': (1.0, flow_l2_loss),
    'smooth': (1.0, smooth_loss),
}
```

`sign` is `+1` for a term that's penalized (pushes the loss up — obs,
dist, flow_l2, smooth) and `-1` for a term that's rewarded (entropy;
higher entropy is desirable, so it's *subtracted*). `loss_fn(params,
cells, true_densities, d_matrices, weights, masks)` takes a `weights`
dict (`{'obs': ..., 'dist': ...}`, one float per term name you want
nonzero — any name not present defaults to weight `0.0`) and loops over
`LOSS_TERMS`, computing every term's value and summing `sign * weight *
value` into the total. It returns `(total, components)`, where
`components` is a dict of every term's raw (unweighted) value — so
`train_model`'s `loss_dict` history has one list per term name,
discovered from whatever keys the dict actually contains, rather than
three hardcoded fields.

The individual terms:

- `obs_loss` — sum of squared residuals between predicted and true weekly
  densities (fit term).
- `distance_loss` — `sum(flow * distance_matrix)` per week, i.e. a penalty
  proportional to how much mass moves how far, where "far" is whatever
  distance/cost matrix was supplied (raised to `dist_pow` and optionally
  normalized before this function ever sees it — that happens in the
  calling script; see "Three kinds of configurable parameter" below).
- `entropy` / `ent_loss` — joint entropy of the flow matrices minus the
  entropy of the marginal densities; this is the same
  "marginal-minus-joint" entropy regularizer that shows up in BirdFlowR's
  math for discouraging degenerate (too-concentrated or too-diffuse)
  transition distributions.
- `flow_l2_loss` — a dummy term, added only to demonstrate and test the
  modularity introduced for the upcoming 1.2.0 release not for
  scientific use. Its default weight is `0.0`, i.e. off.
- `smooth_loss` — a second-difference (curvature) penalty on the model's
  own fitted weekly marginals, added for
  [issue #7](https://github.com/birdflow-science/BirdFlowPy/issues/7) to
  discourage spurious week-to-week movement driven by noise in the input
  abundance rasters: a single noisy week can no longer unilaterally force
  a detour in the fitted model without also increasing this term.

  Equivalently, per cell: linearly extrapolate abundance from weeks
  `t-1` and `t` to predict week `t+1`, then square the difference
  between that extrapolated value and the actual (fitted) value at
  `t+1`, summed across all cells and all interior weeks. A cell whose
  abundance changes at a constant rate (steady migratory drift)
  extrapolates perfectly and costs nothing; a noise-driven
  spike-then-revert does not extrapolate and costs more.

  It uses an **open boundary** (no term at week 0 or the last week) —
  see "Weeks are a linear chain, not a circular one" above for why. Like
  `flow_l2_loss`, its default weight is `0.0`, i.e. off. Unlike the
  normalized penalty sketched in issue #7's discussion, this
  implementation sums raw (unnormalized) squared curvature, matching
  `obs_loss`/`ent_loss`/`flow_l2_loss`'s existing convention rather than
  introducing a differently-scaled term — see `dev/known-issues.md`'s
  entry on `LOSS_TERMS` cell-count scaling for why, and why it's a
  cross-term decision rather than one to make for `smooth_loss` alone.

  **Mask semantics**: a cell outside a given week's `dynamic_mask` can
  mean either "the species genuinely isn't there" or "eBird S&T didn't
  have enough data to predict either way" — `smooth_loss` currently
  can't distinguish these and treats both as true zero. See
  `dev/known-issues.md` for the deferred follow-up (using the smoothing
  term to help impute genuine data gaps) and why true-zero was chosen as
  the simpler starting point.

Adding a new additive loss term going forward means: write a function
matching the shared signature, add one `name: (sign, term_fn)` entry to
`LOSS_TERMS`, and (for the CLI) add a `--<name>_weight` argparse flag in
`update_hdf.py`, default `0.0`. Nothing else needs to change — `loss_fn`,
`train_model`, and `update_hdf.py`'s HDF5-writing code all iterate over
whatever terms/weights are present.

### Three kinds of configurable parameter

Not every tunable in `update_hdf.py` is a loss-term weight, and it's worth
distinguishing the three kinds that exist or are anticipated:

1. **Loss term weights** — tracked by `LOSS_TERMS` as above (`obs`,
   `dist`, `ent`, `flow_l2`, `smooth`). Mechanism: `--<name>_weight`,
   default `0.0`.
2. **Other numeric parameters that aren't loss weights** — e.g.
   `--dist_pow` and `--dont_normalize`, which transform the raw distance
   vector *before* `mask_input` builds the per-week `distance_matrices`
   that `distance_loss` later consumes. These don't go through
   `LOSS_TERMS` or the `weights` dict at all; they shape the *data* a term
   is handed, not how heavily its output counts toward the total loss.
   Mechanism: a plain numeric `argparse` flag, consumed directly in
   `update_hdf.py`'s pre-processing, upstream of `loss_fn`.
3. **Switch-style arguments that change how a term is calculated** — e.g.
   a hypothetical alternative entropy formula. Not yet implemented for any
   term; the anticipated mechanism is a string-selector flag per variable
   term (e.g. `--ent_loss_variant {default,alt}`) dispatching through a
   small `{name: fn}` registry, defaulting to `"default"`.

### `train_model` — the optimization loop

- `model_forward.init(...)` creates the initial (random) parameter dict.
- `optax.adam(...)` is the optimizer (Adam, as in most deep learning —
  think of it as a fancier/adaptive stand-in for plain gradient descent).
- Each step: `jax.value_and_grad(loss_fn, has_aux=True)(params)` computes
  both the loss and its gradient w.r.t. every parameter in the nested dict
  *simultaneously* via autodiff (no hand-derived gradient code anywhere —
  this is the main thing JAX buys you over a hand-written optimizer).
  `optimizer.update` + `optax.apply_updates` then take one Adam step.
- The whole `update` function is wrapped in `jax.jit`, which traces it once
  and compiles it to an XLA kernel (optionally GPU) — this is why the loop
  itself has no vectorization tricks: the heavy lifting is JIT compilation,
  not NumPy-style batching.
- Returns the fitted `params` dict and a `loss_dict` history (`total` plus
  one entry per `LOSS_TERMS` key, built generically from the aux dict
  `loss_fn` returns each step) for diagnostics.

## `update_hdf.py` — the live interface to BirdFlowR

This is the script actually used in the current R↔Python pipeline. It:

1. Copies the input HDF5 (produced by the R preprocessing side) to a new
   output filename encoding the hyperparameters used: the historic
   `obs`/`ent`/`dist`/`pow` fields are always embedded (regardless of
   whether they're zero, nonzero, or at their default — matching
   pre-registry behavior), and `flow_model_training.extra_weight_filename_suffix`
   appends a `_<name><value>` segment for any *other* registered loss
   weight (e.g. `flow_l2`) only when it's nonzero — matching `loss_fn`'s
   own `weights.get(name, 0.0)` convention, where a zero weight means the
   term has no effect on the loss at all. This keeps filenames for runs
   that leave new weights at zero byte-identical to before the registry
   refactor, while ensuring two runs that differ only in a newly added,
   nonzero weight don't collide on the same output filename and silently
   overwrite each other.
2. Reads `distr` (true weekly densities), `distances` (flat pairwise
   distance vector, raised to `dist_pow` and optionally rescaled by
   `1/100^dist_pow`), and `geom/dynamic_mask` (per-week active-cell mask)
   out of the HDF5.
3. Builds a `Datatuple` directly (bypassing `process_data` entirely — this
   is the correct, non-broken construction) and calls `mask_input` to get
   per-week distance matrices and masked densities, plus `cells` (a
   **list** of per-week active cell counts — note this name shadows the
   scalar `cells` field inside `Datatuple`, which holds the *total* raster
   cell count).
4. Trains via `train_model` exactly as above.
5. Writes results back into the **same** HDF5 file: a `marginals` group
   (one dataset per week-to-week flow matrix), updated metadata
   (`birdflow_model_date`), a `metadata/hyperparameters` group, and a
   `metadata/loss_values` group with the full loss history. Both of the
   latter two groups are written by iterating over the `loss_weights` /
   `loss_dict` dicts, so a new `LOSS_TERMS` entry with a corresponding CLI
   flag shows up in the HDF5 output automatically, with no changes to this
   script's writing logic. It also deletes the now-stale `distances`
   dataset. This output structure is what BirdFlowR expects to read back in.

## Stale vs. current paths

`run_model.py` and `birdflow_demo.ipynb` represent an **older** workflow
(single `.tif`/`.npy` raster in, static NaN mask, scalar cell count) that
predates the dynamic-mask change (`64e4a65`, "switched to dynamic masks").
Neither is consistent with the current `flow_model_training.py`:

- Both import `gen_d_matrix` from `flow_model_training.py`, but that
  function was removed in the same commit that introduced `mask_input` and
  no longer exists.
- `run_model.py` calls `process_data`, which has the broken-`Datatuple`
  bug described above.
- The notebook also still uses the deprecated `jax.ops.index` /
  `jax.ops.index_update` API, which was removed from `flow_model.py` itself
  in an earlier commit (`43a7840`) but never updated in the notebook.

In short: **`update_hdf.py` is the maintained, working entry point** for
anything going through the R package; `run_model.py` and the notebook are
historical/aspirational and would need rework (regenerating a per-week
`gen_d_matrix`-equivalent, fixing `Datatuple` arity, updating the JAX index
API) before they'd run against the current code.

## Running on Unity, and testing an unmerged branch there

Production fitting runs inside `BirdFlowContainer` (a separate repo) on
Unity. That container supplies the shared **environment** —
conda/Python, R, and pinned libraries (`jax[cuda12]`, `dm-haiku`,
`optax`, `h5py`, etc.) via `BirdFlowContainer.sif` — the container
does not bake in a copy of this repo's code.

Instead, `BirdFlowPipeline::the$python_repo_path` sets the repo path.
The path is stored in an R environment on package load, so it's user
configurable within an R session, BUT there's no mechanism to change
it from the default within tasks launched on SLURM, so currently
changing it in the RStudio session will NOT change it in launched
tasks.

As of 2026-11-21 my workflow is:

**Configure:**

1. Clone **BirdFlowPy** into `~/BirdFlowPy` and check out desired branch.
2. Clone a copy of **BirdFlowPipeline** into my home directory,
   add and checkout a local branch `override-python-repo`.
   Edit `R/the.R` to set `the$python_repo_path` to `~/BirdFlowPy`.
   Don't push this branch.
3. Rebuild and install **BirdFlowPipeline** (at the default user level).

**Test BirdFlowPy development:**

1. Fetch changes to `~/BirdFlowPy`, checkout desired branch.
2. Test.

**Production:**

1. Checkout main or `v1.1.0`.
2. Fit models.

**Update BirdFlowPipeline**:

1. Fetch (`~/BirdFlowPipeline`)
2. Merge main into `override-python-repo`

All other users are always using the standard shared copy of the repo as configured in the main branch of **BirdFlowPipeline**.

## Tests

`tests/` (pytest) covers `flow_model_training.py` and `update_hdf.py`:

- `tests/test_loss_terms.py` — unit tests for each `LOSS_TERMS` function
  against hand-computed values on a tiny fixture.
- `tests/test_golden_loss_fn.py` — golden-value regression tests: exact
  frozen loss history and fitted parameters for `loss_fn`/`train_model`
  on a small fixed synthetic dataset with a fixed RNG seed, plus checks
  that an unweighted (`0.0`) term is a true no-op and a nonzero-weighted
  term actually changes the loss.
- `tests/test_update_hdf_golden.py` — end-to-end: invokes `update_hdf.py`
  as a subprocess against a synthetic HDF5 fixture
  (`tests/conftest.py::synthetic_hdf5_fixture`) and checks the output
  HDF5's `marginals`/`metadata/loss_values`/`metadata/hyperparameters`
  against frozen expected values. These were captured by diffing a
  pre-refactor run (a `git worktree` at commit `dce4646`, the last commit
  before the `LOSS_TERMS` registry refactor) against a post-refactor run
  on the identical fixture, which confirmed the output is identical
  except for the two new `flow_l2`-related keys the registry now always
  computes.

Adding the `smooth` term (branch `7-temporal-smoothing`) followed the
same playbook: rerunning the golden-value tests with `smooth_weight` at
its default `0.0` reproduced every pre-existing expected value exactly
(`total`/`obs`/`dist`/`ent`/`flow_l2`/fitted-params in
`test_golden_loss_fn.py`; the output filename, `marginals`,
`loss_values`, and `hyperparameters` in `test_update_hdf_golden.py`),
confirming the new term is a true no-op at its default weight; only the
new `smooth` key itself was added to each test's expected-values dict.

Run with `.venv-unity/bin/python -m pytest tests/` — a local venv
pinned to `requirements.txt`, the dependency versions actually
installed in `BirdFlowContainer` on Unity as of 2026-11-17 (see
`dev/known-issues.md`'s issue #5 entry). The golden-value tests'
frozen values are tied to that exact dependency stack (jax's RNG
stream shifts enough between jax/numpy versions to break
`rel=1e-5`-tolerance comparisons), so they must be run under
`.venv-unity`, not an arbitrary local Python install.

See `dev/known-issues.md` for issues identified during development
but deliberately deferred rather than fixed.

## Versioning

`_version.py` holds the single `__version__` string (PEP 440/semver;
no `setup.py`/`pyproject.toml` to attach it to, so it's a standalone
module rather than a package attribute). `CHANGELOG.md` follows [Keep
a Changelog](https://keepachangelog.com/en/1.1.0/); bump `_version.py`
and add a dated entry together when cutting a release.

## Dependencies

`jax`, `haiku` (`dm-haiku`), `optax`, `scipy` (pairwise distances in the
unused `gen_d_matrix`-era code), `h5py` (HDF5 I/O), `numpy`. No
`setup.py`/`pyproject.toml` — this is a loose script collection, not an
installable package. `requirements.txt` pins the versions installed in
the production container (see "Tests" above and
`dev/known-issues.md`).
