# BirdFlowPy architecture

This is a from-scratch summary of how the four top-level scripts fit together,
written for someone who knows the BirdFlowR model/data structures but not JAX
or Haiku. It reflects the state of the code as of commit `de0d77d` (2026-11-16).

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

### Loss components

- `obs_loss` — sum of squared residuals between predicted and true weekly
  densities (fit term).
- `distance_loss` — `sum(flow * distance_matrix)` per week, i.e. a penalty
  proportional to how much mass moves how far, where "far" is whatever
  distance/cost matrix was supplied (raised to `dist_pow` and optionally
  normalized before this function ever sees it — that happens in the
  calling script).
- `entropy` / `ent_loss` — joint entropy of the flow matrices minus the
  entropy of the marginal distributions; this is the same
  "marginal-minus-joint" entropy regularizer that shows up in BirdFlowR's
  math for discouraging degenerate (too-concentrated or too-diffuse)
  transition distributions.
- `loss_fn` — runs the model forward (`model_forward.apply`), derives
  predicted weekly densities from the flow matrices, and combines the three
  terms as `obs_weight*obs + dist_weight*dist - ent_weight*ent`. This is the
  scalar that gets differentiated.

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
- Returns the fitted `params` dict and a `loss_dict` history (total/obs/dist/ent
  per step) for diagnostics.

## `update_hdf.py` — the live interface to BirdFlowR

This is the script actually used in the current R↔Python pipeline. It:

1. Copies the input HDF5 (produced by the R preprocessing side) to a new
   output filename encoding the hyperparameters used.
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
   `metadata/loss_values` group with the full loss history. It also deletes
   the now-stale `distances` dataset. This output structure is what
   BirdFlowR expects to read back in.

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

## Dependencies

`jax`, `haiku` (`dm-haiku`), `optax`, `scipy` (pairwise distances in the
unused `gen_d_matrix`-era code), `h5py` (HDF5 I/O), `numpy`. No
`setup.py`/`pyproject.toml` — this is a loose script collection, not an
installable package.
