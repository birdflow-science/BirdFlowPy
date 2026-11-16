# Work plan

Branch: `pluggable-loss-terms`

## Goal

Make it possible to add new loss-function components (and swap
variants of existing ones) via `update_hdf.py` CLI arguments, without
changing default behavior when those arguments aren't used. Back the
refactor with regression tests that prove the restructuring alone
doesn't change numeric output, so the actual new-term work (not yet
specified) starts from a verified-safe base.

## Environment status (done)

- No local Python/jax/haiku/optax environment existed on this
  machine and the repo has no requirements file. Production fitting
  happens in the `BirdFlowContainer` on Unity, which isn't a
  convenient place to run Claude Code from.
- Created `.venv/` here (gitignored already) for local iteration:
  `numpy<2`, `jax==0.4.18`, `jaxlib==0.4.18`, `dm-haiku==0.0.10`,
  `optax==0.1.7`, `scipy==1.10.1`, `h5py`, `pytest`. These are the
  newest mutually-compatible versions available for Python 3.9
  (Python 3.9 is what's on this Mac; newer haiku/jax/optax require
  3.10+). Verified: `flow_model.py`/`flow_model_training.py` import
  cleanly and a tiny end-to-end `train_model` run executes.
- This venv is for fast local correctness checks only — it is **not**
  a claim that these are the right versions for Unity/production.
  Before merging, the regression tests below should also be run
  inside the actual `BirdFlowContainer` (or you confirm you're fine
  verifying only with the local venv for this change).
- Will add `requirements-dev.txt` pinning the versions above, labeled
  as local-dev-only, so the venv is reproducible without re-deriving
  the pins.

## Step 1 — Regression/characterization tests (write first, against *current* code)

Add a `tests/` directory (pytest). These tests exist to freeze
current behavior before anything is restructured — they're written
and run once against today's code so later steps can prove they
didn't change anything.

- `tests/conftest.py` — fixtures for a small synthetic dataset: fixed
  RNG seed, ~3 weeks, ~4-6 cells per week, fabricated `true_densities`,
  a flat `distances` vector, and per-week `masks` — shaped like what
  `mask_input` (`flow_model_training.py:29`) expects, small enough
  that tests run in well under a second.
- `tests/test_loss_terms.py` — unit tests for `obs_loss`,
  `distance_loss`, `entropy`, `ent_loss` (`flow_model_training.py:44-69`)
  against hand-computed values on a tiny (2-cell, 2-week) example.
  These pin the math itself, independent of how the terms get wired
  together, so they stay valid across the refactor unchanged.
- `tests/test_golden_loss_fn.py` — run today's `loss_fn`
  (`flow_model_training.py:71-81`) on the fixture data with today's
  default weights, and `train_model` (`flow_model_training.py:83-115`)
  for a handful of steps with a fixed `hk.PRNGSequence` seed. Record
  the actual numeric output (total/obs/dist/ent loss values, and the
  full per-step `loss_dict` history) as literal expected values in
  the test. This is the "golden" regression fixture — run once now,
  frozen, and replayed after every later step.
- Run the full suite now and confirm green against current code
  before touching `flow_model_training.py`.

## Step 2 — Low-risk cleanup (behavior-preserving)

- Fix `ent_loss`'s parameter names (`flow_model_training.py:63`): it's
  defined `ent_loss(probs, flows)` but called as
  `ent_loss(flows, pred_densities)` (`flow_model_training.py:79`) — the
  names are swapped from what they actually hold. Rename to match
  usage (e.g. `ent_loss(flows, densities)`); the math and call site
  stay identical, so this is a pure rename.
- Re-run the Step 1 suite — should still be green, confirming the
  rename changed nothing.
- `run_model.py`/`birdflow_demo.ipynb` (already broken independent of
  this work, per `dev/architecture.md`) are out of scope here unless
  you want them touched in this branch — flagging as a decision
  point, not doing anything by default.

## Step 3 — Loss-term registry refactor

Replace the three hardcoded call sites (the weighted sum in `loss_fn`,
the fixed 3-tuple aux return, and `train_model`'s hardcoded
`loss_dict` keys) with a small registry so adding a term means adding
one entry, not editing three functions.

- Give every term function the same signature:
  `term_fn(pred_densities, flows, true_densities, d_matrices) -> scalar`,
  even where a given term ignores most of the arguments (`obs_loss`
  ignores `flows`/`d_matrices`, `distance_loss` ignores
  `pred_densities`/`true_densities`, etc.). Pure signature change, no
  math change.
- Build a registry (ordered dict) mapping term name → `(sign, term_fn)`,
  where `sign` captures today's `-1` on the entropy term — this lets a
  future term be either penalized or rewarded without hardcoding `-1`
  only for entropy. Seed it with today's three terms so default
  behavior is unchanged: `{'obs': (+1, obs_loss), 'dist': (+1,
  distance_loss), 'ent': (-1, ent_loss)}`.
- `loss_fn(params, cells, true_densities, d_matrices, weights)` takes a
  single `weights` dict (`{'obs': ..., 'dist': ..., 'ent': ...}`,
  extendable with new keys defaulting to `0.0`) instead of three
  named kwargs, sums `sign * weight * term_fn(...)` over whatever
  terms have a nonzero or present weight, and returns
  `(total, {name: value for each computed term})` — a dict instead of
  a fixed 3-tuple.
- `train_model` stops unpacking `(obs, dist, ent)` positionally and
  instead builds `loss_dict` from whatever keys are in the aux dict
  (`flow_model_training.py:100-114`), so it doesn't need to change
  again when a term is added.
- `update_hdf.py`'s hyperparameter/loss-value HDF5 groups
  (`update_hdf.py:107-121`) switch from one `create_dataset` call per
  named field to iterating over the weights dict / `loss_dict` keys,
  so new terms show up in the HDF5 output automatically.

## Step 4 — CLI mechanism for alternates (in `update_hdf.py`)

Three distinct kinds of configurable parameter, each with its own
mechanism:

1. **Loss term weights** — tracked by the Step 3 `LOSS_TERMS` registry
   (`obs`, `dist`, `ent`, and new additive terms like `flow_l2`).
   Mechanism: one `--<name>_weight` flag per term, default `0.0`. Zero
   weight is already "off" given the registry, so this preserves
   current default behavior with no branching logic, and matches the
   existing `--obs_weight`/`--dist_weight`/`--ent_weight` convention
   (`update_hdf.py:23-25`) for the R side to construct calls the same
   way it already does.
2. **Other numeric parameters that are not loss-term weights** — e.g.
   `--dist_pow` (`update_hdf.py:26`), which raises the raw distance
   vector to a power *before* `mask_input` builds the per-week
   `distance_matrices` (`update_hdf.py:49-51`), and `--dont_normalize`,
   which rescales that same vector. Neither goes through `LOSS_TERMS`
   or the `weights` dict at all — they shape the *data* a term
   consumes (here, `d_matrices`), not how heavily the term's output is
   weighted in the total loss. `dist_pow` already exists as a working
   example of this category; the mechanism for it is unchanged by
   this plan (a plain `argparse` numeric flag, consumed directly in
   `update_hdf.py`'s pre-processing before `loss_fn` is ever called).
   Any future parameter of this kind (e.g. a transform applied to
   `true_densities` or `masks` before fitting) follows the same
   pattern: a plain flag, applied upstream of `loss_fn`, not a
   registry entry.
3. **Switch-style arguments that change how something is calculated**
   (variants of an existing term, e.g. a different entropy formula,
   not an additional term): a string-selector flag per variable term,
   e.g. `--ent_loss_variant {default,alt}`, dispatching through a
   `{name: fn}` registry, defaulting to `"default"` so existing
   invocations are unaffected. Not yet implemented for any term —
   hypothetical, pending a concrete variant to build.

Not adding a generic `--loss_config` JSON escape hatch for any of the
above — harder to construct/validate from R, and there isn't enough
variety yet in any of the three categories to need that generality.

### Example additive term (smoke test of the mechanism)

Add one concrete new term to prove Step 3's registry and this step's
CLI wiring work end-to-end, not just mechanism scaffolding:

- `flow_l2_loss(pred_densities, flows, true_densities, d_matrices)` —
  `sum(flow ** 2 for flow in flows)`, an L2 penalty on the flow
  matrices themselves (not the raw pre-softmax `z` parameters — this
  keeps it within the uniform `term_fn` signature from Step 3 rather
  than requiring `params` to be threaded through every term). Fits
  the registry as `{'flow_l2': (+1, flow_l2_loss)}`.
- CLI: `--flow_l2_weight`, default `0.0` — off by default, so existing
  invocations are unaffected.
- Covered by its own unit test (hand-computed value on the tiny
  fixture) alongside the Step 1 tests for the existing three terms,
  and included in the Step 5 golden-value re-run (with weight `0.0`,
  confirming it's a no-op by default; a separate test with nonzero
  weight confirms it actually changes the loss).

## Step 5 — Validate the refactor changed nothing

- Re-run the Step 1 golden-value suite against the refactored code
  with default weights — must match the frozen values from Step 1
  exactly (or within tight floating tolerance).
- Add one more regression check: build a tiny synthetic HDF5 fixture
  shaped like what `update_hdf.py` expects (`distr`, `distances`,
  `geom/dynamic_mask`) and run the refactored `update_hdf.py` logic
  against it with default args, confirming the `marginals` and
  `metadata/loss_values` output match a pre-refactor run.

## Step 6 — Docs and worklog

- Update `dev/architecture.md`'s description of the loss function
  (currently describes three hardcoded terms) to describe the
  registry mechanism instead.
- `worklog-entry` once implementation is done.

## Decisions (resolved)

1. **Test execution environment** — local `.venv` is sufficient for
   this branch. A separate verification run inside `BirdFlowContainer`
   on Unity will happen later, after this plan is complete and before
   merging — not part of this branch's own checklist.
2. **`run_model.py`/notebook** — left untouched; out of scope here.
3. **Example new term** — add one trivial new loss term, `flow_l2`
   (an L2 penalty on the flow matrices, `--flow_l2_weight`, default
   `0.0` — see Step 4) as a concrete smoke test that the Step 3/4
   registry + CLI mechanism works end-to-end, in addition to the
   existing three terms.
