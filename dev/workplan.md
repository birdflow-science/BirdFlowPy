# Work plan

Scratch space for the *currently active* plan only. One plan at a
time — this file is overwritten when new work begins. Once work is
implemented, capture the outcome in `worklog.md` and clear this file
back to this placeholder (see the `workplan-close` skill).

## Issue #7 — temporal smoothness loss term

Goal: add a fifth loss-registry term, `smooth`, that penalizes
week-to-week *curvature* (second difference) in the model's own
fitted weekly marginals, to combat spurious migratory movement caused
by frame-to-frame noise in the input abundance rasters. Follows the
`LOSS_TERMS` pluggable-registry pattern already established by
`flow_l2` on the previous branch.

### Decisions confirmed with the project owner

- **Boundary**: circular — week N's "next" neighbor is week 1 (full
  annual-cycle wraparound), not an open/truncated boundary.
- **Mask semantics**: start by treating any cell outside a week's
  `dynamic_mask` as **true zero** density that week (not
  "unknown") — i.e. use union/zero-embedding, not intersection-only
  masking. Owner flagged that this conflates genuine absence with
  insufficient-data gaps in the source eBird S&T model, and that
  using the smoothing term to help impute those gaps is an
  interesting future direction, but explicitly said to start simple
  with true-zero and revisit later.
- **L1 vs L2**: implement L2 (ridge-like curvature) only, as a single
  concrete term — matching how `flow_l2` was added as one concrete
  term rather than a switchable family. L1/trend-filtering can be a
  separate future term if wanted.

### Key implementation simplification found during research

The issue's AI-discussion draft assumed BirdFlowPy has no existing
"embed a masked per-week vector into the full canonical cell space"
utility, and sketched one from scratch (`active_indices`,
`index_copy`, separate "valid" mask bookkeeping for intersection vs.
union support). That machinery isn't needed here:

1. `flow_model.py`'s `shift_density` already does exactly this
   embedding, using plain JAX boolean-mask scatter:
   `full_cells.at[~nan_mask].set(p)` (`flow_model.py:122`). The same
   `.at[mask].set(values)` pattern works for our per-week embedding.
2. Given the "true zero" mask-semantics decision, there's no separate
   "valid" cell bookkeeping at all — every cell in the canonical
   `total_cells`-length space is a legitimate value (real density or
   true zero) in every week, so the curvature sum just runs over all
   `total_cells` cells for every week, unconditionally.
3. `dtuple.masks` (already threaded through `update_hdf.py` and
   `mask_input`) is already in exactly the shape needed:
   `masks[t]` is a length-`total_cells` boolean vector marking week
   `t`'s active cells in the static/canonical index space — no new
   data structure needs to be built or stored.

This means the new term is a small, self-contained function, not new
infrastructure.

### Design choice revisited and deferred: leave `smooth_loss` unnormalized for now

The issue's sketch normalizes the penalty by the number of curvature
terms summed, so `w_smooth`'s effective strength doesn't depend on
species/mask size. Every existing `LOSS_TERMS` function (`obs_loss`,
`distance_loss`, `ent_loss`, `flow_l2_loss`) uses a raw `jnp.sum`
instead, with no such normalization — and an initial version of this
plan proposed normalizing `smooth_loss` alone, since a substantial
chunk of species get their loss weights *transferred* from a similar
species rather than independently tuned, and active-cell count varies
enormously across species at this project's fixed 100km resolution.

On closer look, the four existing terms don't scale uniformly with
active-cell count `C`, so "normalize `smooth_loss`, leave the rest raw"
isn't clearly the right fix in isolation:

- **`distance_loss`**'s scaling was already addressed previously (per
  the project owner; not re-verified here).
- **`ent_loss`** is bounded by `log(C)` — only a mild, logarithmic
  dependence on cell count, already close to self-normalizing.
- **`obs_loss`** and **`flow_l2_loss`** are both sums of squared
  entries from a density/flow-mass vector that sums to ~1 — the same
  mathematical family `smooth_loss` belongs to (also a sum of squared
  differences on a density-shaped vector). Whether these shrink with
  `C` (if densities stay self-similarly diffuse as range grows) or
  stay roughly constant (if densities concentrate into a fixed number
  of hotspots regardless of range size) isn't derivable from the
  formulas alone — open empirical question.

Since `smooth_loss` is in the same family as `obs_loss` — and
`obs_loss` is the term it most directly competes against in the
objective — normalizing only `smooth_loss` risks decoupling their
relative scaling rather than fixing anything: what actually governs
the fitted model's smoothness-vs-fidelity tradeoff is the
`obs`/`smooth` *ratio* staying stable across species, not either term
individually being cell-count-invariant. Normalizing one but not the
other could make that ratio transfer worse, not better.

**Decision: leave `smooth_loss` unnormalized (raw `jnp.sum`) for this
implementation**, consistent with its three (non-`distance`) siblings,
deferring any normalization until the actual scaling behavior of
`obs_loss`/`ent_loss`/`flow_l2_loss` is known and any fix can be
applied consistently across all of them together rather than
piecemeal. Full writeup, including a note that the project's grid-search
output could be mined to answer the open empirical question, is in
`dev/known-issues.md`'s "`LOSS_TERMS` scale inconsistently with
active-cell count" entry.

### Concrete steps

1. **`flow_model_training.py`**
   - Extend the shared term-function signature from 4 to 5 args:
     `term_fn(pred_densities, flows, true_densities, d_matrices,
     masks) -> scalar` (update the comment at lines 44–46 and add an
     unused `masks` parameter to `obs_loss`, `distance_loss`,
     `ent_loss`, `flow_l2_loss` — mechanical, no behavior change).
   - Add `smooth_loss(pred_densities, flows, true_densities,
     d_matrices, masks)`:
     - Embed every week's `pred_densities[t]` into a zero vector of
       length `total_cells` via `jnp.zeros(total_cells).at[masks[t]].set(pred_densities[t])`.
     - For each week `t`, with `tm1 = (t-1) % T`, `tp1 = (t+1) % T`
       (circular), compute `curvature = embedded[tm1] - 2*embedded[t]
       + embedded[tp1]` and accumulate `jnp.sum(curvature ** 2)`.
     - `total_cells` comes from `masks.shape[1]`; `T` is the number of
       weeks (`len(pred_densities)`).
     - Return the raw accumulated total, unnormalized — per the
       deferred-normalization decision above.
   - Add `'smooth': (1.0, smooth_loss)` to `LOSS_TERMS`.
   - Add `masks` parameter to `loss_fn` (currently `loss_fn(params,
     cells, true_densities, d_matrices, weights)`), threading it into
     each `term_fn(...)` call.

2. **`update_hdf.py`**
   - Add `--smooth_weight` CLI flag, default `0.0` (same pattern as
     `--flow_l2_weight`, line 32).
   - Add `'smooth': args.smooth_weight` to the `loss_weights` dict
     (lines 44–49).
   - Pass `masks=masks` into the `loss_fn` partial (lines 78–82) —
     `masks` is already computed at line 67.
   - No other changes: `extra_weight_filename_suffix`,
     `metadata/hyperparameters`, and `metadata/loss_values` writing
     are already generic over `LOSS_TERMS`/`loss_weights` keys, so
     `smooth` shows up automatically with no code change there.

3. **`run_model.py`** — no change. Already broken/stale per
   `dev/known-issues.md`; out of scope here.

4. **Tests**
   - `tests/conftest.py`: add a small fixture exercising circular
     wraparound and zero-embedding with a non-trivial `masks` array
     (e.g. 3–4 weeks, a cell that's active in some weeks and not
     others) for `smooth_loss`'s unit tests. Extend
     `small_training_fixture` to also expose `dtuple.masks` for the
     golden-value tests.
   - `tests/test_loss_terms.py`: add `masks=None`/a 5th arg to all
     existing hand-calculated calls; add new hand-calculated unit
     test(s) for `smooth_loss` (including the circular-wrap and
     zero-embedding behavior).
   - `tests/test_golden_loss_fn.py`: thread `masks` through `_run`;
     regenerate frozen expected values to include the new `'smooth'`
     key (same playbook used when `flow_l2` was added); add a
     `smooth_weight=0.0`-is-a-noop test and a
     nonzero-weight-changes-loss test, mirroring the existing
     `flow_l2` pair.
   - `tests/test_update_hdf_golden.py`: regenerate expected
     `metadata/loss_values` (new `smooth` key); add a
     `--smooth_weight`-nonzero filename-collision test mirroring
     `test_update_hdf_nonzero_flow_l2_weight_gets_distinct_filename`.
   - `tests/test_filename_suffix.py`: no change required (mechanism
     is already generic/covered), but can add `smooth` to
     `ZERO_WEIGHTS` as a completeness check if desired.
   - Regenerate golden values under `.venv-unity`, confirming (as was
     done for `flow_l2`) that the only diffs from a pre-change run
     are the new `smooth`-related keys.

5. **Docs**
   - `dev/architecture.md`: extend the "Loss components: a pluggable
     registry" section with `smooth`/`smooth_loss` (circular
     second-difference penalty, zero-embedding into the canonical
     `total_cells` space via each week's `dynamic_mask`, reusing
     `shift_density`'s existing `.at[].set()` embedding pattern).
   - `CHANGELOG.md`: add an `[Unreleased]` bullet for the new
     `smooth` term + `--smooth_weight` flag, following the existing
     `flow_l2` entry's style.
   - `dev/known-issues.md`: add a short forward-looking note
     capturing the owner's true-zero-vs-insufficient-data mask
     ambiguity and the imputation idea, as a deliberately deferred
     consideration (not a bug, not blocking this work).

### Open points to confirm before/while implementing

- Function name: `smooth_loss` (matches the `'smooth'` registry key,
  consistent with `obs_loss`/`ent_loss`/`flow_l2_loss` naming).
- Unnormalized raw sum, as decided above (deferred, not abandoned —
  see `dev/known-issues.md`). Revisiting this later, if the grid-search
  mining described there happens, would mean updating `smooth_loss`
  alongside `obs_loss`/`ent_loss`/`flow_l2_loss` together, not in
  isolation.

### Correction after implementation: boundary is open, not circular

The "circular" boundary decision above (and its implementation via
`% weeks` modulo indexing) was wrong and has been corrected to an
**open** boundary (no curvature term at week 0 or the last week, per
`range(1, weeks - 1)`), with circular dropped entirely rather than kept
as an unused option.

Reason: `BirdFlowR::preprocess_species()` already linearizes the
annual cycle for full-year models by duplicating week 1's target
distribution as an extra final week before this code ever sees the
data — confirmed by the project owner after the circular version was
already implemented and tested. So a "full year" model is, from
`flow_model_training.py`'s side, just a longer linear chain, not a
true ring; the non-full-year subset-of-weeks case (rarely used, per
the owner) is likewise just a shorter linear span. The circular
version fabricated two spurious curvature terms with no real movement
behind them (week 0 vs. the duplicated final week, and vice versa) —
exactly the kind of artifact `smooth_loss` exists to suppress, not
introduce.

All code/test/doc references to circular wraparound have been updated
to open-boundary; golden values in `tests/test_golden_loss_fn.py` and
`tests/test_update_hdf_golden.py` were regenerated (only the `smooth`
key's values changed — every other term, the fitted params, and the
output filename remained byte-identical, confirming the boundary fix
didn't touch anything else).
