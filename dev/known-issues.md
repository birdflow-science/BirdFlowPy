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
cwd (as documented: `.venv/bin/python -m pytest tests/` from the repo
root). Running pytest from any other directory — a subdirectory, a
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
