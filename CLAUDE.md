# CLAUDE.md

## Experiment code style
- All experiment parameter defaults must be declared in `params_def` in the Experiment `__init__`, not buried as `.get()` fallbacks elsewhere.
- Program classes and `acquire`/`analyze`/`display` methods should use direct attribute access on `cfg.expt` (e.g. `cfg.expt.length`), never `cfg.expt.get("key", default)`. By the time the Program runs, every key it needs must already exist.
- `.get()` on `params` dict (before merge into `cfg.expt`) is fine for pre-init logic. `.get()` on `data` dicts (checking optional analysis results) is also fine.

## Plotting
- Do not add titles to plots. Existing titles in user code should be left as-is.
