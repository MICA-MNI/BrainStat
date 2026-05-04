# Changelog

All notable changes to BrainStat are recorded here. The project adopts
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-05

This release re-greens CI after a long stretch of upstream-driven breakage,
fixes a cluster of high-severity correctness bugs in the statistics core,
adds the FreeSurfer Destrieux atlas to `fetch_parcellation`, hardens the
Neurosynth fetcher, and adopts a literature-aligned default for BigBrain
histology profiles. It also closes the longest-standing maintenance items
in the issue tracker.

### Backwards-incompatible changes

- **`brainstat.context.histology.read_histology_profile` now inverts BigBrain
  intensities by default** (`invert=True`) so that higher = darker = more
  cell-dense, matching Paquola et al. 2021 (eLife) and most downstream
  BigBrain analyses. Pass `invert=False` to recover the previous behaviour.
  MPC and gradient computations are correlation-based and therefore
  invariant to this flip; only direct inspection of profile intensities is
  affected. (#274)
- **`SLM.__init__` now coerces `contrast` to numeric and `mask` to
  boolean.** Pandas Series of object/categorical dtype now raise a clear
  `TypeError` instead of producing the cryptic `can't multiply sequence by
  non-int of type 'float'` deep inside `_t_test`. Integer 0/1 masks (the
  natural shape of `nib.load(...).get_fdata().astype(int)`) are now
  converted to bool rather than degenerating into fancy indexing inside
  `_fdr`. (#342)
- **Pin numpy<2 and pandas<2.1.** `abagen 0.1.3` still imports
  `pkg_resources` and calls `DataFrame.groupby(axis=...)`; until upstream is
  patched, BrainStat must stay on the numpy 1.x ABI to avoid runtime
  binary-incompatibility errors. Will be lifted once abagen ships a
  compatible release.

### Bug fixes — statistics core

- `SLM._surfstat_to_brainstat_rft`: fix off-by-one in Yeo7 indexing.
  `peak_clus` returns 1-based vertex IDs (a SurfStat carry-over); these
  were being used to index a 0-based numpy array, raising `IndexError`
  when a peak landed on the final mesh vertex and silently mis-labelling
  Yeo networks for every other peak. (#347)
- `SLM.py`: `from cmath import sqrt` was importing the **complex** square
  root; switched to `math.sqrt`. The expression was always real but the
  surrounding code would have silently propagated complex values had the
  branch ever been negative.
- `SLM.py`: removed a duplicate `fetch_template_surface` import.
- `_t_test.py`: `sys.exit("Contrast is not estimable :-(")` would kill the
  host Python process when invoked from a notebook or pipeline. Replaced
  with `raise ValueError(...)`.
- `_t_test.py`: replace `float(...)` with `.item()` on the t-statistic
  computation. `numpy>=2.0` raises `TypeError: only 0-dimensional arrays
  can be converted to Python scalars` for `float()` on size-1 ndarrays.

### New features

- `brainstat.datasets.fetch_parcellation(template, "destrieux", ...)`
  now returns the FreeSurfer `aparc.a2009s` (Destrieux 2009)
  parcellation, sourced via
  `nilearn.datasets.fetch_atlas_surf_destrieux`. Native `fsaverage5`;
  other `fsaverage*` templates resampled with nearest-neighbour
  interpolation. `fslr32k` is rejected (FreeSurfer-specific). (#343)
- The `surf=None + correction='rft'` error message now points users at
  `correction='fdr'` for volumetric data instead of just refusing.

### Maintenance

- **CI fully restored.** Both Python and MATLAB suites had been red on
  `master` for a long stretch. Root causes addressed:
  - `setuptools >= 81` no longer ships `pkg_resources`, breaking
    `abagen` import → pinned `setuptools<81` in CI and reinstalled it
    after the editable install.
  - `numpy.dtype size changed` from mixing pinned `pandas<2.1` (built
    against numpy 1.x ABI) with `numpy>=2` → pinned `numpy<2` to keep
    the stack consistent.
  - `pandas 2.0.x` has no Python 3.12 wheels → temporarily dropped 3.12
    from the matrix until abagen is patched and the pin can be lifted.
  - `MATLAB:Python:PythonUnavailable` from the precomputed-pickle tests
    → bumped to `matlab-actions/setup-matlab@v2` +
    `matlab-actions/run-command@v2`, configure `pyenv()` before
    `runtests`, add explicit `addpath(genpath('brainstat_matlab'))`.
  - GitHub Actions versions: `actions/checkout@v4`,
    `actions/setup-python@v5`. (PR #379)
- **Drop `# type: ignore` from `SLM.py` and `_multiple_comparisons.py`.**
  The class-body imports that were tripping
  [python/mypy#10521](https://github.com/python/mypy/issues/10521) have
  been lifted to module level and bound as class attributes (`_linear_model
  = _linear_model`, etc.). mypy now runs to completion on both files
  instead of crashing. The pre-existing untyped-code errors that the
  blanket ignore was hiding are surfaced but tracked separately; the
  `mypy` CI job remains commented out pending that triage. (#199)
- **MATLAB R2019b/R2020a compatibility.** `peak_clus.column_vector`
  replaced the `mustBeVector` validator (introduced in R2020b) with an
  explicit `isvector` check. (#285)
- **Resilient Neurosynth download.** `meta_analytic_decoder/fetch_neurosynth_data`
  validates the downloaded zip via `java.util.zip.ZipFile` before
  `unzip`, retries up to 3 times, cleans up partial zips between
  attempts, and verifies the post-extract file count. Truncated
  downloads no longer leave the toolbox in a permanently broken state.
  (#341)

### Issues closed

- #199 — Remove `# type: ignore` from `SLM` class
- #274 — Flip the bits on the histological profiles
- #285 — `column_vector` uses `mustBeVector` from MATLAB 2020b+
- #341 — Neurosynth data failed to get fetched
- #342 — Volumetric mixed effects (input contract bugs; cluster-level
  RFT/TFCE on volumes remains tracked separately as a feature request)
- #343 — Atlas coverage for FreeSurfer GLM workflows (Destrieux added;
  hosted Desikan-Killiany remains tracked for a follow-up)
- #347 — Retrieval of yeo7 information has off-by-1 error

### Known follow-ups

- **Lift the numpy/pandas/3.12 pins** once `abagen` drops `pkg_resources`
  and the `groupby(axis=)` call.
- **Re-enable the commented-out `mypy` CI job** once the ~380 latent
  typing errors uncovered by removing `# type: ignore` are triaged.
- **Volumetric cluster-level RFT/TFCE** for `SLM` — present in SurfStat,
  not yet ported.
- **Hosted Desikan-Killiany (`aparc`) atlas** in `fetch_parcellation`
  — needs a stable mirror (OSF / Zenodo) for the `.annot` files.

## [0.5.2] — 2026-01

Hotfixes only; no formal changelog was kept for this release.

## [0.5.1] and earlier

See `git log` and the merged-PR history for releases prior to the
introduction of this changelog.
