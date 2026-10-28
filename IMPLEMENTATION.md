# Arbitrary-dataset upgrade: review notes

This is an incremental extension. No existing experiment outputs are regenerated or removed. No GitHub push/deployment is part of this change.

## New Python modules

| File | Purpose |
|---|---|
| `regression/config.py` | Reproducible analysis configuration, typed limits and actionable validation errors. |
| `regression/loading.py` | CSV/XLSX parsing, 10-row preview, missing/duplicate/type/constant/identifier detection. |
| `regression/preprocessing.py` | Fold-fitted numerical/categorical preprocessing and original-to-encoded feature metadata. |
| `regression/engine.py` | Baseline plus five selectors, raw-fold nested CV, bounded k comparisons, fitting, evaluation and dashboard payload. |
| `regression/evaluation.py` | Finite metrics, safe adjusted R², training OLS diagnostics, Shapiro/BP/VIF/QQ and applicable DW. |
| `regression/artifacts.py` | Dataset-independent residual/QQ PNG rendering. |
| `regression/cli.py` | File/target/features/split/seed/CV configuration and explicit new output-directory export. |
| `regression/worker.py` | Isolated subprocess runner used by FastAPI, with temp-only model storage. |
| `api.py` | Upload/schema/analysis/status/predict/export/plot/delete endpoints plus original static dashboard hosting. |

## Modified existing files

| File | What changed and why |
|---|---|
| `feature_selection.py` | Adds optional random seed and strict Lasso scaling/CV mode; old defaults are retained so historical scripts and serialized models remain compatible. |
| `main.py` | Adds a small argument-based dispatch to the reusable CLI. The no-argument historical workflow remains unchanged. |
| `dashboard_export.py` | Versions the added upload script along with existing static assets; does not modify old result files. |
| `dashboard/dist/index.html` | Adds upload/config/preview controls and optional training diagnostic section. |
| `dashboard/dist/app.js` | Runtime dataset registration, arbitrary numeric/category encoding, actual comparison counts, safe user-provided labels, undefined metrics and diagnostic rendering. A small adapter retains the old State snapshot controls. |
| `dashboard/dist/upload.js` | New API connection, file inspection, Y/X/config controls, job polling, runtime result insertion and full JSON export. |
| `dashboard/dist/styles.css` | Extends existing card design for upload controls, settings, schema preview and responsive layouts. |
| `requirements.txt` | Adds openpyxl and local FastAPI runtime dependencies. |
| `requirements-dev.txt` | Separates browser/API test dependencies from the runtime. |
| `README.md` | Local server/CLI commands, API/static hosting distinction, limits, preprocessing, historical compatibility and verification instructions. |

## Automated verification

- `tests/test_engine.py`: schema/CSV/Excel, both built-ins, mixed/20/39-feature tables, missingness, unknown categories, actual pipeline fit indices excluding held-out rows, ID/target copies, limits and singular inference.
- `tests/test_cli.py`: actual main.py CLI dispatch, model/JSON/PNG exports and refusal to overwrite an existing output directory.
- `tests/test_api.py`: real subprocess workflow, file upload, preview, predict with missing/unseen values, exports, diagnostics and cleanup/errors.
- `tests/verify_upload.py`: real browser CSV upload, arbitrary escaped columns/categories, negative values, Python/browser prediction agreement, diagnostic plots, JSON export, static dataset switching and mobile width.
- Existing `verify_dashboard.py` and `verify_housing.py` are retained unmodified and run against the static fallback.

## Review boundaries

The historical no-argument experiment runners remain compatibility entry points, rather than silently replacing previously published datasets with different imputation/CV settings. The reusable engine owns all new arbitrary-dataset analysis; FastAPI and CLI both call it. Historical original files and results remain available for reproducibility.

GitHub Pages has no Python runtime. Local upload is fully usable through FastAPI; a public upload service would require a separate Python deployment and an access/data-retention policy. The backend defaults to loopback and bounded temporary storage. No cloud credentials or uploaded data are checked into the repository.


## Validation performed (2026-10-15, Asia/Taipei)

- 11 engine tests passed, including both built-ins, 20/39 inputs, imputation, unseen categories, actual CV training-row isolation, invalid/limited datasets, singular designs and valid monotonic measurements.
- 3 API tests passed using real isolated subprocess analysis, Excel preview, missing/unseen-value prediction, JSON/PNG export and cleanup/errors.
- 1 CLI integration test passed, including main.py dispatch, exported models/JSON/PNG and refusal to overwrite an existing directory.
- Existing `verify_dashboard.py` and `verify_housing.py` passed without modifying their source or historical data.
- New `verify_upload.py` passed with a real browser upload: escaped arbitrary names/categories/units, raw negative values, exact browser/backend prediction parity, diagnostics, exports, original dataset switching and mobile width.
- JavaScript syntax checks and git whitespace checks passed.
- `git diff --name-only -- results datasets` is empty. No existing experiment outputs or source datasets were changed. No commit/push/deployment was performed.
