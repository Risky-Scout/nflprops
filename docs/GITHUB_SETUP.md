# GitHub setup

This repository should be **private**. The model is intended to contain proprietary
forecasting logic, market-history data, and fitted artifacts.

## First push

1. Install GitHub CLI and authenticate:
   `gh auth login`
2. From the repository root, pin and verify the official BDL NFL specification:
   `make pin-bdl-spec`
   `python tools/sync_runtime_resources.py`
   `make verify-bdl-spec`
3. Generate the dependency lock on a networked machine:
   `uv lock`
4. Run:
   `python tools/sync_runtime_resources.py --check`
   `make test`
5. Commit:
   `git init -b main`
   `git add .`
   `git commit -m "Initial nflprops 2026.1.0"`
6. Create and push the private repository:
   `gh repo create nflprops --private --source=. --remote=origin --push`

Never commit `.env`, BDL credentials, raw market data, canonical Parquet data,
DuckDB files, predictions, or fitted model artifacts unless deliberately encrypted.
