# nflprops 2026.1.1 hotfix

This hotfix corrects two issues exposed by the first live BALLDONTLIE run:

1. The previous local checkout could still contain the older CLI where `ingest bootstrap`
   and `coverage` raised `NotImplementedError`. This package contains the executable
   lean ingestion/coverage pipeline.
2. OpenAPI verification now treats the provider contract as an intentional used subset:
   newly-added BDL fantasy endpoints are allowed only when explicitly recorded in the
   contract ignore-list with a reason. Parameter comparison normalizes `array[T]` and
   `string(date)` notation, and the standings schema is corrected to `NFLStandings`.

The model version remains 2026.1.0 because no fitted model artifact changed.
The package/code version is 2026.1.1.
