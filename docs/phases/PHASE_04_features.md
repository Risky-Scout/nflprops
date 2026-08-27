# Phase 4 — Point-in-time features

## Objective
Build every feature in `contracts/feature_registry.yml`, each with a correct
`available_at` rule, an explicit null policy, and a paired `__is_missing` column.

## Spec sections
§2, §23, §24, and the whole feature registry.

## Files to implement
```
src/nflprops/features/asof.py        # THE point-in-time filter; everything routes through it
src/nflprops/features/registry.py    # loads + validates contracts/feature_registry.yml
src/nflprops/features/game.py
src/nflprops/features/team.py
src/nflprops/features/opponent.py
src/nflprops/features/player.py
src/nflprops/features/roles.py
src/nflprops/features/market.py
```

## Requirements
1. `asof.py` provides the single filter used by every feature builder. No feature
   module queries the warehouse without going through it. Enforced by test.
2. Every feature materializes as `<name>` and `<name>__is_missing`.
   **Blanket zero-filling is prohibited.**
3. Opponent features are built by **reversing team-game rows within a game**. Do not
   source them from `/nfl/v1/team_season_stats` — that schema has no opponent fields.
4. A feature not present in the registry cannot be written. A registry entry without
   an implementation is a build failure. Both directions are tested.
5. Market features carry the `collector_received_at` of the quote they used, so
   leakage tests can verify them.

## Acceptance tests
```
tests/leakage/test_feature_registry_complete.py    # columns <-> registry, both ways
tests/leakage/test_no_future_information.py
tests/leakage/test_no_result_from_predicted_game.py
tests/leakage/test_season_aggregate_not_used_pit.py
tests/unit/test_is_missing_columns_present.py
tests/unit/test_opponent_from_reversed_rows.py
```

## Definition of done
- [ ] `nflprops features build --as-of <ts>` produces `game_features` and
      `player_features` for a historical week
- [ ] Every leakage test green
- [ ] Rebuilding the same `as_of` twice produces byte-identical parquet

## Explicitly out of scope
Empirical-Bayes state fitting (Phase 5). Models.
