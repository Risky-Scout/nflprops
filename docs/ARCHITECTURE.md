# Architecture at a glance

## Dependency direction (one-way, CI-enforced)

```
                 BDL JSON
                    │
          providers/bdl/raw_models.py      permissive
                    │
          providers/bdl/mapper.py          quirks end here
                    │
              domain/models.py             STRICT, canonical, provider-free
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 data/          features/        market/
 warehouse      asof.py          odds, devig, snapshots
    │               │               │
    │            state/             │
    │          empirical_bayes      │
    │               │               │
    └──────────► models/ ◄──────────┘
                    │
              simulation/                  one game, one simulation
                    │
              simulation/props.py          every prop from the same draws
                    │
             calibration/                  OOF + dispersion
                    │
              backtest/                    walk-forward, leakage-tested
                    │
               explain/                    ablation with common random numbers
```

Forbidden edges, checked by `tests/unit/test_import_boundaries.py`:

- `simulation → providers.*`
- `models → providers.*`
- `features → providers.*`
- `state → providers.*`
- any HTTP library imported outside `providers/`
- any `nfl/v1` string outside `providers/bdl/endpoints.py`

## Layer responsibilities

| Layer | Owns | Never does |
|---|---|---|
| `providers/bdl/client.py` | auth, HTTP, retry, pagination, param encoding | compute football features |
| `providers/bdl/quirks.py` | every provider oddity | leak a quirk downstream |
| `providers/bdl/mapper.py` | raw → canonical, `available_at` assignment | model anything |
| `domain/` | canonical types and protocols | know a provider exists |
| `data/` | immutable raw, parquet, identity, quality | interpret football |
| `features/asof.py` | the single point-in-time filter | be bypassed |
| `state/` | EB posteriors **and variances** | be fitted on future data |
| `models/` | component structural models | model QB passing yards (derived) |
| `simulation/` | one coherent game, invariants, props | fit anything |
| `market/` | odds, devig, snapshots, CLV | leak a prop price into `p_fundamental` |
| `calibration/` | OOF calibration, dispersion | fit in-sample |
| `backtest/` | walk-forward, gates | random-split |

## The three files a reviewer should read first

1. `docs/IMPLEMENTATION_SPEC.md` Part I — the non-negotiable rules
2. `contracts/invariants.yml` — what the simulation guarantees
3. `src/nflprops/simulation/invariants.py` — how it is enforced
