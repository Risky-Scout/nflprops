# Driving this build with an implementing agent

## The hand-off pattern

Give the agent exactly four things, and nothing else:

1. `docs/IMPLEMENTATION_SPEC.md`
2. **One** phase document from `docs/phases/`
3. The `contracts/*.yml` files that phase names
4. The existing `src/nflprops/` skeleton

Then this prompt:

> Implement Phase NN as specified in `docs/phases/PHASE_NN_*.md`.
> `docs/IMPLEMENTATION_SPEC.md` is the normative contract; the `contracts/*.yml`
> files are machine-readable and binding.
>
> Fill in only the files that phase lists. Do not touch later phases. Do not add a
> feature, endpoint, prop, or table that is absent from the contracts.
>
> Every acceptance test in that phase document must pass before you report done. If
> something in the spec is wrong, ambiguous, or impossible, **stop and say so** —
> do not work around it silently and do not report a phase complete with a known
> problem in it.
>
> When done, run `make verify-phase-NN` and paste the output.

## Why one phase at a time

Each skeleton module carries a `PHASE:` tag and a `SPEC:` reference. An agent handed
the whole repo will build breadth-first and produce something that looks finished and
is coherent nowhere. Handed one phase, it produces something you can actually verify.

The gates exist to make "done" a binary rather than a vibe:

```bash
make verify-phase-00 ... make verify-phase-10
```

## What the agent may and may not decide

**May:** reorganize internals, add tests, vectorize, improve numerical stability, add
logging, add type annotations, choose data structures.

**May not, without a spec amendment:** change the causal model hierarchy; hardcode
BDL inside model modules; remove point-in-time controls; use random train/test
splits; use season-ending data in historical weeks; model correlated props
independently; skip market snapshot history; drop simulation invariants; assume
undocumented BDL fields; add endpoints absent from the pinned spec.

## Things that will go wrong, and what they mean

| Symptom | Almost certainly |
|---|---|
| Backtest results look extraordinary | Point-in-time leakage. Check `available_at` first, before anything else. |
| An invariant fires | The simulation is producing contradictory props. Fix it; never downgrade the assertion. |
| Model has good means, bad tails | Under-dispersion. Run the PIT histogram (Phase 8) before touching anything else. |
| Every first-TD price looks like value | The NONE state was renormalized away. SPEC §50. |
| A backup RB's projection barely moves when the starter is out | Proportional redistribution. SPEC §37. |
| CLV looks great | Check whether the props you liked were still quoted at close. SPEC §60. |
| `provider verify` fails | Do not edit the contract to match. Find out which side is wrong. |

## The phase-order rule

Strict. A later phase may not begin because an earlier one is "mostly done." Phase 1
in particular: if the provider boundary is wrong, every downstream phase inherits the
error and you will not find out until the backtest.

The one thing that runs out of order: **the market snapshot collector** (part of
Phase 9) should be deployed as soon as Phase 1 lands. Uncollected weeks are gone
forever.
