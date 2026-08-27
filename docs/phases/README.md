# Phase Documents

Hand an implementing agent **exactly one** of these at a time, together with:

- `docs/IMPLEMENTATION_SPEC.md`
- the relevant files under `contracts/`
- the existing `src/nflprops/` skeleton

Each phase document has the same shape:

1. **Objective** — one paragraph
2. **Spec sections** — which parts of the spec are normative here
3. **Files to implement** — exact paths, already stubbed in the skeleton
4. **Contracts consumed** — machine-readable files that must be obeyed
5. **Acceptance tests** — must pass before the phase is done
6. **Definition of done** — checklist
7. **Explicitly out of scope** — what the agent must NOT touch

Build order is strict. A later phase may not be started because an earlier one is
"mostly done."
