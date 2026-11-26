# CovariantRecordHorizons Lean project

This Lake project contains the machine-checked formal verification for the repository.

## Build

The pinned toolchain is specified in `lean-toolchain`, and Mathlib is specified in `lakefile.toml`.

```bash
lake exe cache get
lake build
```

## Sources

- `CovariantRecordHorizons.lean` — library root
- `CovariantRecordHorizons/LFPolytope.lean` — 64-vertex LHV and 96-vertex LF polytope checks
- `CovariantRecordHorizons/LFQuantumBounds.lean` — noncommutative `*`-algebra Sum-of-Squares proofs (`noisy_semiBrukner_sos_identity`, `theorem1_sos_identity`), certificate admissibility and ideal friend-size threshold (`sos_hypothesis_of_le`, `golden_threshold`), coherent Bloch-strategy normalization, and Genuine LF quartic-excess proofs

Build artifacts belong in `.lake/`, which is ignored by Git.