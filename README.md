# Record Leakage in Local Friendliness Tests: Randomness Trade-off, Two-Friend Bounds, and Noise Budgets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24435914.svg)](https://doi.org/10.5281/zenodo.24435914)
[![arXiv](https://img.shields.io/badge/arXiv-TODO.XXXXX%20%5Bquant--ph%5D-b31b1b.svg)](https://arxiv.org/abs/TODO.XXXXX)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Lean 4](https://img.shields.io/badge/Verified%20in-Lean%204-5A67D8.svg)](https://lean-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion numerical simulation and formal machine-checked verification repository for the manuscript:

> **Record Leakage in Local Friendliness Tests: Randomness Trade-off, Two-Friend Bounds, and Noise Budgets**  
> *(Versions up to v1.0.3 were titled "Covariant Record Horizons and the Abandonment of Absolute Events"; the DOI is unchanged.)*  
> **Author:** Kevin J. Baker ([ORCID: 0009-0002-4464-6391](https://orcid.org/0009-0002-4464-6391))  
> **Preprint:** [doi:10.5281/zenodo.24435914](https://doi.org/10.5281/zenodo.24435914) · `arXiv:TODO.XXXXX [quant-ph]` *(link active upon public timestamp)*

---

## Overview

Extended Wigner's friend scenarios (EWFS) and Local Friendliness (LF) no-go theorems demonstrate that unitary quantum mechanics, Local Agency, and the *Absoluteness of Observed Events* (AOE) cannot simultaneously hold at arbitrary scales. However, standard no-go theorems leave two operational questions unresolved:
1. **Covariant Actuality Rule:** When does a physical proposition acquire a definite truth value relative to a localized relativistic observer without invoking foliation-dependent collapse across spacelike hypersurfaces?
2. **Quantitative Decoherence & Leakage Boundaries:** How does imperfect isolation—where partial record distinguishability leaks beyond a superobserver's reversible control—degrade violations across both Bell/CHSH facets and *genuine* multi-friend Local Friendliness polytope facets?

The manuscript introduces the **Causal Actuality Hypothesis (CAH)**, indexing truth valuations $v_O(P, \tau) \in \{T, F, U\}$ to local record algebras $\mathcal{R}_O(\tau) \subset \mathfrak{A}(J^-(x_O(\tau)))$ inside each observer's causal past light cone, bounded fundamentally by horizon soft-quantum decoherence (Danielson–Satishchandran–Wald) and operationally by environmental branch overlap $f = \vert{}\langle e_0 \vert{} e_1 \rangle\vert{} \in [0, 1]$ (with record leakage $\ell = \sqrt{1 - f^2} \in [0, 1]$).

This repository provides **independent, reproducible verification** of the manuscript's discrete polytope geometry, exact analytical operator bounds, asymptotic scaling laws, and quantum-hardware visibility thresholds using **Python** and the **Lean 4 theorem prover**.

---

## Manuscript-to-Code Mapping

| Manuscript Result | Verification Artifact | Method |
|---|---|---|
| $(3,3,2,2)$ LHV (64 vertices) vs. Two-Friend LF (96 vertices) polytope bounds across Bong *et al.* (2020) Table I | `lean/CovariantRecordHorizons/LFPolytope.lean` & `lf_leak_tests.py` | Exhaustive kernel proof (`by decide`) + exact $\mathbb{Q}$/$\mathbb{R}^{15}$ matrix rank |
| Proof that `Genuine LF` ($\le 6$) is NOT a Bell facet (LHV rank $12 < 15$) but IS a true LF facet (LF rank $15$) | `lean/CovariantRecordHorizons/LFPolytope.lean` (`genuineLF_is_strict_LF_facet`) | Fraction-free integer Gaussian elimination inside the Lean 4 kernel |
| Strict violation of `Genuine LF` for all $f \in (0, 1]$ with quartic excess $S_{\mathrm{GLF}}(f) \ge 6 + \frac{f^4}{1 + f^2}$ | `lean/CovariantRecordHorizons/LFQuantumBounds.lean` (`genuineLF_quartic_excess`) | Formal real-algebra proof in Lean 4 + `Mathlib` (zero `sorry`) |
| $\mathcal{O}(f^2)$ (One-Friend) vs. $\mathcal{O}(f^4)$ (Two-Friend) suppression & Werner visibility thresholds $f_{\mathrm{crit}}(\mu)$ | `lf_leak_tests.py` | Bidirectional homotopy `L-BFGS-B` sweeps + `brentq` root-finding |
| Exact One-Friend Semi-Brukner bound $B_{\max}^{\mathrm{SB}}(f) = 2\sqrt{1 + f^2} = 2\sqrt{2 - \ell^2}$ and generalized noisy SOS identity | `lean/CovariantRecordHorizons/LFQuantumBounds.lean` (`noisy_semiBrukner_sos_identity`, `theorem1_sos_identity`) & `lf_leak_tests.py` | Machine-checked noncommutative $*$-algebra Sum-of-Squares proof in Lean 4 + `Mathlib` & multi-start Bloch/Schmidt optimization ($< 10^{-15}$ error) |
| Scope of the two-friend results: Brukner bound in every dimension (Jordan's lemma), failure of the quartic law for the other functionals under coarse-grained friend measurements, constant-outcome reduction of $I_{3322}$ | `lf_dimension_checks.py` | Seesaw search (explicit strategies, i.e. lower bounds) in local dimension 2, 3 and 4 |
| Two-qubit product bound $S \le 2\mu\sqrt{1 + f_A^2 f_B^2}$ for the CHSH-type functionals (Proposition 2) and the two-lab threshold $f_{\mathrm{crit}} = (\mu^{-2} - 1)^{1/4}$ | `lf_leak_tests.py` (section 5 of the console report) | Multi-start optimization at nine $(f_A, f_B)$ pairs: no value above the bound, `Bell non-LF` and two-friend `Semi-Brukner` on it to $10^{-12}$; closed form vs. `brentq` thresholds |
| Noise budget for circuit implementations: setting-dependent visibilities, exact relaxed LF bounds under readout error (linear program), one-friend closed form $2\mu\eta_0\sqrt{\eta_1^2 + \eta_2^2 f^2}$, critical gate and readout errors for the one-friend, Brukner and Genuine LF tests, and Willow-class reference point | `lf_noise_budget.py` | Density-matrix simulation of the circuit vs. visibility factors ($< 10^{-12}$), sum-of-squares identity on random matrices, linear program with dual certificates for the relaxed bounds, seesaw search (explicit strategies) with bisection for the thresholds |
| Friend-size scaling in entangling gates ($g_{\mathrm{rec}}, g_{\mathrm{rev}}$), ideal visibility threshold $v^2 > (\sqrt{5}-1)/2$, and maximum fan-out friend size $G$ | `lf_noise_budget.py` (`max_friend_size`, `behaviour_fanout_one_friend`) | Exact multi-qubit fan-out density-matrix simulation for $G \le 3$ vs. Eq. (`eq:eta_gates`) ($< 10^{-15}$) and closed-form threshold evaluation |

*(Note: Relativistic horizon soft-graviton/soft-photon ceilings following Danielson, Satishchandran, and Wald are treated analytically in the manuscript text and are not simulated here.)*

---

## Repository Structure

```text
.
├── lf_leak_tests.py                         # Numerical verifier and optimization suite
├── lf_dimension_checks.py                   # Seesaw checks beyond two qubits and rank-one measurements
├── lf_noise_budget.py                       # Circuit-level noise budget and device lookup
├── requirements.txt                         # NumPy, SciPy, and Matplotlib dependencies
├── .github/workflows/
│   ├── lean_action_ci.yml                   # Lean and Mathlib verification
│   ├── manuscript_build.yml                 # Figure generation and PDF build
│   └── release.yml                          # Tagged GitHub Release, arXiv & Zenodo publisher
├── scripts/
│   └── publish_zenodo.py                    # Zenodo Deposit REST API release publisher
├── lean/
│   ├── lakefile.toml                        # Lake package and Mathlib dependency
│   ├── lake-manifest.json                   # Resolved Lake dependency revisions
│   ├── lean-toolchain                       # Pinned Lean 4 toolchain
│   ├── README.md                            # Lean package build instructions
│   ├── CovariantRecordHorizons.lean         # Lean library root
│   └── CovariantRecordHorizons/
│       ├── LFPolytope.lean                 # Kernel-checked polytope/rank proofs
│       └── LFQuantumBounds.lean            # Mathlib *-algebra SOS & real-algebra proofs
├── manuscript/
│   ├── main.tex                             # RevTeX manuscript driver
│   ├── Makefile                             # Clean rebuild, audit, and incremental targets
│   ├── build.sh                             # Cross-platform convenience wrapper
│   ├── README.md                            # Manuscript build documentation
│   ├── sections/                            # Individual section .tex sources
│   ├── bibliography/references.bib          # BibTeX reference database
│   ├── figures/                             # Generated PDF/PNG publication figures
│   └── data/                                # Generated JSON verification & noise-budget summaries
├── CITATION.cff                             # GitHub & Zenodo citation metadata (CFF 1.2.0)
├── README.md
└── LICENSE                                  # Repository license
```

---

## Quick Start & Reproduction

### 1. Python Numerical Suite (`lf_leak_tests.py`)

Requires Python 3.10+. The numerical and figure dependencies are declared in `requirements.txt` (`numpy`, `scipy`, and `matplotlib`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Run full console verification (polytope ranks, analytical error checks, leakage tables, and hardware noise thresholds):**
```bash
python3 lf_leak_tests.py
```
The default sweep samples $f$ at 51 points, which resolves the `Genuine LF` branch switch near $f \approx 0.85$; the whole run takes on the order of five minutes on a single core. `--grid-points 11` restores the coarse 0.1-spaced sweep (about a quarter less time), and `--grid-points 101` gives a smoother figure. The tabulated 0.1-spaced rows are the same in every case.

**Generate publication PDF/PNG figures and export JSON data:**
```bash
python3 lf_leak_tests.py --plot --save-json lf_verification_results.json
```
This produces `lf_leakage_bounds.pdf` and `lf_leakage_bounds.png`, illustrating:
* **Panel (a):** Quantum excess $\Delta S(f) = S_{\max}(f) - B$ over the tabulated bound $B$ across coherence retention $f \in [0, 1]$ for the five functionals `Semi-Brukner`, `Brukner`, `Bell non-LF`, `I3322-LF`, and `Genuine LF`. For `Bell non-LF`, $B = 2$ is the Bell bound; its LF bound is 4, so that curve shows Bell nonlocality, not an LF violation.
* **Panel (b):** Log-log asymptotic scaling in the high-leakage limit ($f \to 0, \ell \to 1$), demonstrating the sharp split between **quadratic** One-Friend excess $\Delta S \simeq f^2$ and **quartic** Two-Friend suppression $\Delta S = \mathcal{O}(f^4)$ (closely tracking the collinear lower bound $\Delta S_{2\mathrm{F}}^{\mathrm{col}} = f^4 / (1 + f^2)$ as $f \to 0$).

**Restrict state optimization to maximally entangled Bell states ($|\Phi^+\rangle$):**
```bash
python3 lf_leak_tests.py --maximally-entangled-only
```

**Survey the two LF facet classes the manuscript does not analyze** (the second genuine LF class and the $I_{3322}$ class with marginals on inputs 2 and 3; adds about 20 seconds):
```bash
python3 lf_leak_tests.py --extended-classes
```
This prints their facet checks and the excess at a few values of $f$ for rank-one projective measurements. These are lower bounds, not maxima: a constant-outcome observable can do better (see the next command).

**Check the two-qubit, rank-one restrictions** (seesaw search over measurements of any rank in local dimension 2, 3 and 4; about a minute):
```bash
python3 lf_dimension_checks.py
```
Every printed number is the value of an explicit strategy. The table shows that the Brukner optimum does not grow with dimension, that a coarse-grained friend measurement on a larger system lets the other two-friend functionals escape the quartic law, and that for $I_{3322}$ a constant-outcome observable beats the rank-one optimum below $f \approx 0.83$ (`lf_leak_tests.py` takes that reduction into account for the figure).

**Noise budget for a circuit implementation** (two-qubit gate error, readout error and record leakage together; about three minutes with the figure):
```bash
python3 lf_noise_budget.py --plot --save-json lf_noise_budget.json
```
The model is one friend qubit per laboratory, a CNOT for the friend's measurement and for its reversal, two-qubit depolarizing noise after every entangling gate, dephasing of the record with coherence retention $f$, and a symmetric readout error $\varepsilon$ that both lowers the visibilities and loosens the LF bounds. The script first checks the model against a density-matrix simulation of the circuit, then prints the largest gate error at which each of the three tests (`Semi-Brukner` with one friend, `Brukner` and `Genuine LF` with two) is still violated, and writes `lf_noise_budget.pdf`/`.png`. One-friend numbers are exact; two-friend numbers compare explicit strategies with exact relaxed bounds, so the true tolerances can only be larger.

**Look up one device:**
```bash
python3 lf_noise_budget.py --r2q 0.004 --eps 0.01 --f 0.98
python3 lf_noise_budget.py --r2q 0.003 --eps 0.01 --friend-gates 10 --record-gates 1
```
`--r2q` is the average two-qubit gate infidelity, `--eps` the readout assignment error, `--f` the coherence retention of the record (for a record held for a time $t$, $f = e^{-t/T_{\varphi,S} - t/T_{\varphi,F}}$ with the pure-dephasing times of the system and friend qubits), and `--friend-gates G` (with optional `--record-gates k`) sets the entangling-gate counts $g_{\mathrm{rev}} = 2G$ and $g_{\mathrm{rec}} = k$ of a multi-gate friend. The output lists, for each test, the best value found, the relaxed LF bound, the margin, and a shot-noise estimate of the shots needed for five standard deviations. Energy relaxation, coherent errors, crosstalk, and single-qubit gate errors are not modelled.

---

### 2. Formal Machine-Checked Proofs (Lean 4)

Install Lean 4 via [`elan`](https://github.com/leanprover/elan) (or open the folder in VS Code with the official `lean4` extension).

#### A. Instant Zero-Dependency Polytope Verification (`LFPolytope.lean`)
Because `LFPolytope.lean` uses only the core Lean 4 standard library, the kernel can verify it directly without downloading `Mathlib`:
```bash
lean lean/CovariantRecordHorizons/LFPolytope.lean
```
A silent exit (code `0`) confirms that the Lean 4 kernel independently verifies every theorem with zero `sorry` placeholders:
- `vertex_counts`: Confirms exact generation of all 64 deterministic LHV vertices and all 96 Two-Friend LF vertices (64 LHV + 32 Popescu–Rohrlich non-signaling sub-boxes).
- `valid_lf_inequalities`: Exhaustively verifies `Semi-Brukner <= 2`, `Brukner <= 2`, `I3322-LF <= 4`, and `Genuine LF <= 6` across every LF extreme point.
- `bellNonLF_separates_LHV_and_LF`: Proves `Bell non-LF <= 2` on LHV vertices while reaching `4` on the LF polytope.
- `genuineLF_is_strict_LF_facet`: Proves `Genuine LF == 6` holds on 14 LHV vertices (homogeneous rank `12 < 15`, ruling it out as a Bell facet) while achieving full facet rank `15` across the 96-vertex LF polytope.
- `semiBrukner_lhv_rank`, `semiBrukner_lf_rank`, `brukner_lhv_rank`, `brukner_lf_rank`, `i3322LF_lhv_rank`, `i3322LF_lf_rank`, `bellNonLF_lhv_rank`: Prove the remaining ranks of the manuscript's Table I (all `15`), so that every rank in that table is kernel-checked.

#### B. Full Lake Build & Continuous Quantum Bounds (`LFQuantumBounds.lean`)
Fetch precompiled `Mathlib` binaries first (avoids compiling `Mathlib` from source), then build the full package:
```bash
cd lean
lake exe cache get
lake build
```
This builds both modules and formally verifies `noisy_semiBrukner_sos_identity`, `theorem1_sos_identity`, `genuineLF_bloch_normalization`, and `genuineLF_quartic_excess`, proving the dimension-independent noncommutative $*$-algebra Sum-of-Squares certificates of Theorem 1 and Eq. (26) as well as the continuous Bloch strategy lower bound for Bong *et al.*'s `Genuine LF` inequality across all $f > 0$.

---

## Forking & Extending This Repository

This repository is designed to be forked and adapted for experimental and theoretical follow-up work:
* **Custom Quantum Hardware Noise Models:** `lf_noise_budget.py` is the starting point. Its `visibilities()` and `noisy_observable()` encode Pauli noise as setting-dependent visibility factors, and `behaviour_circuit()` is the density-matrix simulation they are checked against; extend the latter with $T_1$ relaxation, spectator-qubit crosstalk, or readout confusion matrices calibrated to a specific processor, and use it directly where the noise is no longer a visibility.
* **Asymmetric Bilateral Leakage ($f_A \neq f_B$):** Pass `f_bob` to `evaluate_ewfs_quantum_value()` or `optimize_quantum_bound_sweep()` in `lf_leak_tests.py` to give Bob's laboratory its own coherence retention (section 5 of the console report does this for the CHSH-type functionals), e.g. when modeling hybrid photonic-matter EWFS setups.
* **Higher-Setting Polytopes:** Extend `Corr15` in `LFPolytope.lean` and `EWFSIInequality` in `lf_leak_tests.py` to test $(4,4,2,2)$ or sequential multi-friend facets.

---

## Plain-Language Guide: What Is This Paper About?

*(Written for students, programmers, and curious readers without a graduate quantum physics background.)*

### 1. The "Friend in a Box" Puzzle
In everyday life, when something happens—say, a coin lands heads—it is simply a fact for everyone in the universe. Quantum mechanics makes things stranger. At the scale of atoms and photons, particles can exist in a **superposition** (a wave-like combination of possibilities) until a measurement forces a specific outcome.

In 1961, physicist Eugene Wigner asked a famous question: suppose a scientist ("the Friend") is sealed inside a perfectly isolated laboratory and measures a quantum particle, seeing a definite answer—say, **Up**. Meanwhile, Wigner stands outside the sealed lab. Because nothing has leaked out of the room, quantum theory says Wigner can still treat *the entire lab—including his Friend—as being in a wave-like superposition* of having seen **Up** and having seen **Down**, and could even run the lab's quantum evolution backward to erase the measurement!

Recent upgrades to this thought experiment (called **Local Friendliness** tests) combine Wigner's setup with pairs of entangled particles. They prove mathematically that if quantum mechanics continues to work on complex observers, **we cannot treat every observed outcome as an immediate, universal fact for everyone everywhere** without breaking the basic rules of cause and effect.

### 2. Cosmic Speed Limits and "Causal Receipts"
Nothing in nature travels faster than the speed of light. Imagine looking up at **Betelgeuse**, an unstable red supergiant star roughly 500 to 650 light-years from Earth. Betelgeuse is nearing the end of its life and could explode as a supernova any century now. In fact, it *might have already exploded 300 years ago*—and yet tonight, its red light is still shining calmly in our sky. Until the flash of light and neutrinos from that explosion physically reaches Earth, no trace of the explosion can affect anything in our neighborhood.

This paper applies that same strict speed-of-light bookkeeping directly to quantum measurements:
* An event only becomes a definite **True** or **False** fact for an observer when a **physical record** of it—think of it as a "causal receipt," like a flash of light, an electrical pulse in a sensor, or a bit written to memory—actually reaches that observer inside the speed-of-light limit (their **record horizon**).
* Before any physical receipt arrives, the distant outcome isn't merely a secret hidden in an envelope; relative to the outside observer, it remains genuinely **Undetermined**.
* Once two observers finally meet and compare notes, the underlying laws of quantum mechanics guarantee their receipts always match up cleanly without contradictions.

### 3. Why Everyday Objects Don't Stay Blurry (Leakage)
If outcomes depend on physical receipts, why don't humans ever experience blurry quantum superpositions in daily life? Because keeping a measurement reversible requires **zero leakage**. A human body contains roughly $10^{27}$ atoms constantly bouncing heat and stray light into the room—and at the ultimate physical limit, even unshieldable gravity carries tiny ripples across cosmic horizons. The instant even a tiny trace of a measurement leaks out where Wigner cannot gather it back up, the receipt is permanently locked in.

Because humans and cats are far too messy to isolate, physicists test these ideas on **quantum computers**, where a "Friend" can be stripped down to a single ultra-cold qubit (quantum bit) acting as a minimalist memory register. Even on the best quantum chips, however, tiny amounts of information still leak out.

### 4. How the Code in This Repo Backs Up the Paper
Think of this repository as a two-part engine that checks the paper's math from two independent angles:

1. **The Python Simulator (`lf_leak_tests.py`) — Testing Leaky Quantum Labs:**  
   The paper derives formulas predicting how much "leakage" a quantum processor can suffer before the signature quantum effect disappears. The Python code simulates thousands of measurement angles and noise levels across real experimental test formulas. One of its coolest findings is a huge difficulty jump: testing **one** isolated quantum "Friend" is fairly forgiving when a little information leaks out, whereas testing **two** isolated quantum "Friends" at the same time is drastically harder—any leak suppresses the quantum signal far more strongly (scaling like $f^4$ instead of $f^2$). That tells experimentalists exactly how clean their quantum chips need to be to see the effect. A companion script (`lf_noise_budget.py`) adds the other imperfections of a real chip, namely imperfect two-qubit gates and readout, and reports how good a processor has to be for each test.
2. **The Lean 4 Proof-Checker (`lean/`) — Bulletproof Math Verification:**  
   Traditionally, scientists check math proofs by reading them line by line, which means subtle human errors can occasionally slip past reviewers. **Lean 4** is a specialized programming language where mathematical theorems are written as code: if even a tiny logical step is flawed or incomplete, the computer refuses to compile the file. The Lean files in this repository check all 96 geometric corners and facet ranks of the quantum test rules, verify the noncommutative operator Sum-of-Squares identity of Theorem 1, and verify the algebraic identities behind the paper's lower bound step-by-step, so readers do not have to take those proofs on faith.

---

## Citation

If you use or fork this verification suite in your research, please cite:

```bibtex
@misc{baker2026covariant,
  title     = {Record Leakage in Local Friendliness Tests: Randomness Trade-off, Two-Friend Bounds, and Noise Budgets},
  author    = {Baker, Kevin J.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24435914}
}
```

## License

Released under the [MIT License](LICENSE).