# Covariant Record Horizons and the Abandonment of Absolute Events

[![arXiv](https://img.shields.io/badge/arXiv-TODO.XXXXX%20%5Bquant--ph%5D-b31b1b.svg)](https://arxiv.org/abs/TODO.XXXXX)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Lean 4](https://img.shields.io/badge/Verified%20in-Lean%204-5A67D8.svg)](https://lean-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion numerical simulation and formal machine-checked verification repository for the manuscript:

> **Covariant Record Horizons and the Abandonment of Absolute Events**  
> Preprint: `arXiv:TODO.XXXXX [quant-ph]` *(link active upon public timestamp)*

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
| Exact One-Friend Semi-Brukner bound $B_{\max}^{\mathrm{SB}}(f) = 2\sqrt{1 + f^2} = 2\sqrt{2 - \ell^2}$ | `lf_leak_tests.py` | Analytical benchmark vs. multi-start Bloch/Schmidt optimization ($< 10^{-15}$ error) |

*(Note: Relativistic horizon soft-graviton/soft-photon ceilings following Danielson, Satishchandran, and Wald are treated analytically in the manuscript text and are not simulated here.)*

---

## Repository Structure

```text
.
├── lf_leak_tests.py                         # Numerical verifier and optimization suite
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
│       └── LFQuantumBounds.lean            # Mathlib real-algebra proofs
├── manuscript/
│   ├── main.tex                             # RevTeX manuscript driver
│   ├── Makefile                             # Clean rebuild, audit, and incremental targets
│   ├── build.sh                             # Cross-platform convenience wrapper
│   ├── README.md                            # Manuscript build documentation
│   ├── sections/                            # Individual section .tex sources
│   ├── bibliography/references.bib          # BibTeX reference database
│   ├── figures/                             # Generated PDF/PNG publication figures
│   └── data/lf_verification_results.json    # Generated numerical verification data
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

**Generate publication PDF/PNG figures and export JSON data:**
```bash
python3 lf_leak_tests.py --plot --save-json lf_verification_results.json
```
This produces `lf_leakage_bounds.pdf` and `lf_leakage_bounds.png`, illustrating:
* **Panel (a):** Quantum violation excess $\Delta S(f) = S_{\max}(f) - B_{\mathrm{LF}}$ across coherence retention $f \in [0, 1]$ for all five canonical EWFS inequalities (`Semi-Brukner`, `Brukner`, `Bell non-LF`, `I3322-LF`, and `Genuine LF`).
* **Panel (b):** Log-log asymptotic scaling in the high-leakage limit ($f \to 0, \ell \to 1$), demonstrating the sharp split between **quadratic** One-Friend excess $\Delta S \simeq f^2$ and **quartic** Two-Friend excess $\Delta S \simeq f^4 / (1 + f^2)$.

**Restrict state optimization to maximally entangled Bell states ($|\Phi^+\rangle$):**
```bash
python3 lf_leak_tests.py --maximally-entangled-only
```

---

### 2. Formal Machine-Checked Proofs (Lean 4)

Install Lean 4 via [`elan`](https://github.com/leanprover/elan) (or open the folder in VS Code with the official `lean4` extension).

#### A. Instant Zero-Dependency Polytope Verification (`LFPolytope.lean`)
Because `LFPolytope.lean` uses only the core Lean 4 standard library, the kernel can verify it directly in under one second without downloading `Mathlib`:
```bash
lean lean/CovariantRecordHorizons/LFPolytope.lean
```
A silent exit (code `0`) confirms that the Lean 4 kernel independently verifies every theorem with zero `sorry` placeholders:
- `vertex_counts`: Confirms exact generation of all 64 deterministic LHV vertices and all 96 Two-Friend LF vertices (64 LHV + 32 Popescu–Rohrlich non-signaling sub-boxes).
- `valid_lf_inequalities`: Exhaustively verifies `Semi-Brukner <= 2`, `Brukner <= 2`, `I3322-LF <= 4`, and `Genuine LF <= 6` across every LF extreme point.
- `bellNonLF_separates_LHV_and_LF`: Proves `Bell non-LF <= 2` on LHV vertices while reaching `4` on the LF polytope.
- `genuineLF_is_strict_LF_facet`: Proves `Genuine LF == 6` holds on 14 LHV vertices (homogeneous rank `12 < 15`, ruling it out as a Bell facet) while achieving full facet rank `15` across the 96-vertex LF polytope.

#### B. Full Lake Build & Continuous Quantum Bounds (`LFQuantumBounds.lean`)
Fetch precompiled `Mathlib` binaries first (avoids compiling `Mathlib` from source), then build the full package:
```bash
cd lean
lake exe cache get
lake build
```
This builds both modules and formally verifies `genuineLF_bloch_normalization` and `genuineLF_quartic_excess`, proving over $\mathbb{R}$ that the analytical Bloch measurement family violates Bong *et al.*'s `Genuine LF` inequality for every nonzero coherence retention $f > 0$.

---

## Forking & Extending This Repository

This repository is designed to be forked and adapted for experimental and theoretical follow-up work:
* **Custom Quantum Hardware Noise Models:** Replace the isotropic Werner state visibility parameter `mu` in `evaluate_ewfs_quantum_value()` (`lf_leak_tests.py`) with asymmetric $T_1 / T_2$ relaxation, spectator-qubit crosstalk, or mid-circuit ancilla readout confusion matrices calibrated to specific superconducting or trapped-ion processors.
* **Asymmetric Bilateral Leakage ($f_A \neq f_B$):** Modify `dephase_along_axis()` calls in `lf_leak_tests.py` to pass independent coherence retention factors `(f_A, f_B)` when modeling hybrid photonic-matter EWFS setups.
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
   The paper derives formulas predicting how much "leakage" a quantum processor can suffer before the signature quantum effect disappears. The Python code simulates thousands of measurement angles and noise levels across real experimental test formulas. One of its coolest findings is a huge difficulty jump: testing **one** isolated quantum "Friend" is fairly forgiving when a little information leaks out, whereas testing **two** isolated quantum "Friends" at the same time is drastically harder—any tiny leak crushes the quantum signal exponentially faster (scaling like $f^4$ instead of $f^2$). That tells experimentalists exactly how clean their quantum chips need to be to see the effect.
2. **The Lean 4 Proof-Checker (`lean/`) — Bulletproof Math Verification:**  
   Traditionally, scientists check math proofs by reading them line by line, which means subtle human errors can occasionally slip past reviewers. **Lean 4** is a specialized programming language where mathematical theorems are written as code: if even a tiny logical step is flawed or incomplete, the computer refuses to compile the file. The Lean files in this repository check all 96 geometric corners of the quantum test rules and verify the core algebraic inequalities step-by-step, giving readers 100% mathematical certainty without having to take the algebra on faith.

---

## Citation

If you use or fork this verification suite in your research, please cite:

```bibtex
@article{baker2026covariant,
  title   = {Covariant Record Horizons and the Abandonment of Absolute Events},
  author  = {Baker, Kevin J.},
  journal = {arXiv preprint arXiv:TODO.XXXXX},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).