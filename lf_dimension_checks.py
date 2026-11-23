#!/usr/bin/env python3
"""
LF functionals under record leakage: beyond two qubits and rank-one measurements
================================================================================

lf_leak_tests.py optimizes over two-qubit states and rank-one projective reversal
measurements (Bloch vectors). This companion check lifts both restrictions:

  * local dimension 2, 3 or 4, with a binary friend measurement {P+, P-} that is
    degenerate once the dimension exceeds two;
  * reversal observables of any rank, including constant-outcome ones.

Model. Setting 1 on each side is the friend's outcome, A_1 = P+ - P-. For settings
2 and 3 the superobserver undoes the friend's measurement while the environment keeps
a marker with overlap f = |<e_+|e_->|, so the effective observable is

    Phi_f(N) = P+ N P+ + P- N P- + f (P+ N P- + P- N P+).

Method. Every functional is linear in each N. For a fixed state and fixed observables
of the other party the best N is therefore a matrix sign function, and for fixed
observables the best state is the top eigenvector of the Bell operator. Iterating the
two steps from random starts is a seesaw.

Reading the output. Each number is the value of an explicit strategy and hence a
rigorous LOWER bound on the maximum. The seesaw can stop in a local optimum, most
easily when the excess is tiny, so a value below the one printed by lf_leak_tests.py
proves nothing; a value above it does.

What the table shows (see Sec. III.D of the manuscript):
  * Brukner: the two-qubit optimum is not exceeded in dimension 3 or 4, as Jordan's
    lemma implies (each side pairs the friend's outcome with one reversal observable).
  * Bell non-LF and two-friend Semi-Brukner: a degenerate friend measurement leaves
    room for reversal observables inside one eigenspace, which the leak does not
    touch, so the two-qubit bound 2*sqrt(1 + fA^2 fB^2) is exceeded.
  * Genuine LF, Genuine LF 2 and both I3322 classes: in dimension 3 the best strategies
    found have an excess of order f^2 instead of f^4 (for Genuine LF, with one friend's
    outcome deterministic).
  * I3322-LF, two qubits: a constant-outcome observable on each side beats the
    rank-one optimum below f = 0.83 (the reduction used in lf_leak_tests.py).

Usage:
    python3 lf_dimension_checks.py              # about a minute
    python3 lf_dimension_checks.py --quick      # fewer restarts
"""

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np

Projectors = Tuple[np.ndarray, np.ndarray]

# Coefficient tables J[x][y] of A_x (x) B_y, with index 0 = identity and 1..3 = settings.
FUNCTIONALS: Dict[str, Tuple[List[List[float]], float]] = {
    "Semi-Brukner (2F)": ([[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0], [0, 0, -1, -1]], 2.0),
    "Brukner":           ([[0, 0, 0, 0], [0, -1, 1, 0], [0, -1, -1, 0], [0, 0, 0, 0]], 2.0),
    "Bell non-LF":       ([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, -1], [0, 0, -1, -1]], 2.0),
    "I3322-LF":          ([[0, 1, -1, 0], [-1, 1, -1, -1], [1, -1, 1, -1], [0, -1, -1, 0]], 4.0),
    "Genuine LF":        ([[0, -1, -1, 0], [-1, -1, -2, 0], [-1, -2, 2, -1], [0, 0, -1, -1]], 6.0),
    "Genuine LF 2":      ([[0, -1, 0, 0], [-1, -1, -2, 0], [-1, -1, 1, -1], [-1, -1, 1, 1]], 5.0),
    "I3322-LF (2,3)":    ([[0, 0, -1, -1], [0, 0, -1, 1], [-1, -1, -1, -1], [-1, 1, -1, -1]], 4.0),
}


def friend_projectors(dim: int, rank_plus: int) -> Projectors:
    """Binary friend measurement: P+ projects on the first `rank_plus` basis states."""
    p_plus = np.zeros((dim, dim), dtype=complex)
    p_plus[:rank_plus, :rank_plus] = np.eye(rank_plus)
    return p_plus, np.eye(dim, dtype=complex) - p_plus


def dephase(obs: np.ndarray, proj: Projectors, f: float) -> np.ndarray:
    """Effective observable Phi_f(N) after an imperfect reversal."""
    p_plus, p_minus = proj
    return (p_plus @ obs @ p_plus + p_minus @ obs @ p_minus
            + f * (p_plus @ obs @ p_minus + p_minus @ obs @ p_plus))


def matrix_sign(herm: np.ndarray) -> np.ndarray:
    """+-1-valued observable maximizing Tr[N herm]; zero eigenvalues are sent to +1."""
    vals, vecs = np.linalg.eigh((herm + herm.conj().T) / 2.0)
    return (vecs * np.where(vals >= 0.0, 1.0, -1.0)) @ vecs.conj().T


def seesaw(
    coeffs: List[List[float]],
    proj_a: Projectors,
    proj_b: Projectors,
    f_alice: float,
    f_bob: float,
    rng: np.random.Generator,
    restarts: int,
    max_iter: int = 800,
) -> float:
    """Best value of the functional found from `restarts` random starting points."""
    dim_a, dim_b = proj_a[0].shape[0], proj_b[0].shape[0]
    id_a, id_b = np.eye(dim_a, dtype=complex), np.eye(dim_b, dtype=complex)
    a_1, b_1 = proj_a[0] - proj_a[1], proj_b[0] - proj_b[1]

    def random_observable(dim: int) -> np.ndarray:
        h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        return matrix_sign(h + h.conj().T)

    def effective(raw: Dict[int, np.ndarray], first, identity, proj, f) -> Dict[int, np.ndarray]:
        return {0: identity, 1: first, 2: dephase(raw[2], proj, f), 3: dephase(raw[3], proj, f)}

    best = -math.inf
    for _ in range(restarts):
        raw_a = {x: random_observable(dim_a) for x in (2, 3)}
        raw_b = {y: random_observable(dim_b) for y in (2, 3)}
        previous = -math.inf
        for iteration in range(max_iter):
            a = effective(raw_a, a_1, id_a, proj_a, f_alice)
            b = effective(raw_b, b_1, id_b, proj_b, f_bob)
            bell = sum(coeffs[x][y] * np.kron(a[x], b[y])
                       for x in range(4) for y in range(4) if coeffs[x][y])
            vals, vecs = np.linalg.eigh(bell)
            value, psi = float(vals[-1]), vecs[:, -1]
            if abs(value - previous) < 1e-14 and iteration > 10:
                break
            previous = value
            rho = np.outer(psi, psi.conj()).reshape(dim_a, dim_b, dim_a, dim_b)
            for x in (2, 3):
                partner = sum(coeffs[x][y] * b[y] for y in range(4))
                reduced = np.einsum("abcd,db->ac", rho, partner)      # Tr_B[rho (1 (x) partner)]
                raw_a[x] = matrix_sign(dephase(reduced.conj().T, proj_a, f_alice))
            a = effective(raw_a, a_1, id_a, proj_a, f_alice)
            for y in (2, 3):
                partner = sum(coeffs[x][y] * a[x] for x in range(4))
                reduced = np.einsum("abcd,ca->bd", rho, partner)      # Tr_A[rho (partner (x) 1)]
                raw_b[y] = matrix_sign(dephase(reduced.conj().T, proj_b, f_bob))
        best = max(best, previous)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="Fewer random restarts (less reliable)")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    scale = 0.4 if args.quick else 1.0
    configs = [
        ("2 x 2 (1+1)", friend_projectors(2, 1), max(4, int(60 * scale))),
        ("3 x 3 (1+2)", friend_projectors(3, 1), max(4, int(30 * scale))),
        ("4 x 4 (2+2)", friend_projectors(4, 2), max(4, int(20 * scale))),
    ]
    f_values = [0.3, 0.5, 0.8]

    print("=" * 78)
    print("EXCESS OVER THE TABLE I BOUND, BOTH LABORATORIES LEAKY (f_A = f_B = f)")
    print("local dimension (ranks of the friend's two projectors); best strategy found")
    print("=" * 78)
    results: Dict[str, Dict[float, List[float]]] = {}
    for name, (coeffs, bound) in FUNCTIONALS.items():
        print(f"\n{name}  (bound {bound:g})")
        print("    f   " + "".join(f"{label:>14}" for label, _, _ in configs))
        results[name] = {}
        for f in f_values:
            row = [seesaw(coeffs, proj, proj, f, f, rng, restarts) - bound for _, proj, restarts in configs]
            results[name][f] = row
            print(f"  {f:4.2f}  " + "".join(f"{v:14.6f}" for v in row), flush=True)

    print("\n" + "=" * 78)
    print("CHECKS")
    print("=" * 78)
    # 1. Full leakage: every LF facet must return to its bound in every dimension.
    worst_full_leak = 0.0
    for name, (coeffs, bound) in FUNCTIONALS.items():
        if name == "Bell non-LF":
            continue
        for _, proj, _ in configs:
            worst_full_leak = max(worst_full_leak, seesaw(coeffs, proj, proj, 0.0, 0.0, rng, 6) - bound)
    print(f"f = 0, all LF facets, all dimensions: largest excess {worst_full_leak:.2e} (must vanish)")
    # 2. Brukner: no gain from dimension.
    bruk_gain = max(max(row[1:]) - row[0] for row in results["Brukner"].values())
    print(f"Brukner: largest gain of dimension 3 or 4 over two qubits {bruk_gain:+.2e} (Jordan's lemma: none)")
    # 3. Two-qubit product bound and its failure beyond qubits.
    for f in f_values:
        product = 2.0 * math.sqrt(1.0 + f ** 4) - 2.0
        one_lab = 2.0 * math.sqrt(1.0 + f ** 2) - 2.0
        sb = results["Semi-Brukner (2F)"][f]
        print(f"f = {f}: Semi-Brukner (2F) two qubits {sb[0]:.6f} [2 sqrt(1+f^4) - 2 = {product:.6f}], "
              f"dimension 3 {sb[1]:.6f} [2 sqrt(1+f^2) - 2 = {one_lab:.6f}]")
    # 4. Genuine LF: dimension 3 against two qubits.
    for f in f_values:
        glf = results["Genuine LF"][f]
        print(f"f = {f}: Genuine LF two qubits {glf[0]:.6f}, dimension 3 {glf[1]:.6f} "
              f"(ratio to f^2: {glf[1] / f ** 2:.3f})")


if __name__ == "__main__":
    main()
