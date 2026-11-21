#!/usr/bin/env python3
"""
lf_leak_tests.py
================
Numerical verification suite for:
"Covariant Record Horizons and the Abandonment of Absolute Events"

Verifies:
1. Polytope bounds and affine facet ranks (in R^15) across all 5 canonical
   inequality classes of Bong et al., Nature Physics 16, 1199-1205 (2020),
   confirming from first principles that 'Genuine LF' is a true facet of the
   (3,3,2,2) Local Friendliness polytope (rank 15) and NOT a Bell/LHV facet.
2. Exact analytical coherence-retention bound B_max(f) = 2 * sqrt(1 + f^2)
   = 2 * sqrt(2 - ell^2) for the One-Friend Semi-Brukner inequality, where
   f = |<e_0|e_1>| in [0, 1] is coherence retention and ell = sqrt(1 - f^2)
   is environmental record leakage.
3. Bilateral Two-Friend dephasing across Brukner, Bell non-LF, I3322-LF, and
   Genuine LF facets, demonstrating the transition from quadratic excess
   violation O(f^2) (One-Friend) to quartic suppression O(f^4) (Two-Friend)
   in the high-leakage regime (f -> 0, ell -> 1).
4. Critical coherence thresholds f_crit(mu) and maximum tolerable leakage
   ell_max(mu) under realistic quantum-processor visibility mu in (0, 1].

Usage:
    python3 lf_leak_tests.py
    python3 lf_leak_tests.py --plot --save-json lf_verification_results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize


# ---------------------------------------------------------------------------
# 1. Inequality Definitions (Bong et al., Nature Physics 2020, Table I)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EWFSIInequality:
    """
    Linear inequality on (3,3,2,2) binary-outcome expectations in {-1, +1}:
        S = sum_{x,y} J[x,y] <A_x B_y> + sum_x ca[x] <A_x> + sum_y cb[y] <B_y> <= bound
    Settings x, y in {0, 1, 2} correspond to manuscript labels {1, 2, 3},
    where index 0 (setting 1) is the direct friend-readout setting.
    """
    name: str
    J: np.ndarray               # shape (3, 3)
    ca: np.ndarray              # shape (3,)
    cb: np.ndarray              # shape (3,)
    bound: float
    alice_has_friend: bool
    bob_has_friend: bool
    category_note: str


def get_bong_2020_inequalities() -> List[EWFSIInequality]:
    """Returns the 5 canonical inequality classes from Bong et al. (2020) Table I."""
    return [
        EWFSIInequality(
            name="Semi-Brukner",
            J=np.array([
                [0.0, -1.0,  1.0],
                [0.0,  0.0,  0.0],
                [0.0, -1.0, -1.0],
            ]),
            ca=np.zeros(3),
            cb=np.zeros(3),
            bound=2.0,
            alice_has_friend=True,
            bob_has_friend=False,
            category_note="1-Friend LF & Bell facet (analytical 2*sqrt(1+f^2))",
        ),
        EWFSIInequality(
            name="Brukner",
            J=np.array([
                [-1.0,  1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [ 0.0,  0.0, 0.0],
            ]),
            ca=np.zeros(3),
            cb=np.zeros(3),
            bound=2.0,
            alice_has_friend=True,
            bob_has_friend=True,
            category_note="2-Friend LF & Bell facet (bilateral friend dephasing)",
        ),
        EWFSIInequality(
            name="Bell non-LF",
            J=np.array([
                [0.0,  0.0,  0.0],
                [0.0,  1.0, -1.0],
                [0.0, -1.0, -1.0],
            ]),
            ca=np.zeros(3),
            cb=np.zeros(3),
            bound=2.0,
            alice_has_friend=True,
            bob_has_friend=True,
            category_note="Bell facet ONLY (violated by LF PR-boxes up to 4.0)",
        ),
        EWFSIInequality(
            name="I3322-LF",
            J=np.array([
                [ 1.0, -1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  0.0],
            ]),
            ca=np.array([-1.0,  1.0, 0.0]),
            cb=np.array([ 1.0, -1.0, 0.0]),
            bound=4.0,
            alice_has_friend=True,
            bob_has_friend=True,
            category_note="3-setting 2-Friend LF & Bell facet",
        ),
        EWFSIInequality(
            name="Genuine LF",
            J=np.array([
                [-1.0, -2.0,  0.0],
                [-2.0,  2.0, -1.0],
                [ 0.0, -1.0, -1.0],
            ]),
            ca=np.array([-1.0, -1.0, 0.0]),
            cb=np.array([-1.0, -1.0, 0.0]),
            bound=6.0,
            alice_has_friend=True,
            bob_has_friend=True,
            category_note="True LF polytope facet (NOT a Bell/LHV facet)",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. Exact Polytope Vertex & Facet-Rank Verifier in R^15
# ---------------------------------------------------------------------------

def generate_lhv_and_lf_vertices() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates all extreme points in R^15 = (<A_1..3>, <B_1..3>, <A_1 B_1>..<A_3 B_3>)
    for:
      - The (3,3,2,2) Local Hidden Variable (LHV) polytope: 64 deterministic vertices.
      - The (3,3,2,2) Two-Friend Local Friendliness (LF) polytope: 64 LHV vertices
        plus 4 * 8 = 32 Popescu-Rohrlich (PR) non-signaling boxes on {2,3}x{2,3}
        conditioned on deterministic friend outcomes (c, d) in {-1, +1}^2.
    """
    lhv_vertices = []
    for outcomes in itertools.product([-1.0, 1.0], repeat=6):
        a = np.array(outcomes[:3])
        b = np.array(outcomes[3:])
        E = np.outer(a, b).ravel()
        lhv_vertices.append(np.concatenate([a, b, E]))
    lhv_arr = np.array(lhv_vertices)

    # 8 PR boxes on settings {1, 2} x {1, 2} (manuscript {2, 3} x {2, 3}):
    # Each entry in {-1, +1} with product E_11 * E_12 * E_21 * E_22 == -1
    pr_subboxes = [
        sub for sub in itertools.product([-1.0, 1.0], repeat=4)
        if sub[0] * sub[1] * sub[2] * sub[3] == -1.0
    ]

    pr_lf_vertices = []
    for c, d in itertools.product([-1.0, 1.0], repeat=2):
        for e11, e12, e21, e22 in pr_subboxes:
            a = np.array([c, 0.0, 0.0])
            b = np.array([d, 0.0, 0.0])
            E = np.array([
                [c * d, 0.0, 0.0],
                [0.0,   e11, e12],
                [0.0,   e21, e22],
            ]).ravel()
            pr_lf_vertices.append(np.concatenate([a, b, E]))

    lf_arr = np.vstack([lhv_arr, np.array(pr_lf_vertices)])
    return lhv_arr, lf_arr


@dataclass
class PolytopeCheckResult:
    name: str
    bound: float
    lhv_max: float
    lf_max: float
    lhv_tight_rank: int   # Full facet in R^15 requires homogeneous rank == 15
    lf_tight_rank: int
    is_bell_facet: bool
    is_lf_facet: bool


def verify_polytope_structure(
    inequalities: List[EWFSIInequality],
) -> List[PolytopeCheckResult]:
    """Verifies LHV/LF maxima and affine facet ranks (homogeneous rank == 15 in R^15)."""
    lhv_verts, lf_verts = generate_lhv_and_lf_vertices()
    results: List[PolytopeCheckResult] = []

    for ineq in inequalities:
        weights = np.concatenate([ineq.ca, ineq.cb, ineq.J.ravel()])
        lhv_vals = lhv_verts @ weights
        lf_vals = lf_verts @ weights

        lhv_max = float(np.max(lhv_vals))
        lf_max = float(np.max(lf_vals))

        # Tight vertices at ineq.bound (homogeneous coordinates [1, v] in R^16)
        lhv_tight = lhv_verts[np.isclose(lhv_vals, ineq.bound, atol=1e-9)]
        lf_tight = lf_verts[np.isclose(lf_vals, ineq.bound, atol=1e-9)]

        lhv_rank = (
            int(np.linalg.matrix_rank(np.column_stack([np.ones(len(lhv_tight)), lhv_tight])))
            if len(lhv_tight) > 0 else 0
        )
        lf_rank = (
            int(np.linalg.matrix_rank(np.column_stack([np.ones(len(lf_tight)), lf_tight])))
            if len(lf_tight) > 0 else 0
        )

        is_bell_facet = math.isclose(lhv_max, ineq.bound, abs_tol=1e-9) and (lhv_rank == 15)
        is_lf_facet = math.isclose(lf_max, ineq.bound, abs_tol=1e-9) and (lf_rank == 15)

        results.append(PolytopeCheckResult(
            name=ineq.name,
            bound=ineq.bound,
            lhv_max=lhv_max,
            lf_max=lf_max,
            lhv_tight_rank=lhv_rank,
            lf_tight_rank=lf_rank,
            is_bell_facet=is_bell_facet,
            is_lf_facet=is_lf_facet,
        ))
    return results


# ---------------------------------------------------------------------------
# 3. Analytical Reference Bounds (One-Friend & Two-Friend Brukner)
# ---------------------------------------------------------------------------

def analytical_semi_brukner(f: float, mu: float = 1.0) -> float:
    """Exact bound B_max^{SB}(f) = mu * 2 * sqrt(1 + f^2) = mu * 2 * sqrt(2 - ell^2)."""
    return mu * 2.0 * math.sqrt(1.0 + f * f)


def analytical_two_friend_brukner(f: float, mu: float = 1.0, grid_size: int = 4001) -> float:
    """
    Closed-form maximization for Two-Friend Brukner under symmetric coherence f:
        B_max^{Brukner}(f) = mu * max_{u in [-1, 1]}
            sqrt( [1 + 2u - u^2 + f^2(1 - u^2)]^2 + 4 f^2 (1 - u^2)(1 - u)^2 )
    """
    u = np.linspace(-1.0, 1.0, grid_size)
    one_minus_u2 = np.maximum(0.0, 1.0 - u * u)
    term_x = 1.0 + 2.0 * u - u * u + (f * f) * one_minus_u2
    term_y2 = 4.0 * (f * f) * one_minus_u2 * ((1.0 - u) ** 2)
    return float(mu * np.max(np.sqrt(term_x * term_x + term_y2)))


def analytical_genuine_lf_collinear_lower_bound(f: float) -> float:
    """
    Exact collinear-friend-basis family for Genuine LF at mu = 1:
        S_collinear(f) = 6 + f^4 / (1 + f^2)
    Demonstrates strict violation > 6 for all f > 0 with quartic O(f^4) onset.
    """
    return 6.0 + (f ** 4) / (1.0 + f * f)


# ---------------------------------------------------------------------------
# 4. Quantum EWFS Channel & Multi-Start / Homotopy Optimizer
# ---------------------------------------------------------------------------

def bloch_unit(theta: float, phi: float) -> np.ndarray:
    st = math.sin(theta)
    return np.array([st * math.cos(phi), st * math.sin(phi), math.cos(theta)])


def dephase_along_axis(n_vec: np.ndarray, axis: np.ndarray, f_party: float) -> np.ndarray:
    """
    Heisenberg-picture effective Bloch vector after uncomputing a friend who
    measured along `axis` with coherence retention `f_party` in [0, 1]:
        a_eff = f * n + (1 - f) * (n . u) * u
    """
    proj = float(np.dot(n_vec, axis))
    return f_party * n_vec + (1.0 - f_party) * proj * axis


def evaluate_ewfs_quantum_value(
    params: np.ndarray,
    ineq: EWFSIInequality,
    f: float,
    mu: float = 1.0,
    fix_maximally_entangled: bool = False,
) -> float:
    """
    Evaluates S for a general 2-qubit Schmidt state mixed with white noise:
        rho_mu = mu |psi(vartheta)><psi(vartheta)| + (1 - mu) I_4 / 4
        |psi(vartheta)> = cos(vartheta)|00> + sin(vartheta)|11>
    Parameter vector `params` (length 13):
      params[0]     : vartheta (ignored if fix_maximally_entangled=True -> pi/4)
      params[1..6]  : (theta, phi) for Alice settings x = 1, 2, 3
                      (when alice_has_friend=True, x=1 defines Charlie's axis u_C)
      params[7..12] : (theta, phi) for Bob settings y = 1, 2, 3
                      (when bob_has_friend=True, y=1 defines Debbie's axis u_D)
    """
    vartheta = (math.pi / 4.0) if fix_maximally_entangled else float(params[0])
    c2v = math.cos(2.0 * vartheta)
    s2v = math.sin(2.0 * vartheta)

    a_raw = [bloch_unit(params[1 + 2 * i], params[2 + 2 * i]) for i in range(3)]
    b_raw = [bloch_unit(params[7 + 2 * j], params[8 + 2 * j]) for j in range(3)]

    if ineq.alice_has_friend:
        u_c = a_raw[0]
        a_eff = [u_c, dephase_along_axis(a_raw[1], u_c, f), dephase_along_axis(a_raw[2], u_c, f)]
    else:
        a_eff = a_raw

    if ineq.bob_has_friend:
        u_d = b_raw[0]
        b_eff = [u_d, dephase_along_axis(b_raw[1], u_d, f), dephase_along_axis(b_raw[2], u_d, f)]
    else:
        b_eff = b_raw

    A_mat = np.array(a_eff)  # (3, 3)
    B_mat = np.array(b_eff)  # (3, 3)

    # <A_x> = mu * cos(2*vartheta) * a_{x,z}, <B_y> = mu * cos(2*vartheta) * b_{y,z}
    exp_a = mu * c2v * A_mat[:, 2]
    exp_b = mu * c2v * B_mat[:, 2]

    # <A_x B_y> = mu * (sin(2*vartheta)*(a_x b_x - a_y b_y) + a_z b_z)
    diag_t = np.array([s2v, -s2v, 1.0])
    exp_ab = mu * ((A_mat * diag_t) @ B_mat.T)

    val = float(np.sum(ineq.J * exp_ab) + np.dot(ineq.ca, exp_a) + np.dot(ineq.cb, exp_b))
    return val


def build_analytical_warm_starts(ineq: EWFSIInequality, f: float) -> List[np.ndarray]:
    """Constructs targeted analytical warm-start seeds in the X-Z plane (phi=0)."""
    seeds: List[np.ndarray] = []

    # Collinear small-f stationary angle cos(theta*) = 1 / (1 + f^2)
    th_star = math.acos(min(1.0, 1.0 / (1.0 + f * f)))

    # Seed 1: Genuine LF collinear quartic branch (A1=B1=Z, A2=B2=-Z, A3/B3=+-th_star)
    s1 = np.zeros(13)
    s1[0] = math.pi / 4.0
    s1[1], s1[3], s1[5] = 0.0, math.pi, th_star
    s1[7], s1[9], s1[11] = 0.0, math.pi, -th_star
    seeds.append(s1)

    # Seed 2: Genuine LF high-f tilted branch (tau = pi/8, gamma = 5*pi/8)
    s2 = np.zeros(13)
    s2[0] = math.pi / 4.0
    tau, gamma = math.pi / 8.0, 5.0 * math.pi / 8.0
    s2[1], s2[3], s2[5] = math.pi - tau, tau, gamma
    s2[7], s2[9], s2[11] = math.pi + tau, -tau, -gamma
    seeds.append(s2)

    # Seed 3: Standard Semi-Brukner / CHSH optimal orientation
    s3 = np.zeros(13)
    s3[0] = math.pi / 4.0
    phi_sb = math.atan2(f, 1.0)
    s3[1], s3[3], s3[5] = 0.0, math.pi / 2.0, math.pi / 2.0
    s3[7], s3[9], s3[11] = 0.0, math.pi - phi_sb, -phi_sb
    seeds.append(s3)

    # Seed 4: Brukner collinear-to-tilted seed
    s4 = np.zeros(13)
    s4[0] = math.pi / 4.0
    s4[1], s4[3], s4[5] = math.pi, th_star, 0.0
    s4[7], s4[9], s4[11] = 0.0, -th_star, 0.0
    seeds.append(s4)

    return seeds


def optimize_quantum_bound_sweep(
    ineq: EWFSIInequality,
    f_grid: np.ndarray,
    mu: float = 1.0,
    fix_maximally_entangled: bool = False,
    n_random_starts: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Computes the global quantum maximum S_max(f) across `f_grid` using:
      1. Analytical stationary-point seeds,
      2. Seeded 3D random multi-starts,
      3. Bidirectional homotopy continuation (sweeping 0 -> 1 and 1 -> 0)
         so sharp branch crossovers near small f are resolved accurately.
    """
    rng = np.random.default_rng(seed)
    n_pts = len(f_grid)
    best_vals = np.full(n_pts, -np.inf)
    best_params: List[Optional[np.ndarray]] = [None] * n_pts

    def run_local(x0: np.ndarray, f_val: float) -> Tuple[float, np.ndarray]:
        res = minimize(
            lambda p: -evaluate_ewfs_quantum_value(p, ineq, f_val, mu, fix_maximally_entangled),
            x0,
            method="L-BFGS-B",
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
        )
        return float(-res.fun), res.x

    # Pass 1: Forward sweep (low f -> high f) with random + analytical + continuation
    prev_opt: Optional[np.ndarray] = None
    for idx, f_val in enumerate(f_grid):
        candidates = build_analytical_warm_starts(ineq, float(f_val))
        if prev_opt is not None:
            candidates.append(prev_opt)
        for _ in range(n_random_starts):
            rnd = rng.uniform(-math.pi, math.pi, size=13)
            rnd[0] = rng.uniform(0.05, math.pi / 2.0 - 0.05)
            candidates.append(rnd)

        for x0 in candidates:
            val, x_opt = run_local(x0, float(f_val))
            if val > best_vals[idx]:
                best_vals[idx] = val
                best_params[idx] = x_opt
        prev_opt = best_params[idx]

    # Pass 2: Backward homotopy sweep (high f -> low f) to polish crossovers
    next_opt: Optional[np.ndarray] = None
    for idx in range(n_pts - 1, -1, -1):
        f_val = float(f_grid[idx])
        if next_opt is not None:
            val, x_opt = run_local(next_opt, f_val)
            if val > best_vals[idx]:
                best_vals[idx] = val
                best_params[idx] = x_opt
        next_opt = best_params[idx]

    return best_vals


# ---------------------------------------------------------------------------
# 5. Visibility-vs-Leakage Critical Threshold Solver
# ---------------------------------------------------------------------------

def solve_critical_coherence(
    ineq: EWFSIInequality,
    mu: float,
    fix_maximally_entangled: bool = False,
) -> Optional[float]:
    """
    Finds the minimum coherence retention f_crit in [0, 1] required to violate
    `ineq` (S_max(f, mu) > ineq.bound) at Werner state visibility `mu`.
    Returns None if no violation occurs even at perfect coherence f = 1.
    """
    s_at_zero = optimize_quantum_bound_sweep(
        ineq, np.array([0.0]), mu=mu, fix_maximally_entangled=fix_maximally_entangled, n_random_starts=24
    )[0]
    if s_at_zero > ineq.bound + 1e-7:
        return 0.0

    s_at_one = optimize_quantum_bound_sweep(
        ineq, np.array([1.0]), mu=mu, fix_maximally_entangled=fix_maximally_entangled, n_random_starts=32
    )[0]
    if s_at_one <= ineq.bound + 1e-7:
        return None

    def gap(f_scalar: float) -> float:
        val = optimize_quantum_bound_sweep(
            ineq,
            np.array([f_scalar]),
            mu=mu,
            fix_maximally_entangled=fix_maximally_entangled,
            n_random_starts=20,
        )[0]
        return val - ineq.bound

    return float(brentq(gap, 0.0, 1.0, xtol=1e-4))


# ---------------------------------------------------------------------------
# 6. CLI Reporting & Optional Manuscript Plot Generation
# ---------------------------------------------------------------------------

def maybe_generate_plots(
    f_grid: np.ndarray,
    excess_curves: Dict[str, np.ndarray],
    output_prefix: str = "lf_leakage_bounds",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARN] matplotlib is not installed; skipping figure generation.")
        return

    ell_grid = np.sqrt(np.maximum(0.0, 1.0 - f_grid ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    styles = {
        "Semi-Brukner": ("#1f77b4", "-", "Semi-Brukner (1-Friend, CHSH)"),
        "Brukner": ("#ff7f0e", "--", "Brukner (2-Friend)"),
        "Bell non-LF": ("#7f7f7f", ":", "Bell non-LF (2-Friend uncomputed)"),
        "I3322-LF": ("#2ca02c", "-.", "I3322-LF (2-Friend, 3-setting)"),
        "Genuine LF": ("#d62728", "-", "Genuine LF (Bong et al. facet)"),
    }

    # Panel A: Excess quantum violation Delta S = S_max(f) - Bound vs. Coherence Retention f
    ax0 = axes[0]
    for name, curve in excess_curves.items():
        color, ls, label = styles[name]
        ax0.plot(f_grid, np.maximum(0.0, curve), color=color, linestyle=ls, lw=2.0, label=label)
    ax0.set_xlabel(r"Coherence retention $f = |\langle e_0 | e_1 \rangle|$")
    ax0.set_ylabel(r"Quantum violation excess $\Delta S = S_{\max}(f) - B_{\mathrm{LF}}$")
    ax0.set_title("(a) Violation Excess Across Coherence Retention")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8.5, loc="upper left")

    # Panel B: Log-log asymptotic scaling near high leakage (f -> 0, ell -> 1)
    ax1 = axes[1]
    mask = (f_grid > 0.04) & (f_grid <= 0.65)
    for name in ["Semi-Brukner", "Brukner", "Genuine LF"]:
        color, ls, label = styles[name]
        vals = np.maximum(1e-12, excess_curves[name][mask])
        ax1.loglog(f_grid[mask], vals, color=color, linestyle=ls, lw=2.2, marker="o", ms=4, label=label)

    # Reference slopes O(f^2) and O(f^4)
    f_ref = f_grid[mask]
    ax1.loglog(f_ref, f_ref ** 2, "k--", lw=1.0, alpha=0.5, label=r"Reference $\propto f^2$")
    ax1.loglog(f_ref, (f_ref ** 4) / (1.0 + f_ref ** 2), "k:", lw=1.2, alpha=0.6, label=r"Reference $f^4/(1+f^2)$")
    ax1.set_xlabel(r"Coherence retention $f$ ($\ell = \sqrt{1 - f^2} \to 1$ at left)")
    ax1.set_ylabel(r"Excess violation $\Delta S(f)$ (log scale)")
    ax1.set_title(r"(b) One-Friend $\mathcal{O}(f^2)$ vs. Two-Friend $\mathcal{O}(f^4)$ Scaling")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.0, loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{output_prefix}.pdf")
    fig.savefig(f"{output_prefix}.png", dpi=220)
    print(f"\n[OK] Saved publication figures: {output_prefix}.pdf and {output_prefix}.png")
    _ = ell_grid  # available if custom ell-axis plotting is desired


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Friendliness Leakage Verification Suite")
    parser.add_argument("--plot", action="store_true", help="Export PDF/PNG figures via matplotlib")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to export JSON summary")
    parser.add_argument("--maximally-entangled-only", action="store_true",
                        help="Fix |Phi+> (vartheta=pi/4) instead of optimizing general Schmidt states")
    args = parser.parse_args()

    inequalities = get_bong_2020_inequalities()

    # --- Step 1: Polytope & Facet-Rank Verification ---
    print("=" * 86)
    print("1. EXACT (3,3,2,2) POLYTOPE & AFFINE FACET-RANK VERIFICATION (IN R^15)")
    print("=" * 86)
    poly_results = verify_polytope_structure(inequalities)
    print(f"{'Inequality':<15} {'Bound':>6} {'LHV Max':>8} {'LF Max':>8} "
          f"{'LHV Rank':>9} {'LF Rank':>8} {'Bell Facet?':>12} {'LF Facet?':>10}")
    print("-" * 86)
    for pr in poly_results:
        print(f"{pr.name:<15} {pr.bound:6.1f} {pr.lhv_max:8.1f} {pr.lf_max:8.1f} "
              f"{pr.lhv_tight_rank:9d} {pr.lf_tight_rank:8d} "
              f"{str(pr.is_bell_facet):>12} {str(pr.is_lf_facet):>10}")

    # --- Step 2: Coherence-Retention Sweep at Pure State Visibility (mu = 1.0) ---
    f_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    raw_curves: Dict[str, np.ndarray] = {}
    excess_curves: Dict[str, np.ndarray] = {}

    for ineq in inequalities:
        vals = optimize_quantum_bound_sweep(
            ineq,
            f_grid,
            mu=1.0,
            fix_maximally_entangled=args.maximally_entangled_only,
            n_random_starts=24,
        )
        raw_curves[ineq.name] = vals
        excess_curves[ineq.name] = vals - ineq.bound

    # Analytical benchmark comparisons
    sb_exact = np.array([analytical_semi_brukner(float(f)) for f in f_grid])
    sb_max_err = float(np.max(np.abs(raw_curves["Semi-Brukner"] - sb_exact)))

    bruk_exact = np.array([analytical_two_friend_brukner(float(f)) for f in f_grid])
    bruk_max_err = float(np.max(np.abs(raw_curves["Brukner"] - bruk_exact)))

    print("\n" + "=" * 86)
    print("2. ANALYTICAL VS. NUMERICAL EXACTNESS CHECKS (mu = 1.0)")
    print("=" * 86)
    print(f"Max |S_num - 2*sqrt(1 + f^2)| on Semi-Brukner : {sb_max_err:.3e}")
    print(f"Max |S_num - B_exact(f)|      on 2-Fr Brukner : {bruk_max_err:.3e}")

    print("\n" + "=" * 86)
    print("3. EXCESS QUANTUM VIOLATION Delta S = S_max(f) - Bound ACROSS LEAKAGE")
    print("   (f = |<e_0|e_1>| coherence retention, ell = sqrt(1 - f^2) record leakage)")
    print("=" * 86)
    header = (f"{'f':>5} {'ell':>6} | {'Semi-Brukner':>13} {'Brukner (2F)':>13} "
              f"{'Bell non-LF':>12} {'I3322-LF':>11} {'Genuine LF':>12}")
    print(header)
    print("-" * 86)
    for idx, f_val in enumerate(f_grid):
        ell_val = math.sqrt(max(0.0, 1.0 - float(f_val) ** 2))
        print(
            f"{f_val:5.2f} {ell_val:6.3f} | "
            f"{excess_curves['Semi-Brukner'][idx]:+13.6f} "
            f"{excess_curves['Brukner'][idx]:+13.6f} "
            f"{excess_curves['Bell non-LF'][idx]:+12.6f} "
            f"{excess_curves['I3322-LF'][idx]:+11.6f} "
            f"{excess_curves['Genuine LF'][idx]:+12.6f}"
        )

    # --- Step 3: Critical Coherence Thresholds Under Hardware Visibility mu < 1 ---
    print("\n" + "=" * 86)
    print("4. CRITICAL COHERENCE f_crit(mu) [MAX LEAKAGE ell_max] UNDER HARDWARE NOISE")
    print("   (Fixes maximally entangled Werner state rho_mu = mu |Phi+><Phi+| + (1-mu)I/4)")
    print("=" * 86)
    mu_list = [0.75, 0.82, 0.88, 0.92, 0.96, 1.00]
    target_names = ["Bell non-LF", "Semi-Brukner", "Brukner", "Genuine LF"]
    target_ineqs = [iq for iq in inequalities if iq.name in target_names]

    print(f"{'mu':>6} | " + " | ".join(f"{iq.name:^18}" for iq in target_ineqs))
    print("-" * 86)
    threshold_records = {}
    for mu_val in mu_list:
        cells = []
        threshold_records[str(mu_val)] = {}
        for iq in target_ineqs:
            fc = solve_critical_coherence(iq, mu=mu_val, fix_maximally_entangled=True)
            if fc is None:
                cells.append(f"{'No violation':^18}")
                threshold_records[str(mu_val)][iq.name] = None
            else:
                ell_max = math.sqrt(max(0.0, 1.0 - fc * fc))
                cells.append(f"f>{fc:4.2f} (l<{ell_max:4.2f})")
                threshold_records[str(mu_val)][iq.name] = {
                    "f_crit": round(fc, 5),
                    "ell_max": round(ell_max, 5),
                }
        print(f"{mu_val:6.2f} | " + " | ".join(cells))

    if args.save_json:
        payload = {
            "f_grid": f_grid.tolist(),
            "ell_grid": np.sqrt(np.maximum(0.0, 1.0 - f_grid ** 2)).tolist(),
            "polytope_checks": [pr.__dict__ for pr in poly_results],
            "semi_brukner_max_abs_err": sb_max_err,
            "brukner_max_abs_err": bruk_max_err,
            "excess_violation_mu_1": {k: v.tolist() for k, v in excess_curves.items()},
            "visibility_thresholds_werner": threshold_records,
        }
        with open(args.save_json, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"\n[OK] Exported JSON verification summary to: {args.save_json}")

    if args.plot:
        maybe_generate_plots(f_grid, excess_curves)


if __name__ == "__main__":
    main()