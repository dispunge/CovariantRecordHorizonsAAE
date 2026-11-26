#!/usr/bin/env python3
"""
Noise budget for circuit implementations of Local Friendliness tests
====================================================================

Companion to ``lf_leak_tests.py``. That script varies one imperfection, the
coherence retention ``f`` of the friend's record. A processor implementation
has three more, and they enter the tests differently for different settings.
This script puts all four into one model and answers, for a given device,
which Local Friendliness (LF) test is violated and by how much.

Circuit model (one laboratory with a friend)
--------------------------------------------
* System qubit S, friend qubit F prepared in |0>.
* Friend's measurement: CNOT(S -> F), followed by two-qubit depolarizing
  noise  rho -> (1 - lam) rho + lam 1/4  on SF.  The average gate infidelity
  is r = 3 lam / 4.
* The record then dephases with coherence retention f (leakage
  ell = sqrt(1 - f^2)), as in the manuscript.  For a record held for a time t
  on qubits with pure-dephasing times T_phi,S and T_phi,F this is
  f = exp(-t / T_phi,S - t / T_phi,F).
* Setting 1: the superobserver reads F, with symmetric assignment error eps.
* Settings 2, 3: the CNOT is applied again (same depolarizing noise) and S is
  measured along a chosen axis, with the same assignment error.
* A laboratory without a friend measures S directly (assignment error eps).
* The entangled pair is mixed with white noise at visibility mu; preparing it
  with one entangling gate of the same quality gives mu = 1 - lam, which is
  the default.

In the Heisenberg picture this multiplies the ideal observables by

    eta_1 = (1 - lam) (1 - 2 eps)        (friend's outcome, one gate)
    eta_2 = (1 - lam)^2 (1 - 2 eps)      (reversal settings, two gates)
    eta_0 = 1 - 2 eps                    (laboratory without a friend)

These factors apply to observables that are actually measured. A setting on
which a party outputs a constant needs no qubit and is taken to be noise-free;
the search below allows such settings. The self-test checks the factors
against a density-matrix simulation of the circuit.

Relaxed LF bounds
-----------------
With readout error the superobserver's setting-1 outcome a is no longer
identical to the friend's outcome c. If P(a != c | x = 1, y) <= eps_A and
P(b != d | y = 1, x) <= eps_B, every LF inequality loosens. The exact relaxed
bound is the maximum of the functional over the relaxed LF polytope, a linear
program in the joint distributions P(a, b, c, d | x, y) (relaxed_lf_bound).
For the three tests it evaluates to

    one-friend Semi-Brukner   2 + 4 eps_A
    Brukner                   2 + 4 min(eps_A, eps_B)
    Genuine LF                6 + 4 (eps_A + eps_B)

(checked on a grid of eps_A, eps_B up to 1/4, with dual certificates).

The first is the bound of Moreno et al., Quantum 6, 785 (2022), used on
hardware by Laux and Cavalcanti, arXiv:2609.12527.  An analytic upper bound
for any inequality is  B + 2 eps_A kappa_A + 2 eps_B kappa_B,  with kappa_A
(kappa_B) the sum of the absolute coefficients of all terms that contain
Alice's (Bob's) setting 1 (relaxation); it is tight for the one-friend test
and too large by a factor of two in eps for the two-friend tests.

What is exact and what is numerical
-----------------------------------
* One friend: S_max = 2 mu eta_0 sqrt(eta_1^2 + eta_2^2 f^2) whenever this
  exceeds the LF bound 2. The underlying sum-of-squares identity holds for
  involutions in any dimension (verified here on random matrices), so larger
  systems with the same visibility factors do no better.
* Two friends: seesaw search over two-qubit strategies. Every number is the
  value of an explicit strategy, i.e. a lower bound on the quantum maximum, so
  the critical noise levels printed for the two-friend tests are sufficient
  conditions: the true tolerances can only be larger.

Not modelled: energy relaxation and other non-unital noise, coherent gate
errors, crosstalk, single-qubit gate errors.

Usage
-----
    python3 lf_noise_budget.py                    # self-test, tables, calibration
    python3 lf_noise_budget.py --plot --save-json lf_noise_budget.json
    python3 lf_noise_budget.py --r2q 0.004 --eps 0.01 --f 0.98   # one device
    python3 lf_noise_budget.py --r2q 0.003 --eps 0.01 --friend-gates 10 --record-gates 1
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog, minimize_scalar

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P_PLUS = np.array([[1, 0], [0, 0]], dtype=complex)
P_MINUS = np.array([[0, 0], [0, 1]], dtype=complex)

# Coefficient tables: row = Alice's setting, column = Bob's setting,
# index 0 = identity (marginal terms), index 1 = friend's outcome.
# Same functionals as Eqs. (5), (6) and (9) of the manuscript.
TESTS: Dict[str, dict] = {
    "Semi-Brukner (1 friend)": {
        "coeffs": [[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0], [0, 0, -1, -1]],
        "bound": 2.0,
        "friends": (True, False),
    },
    "Brukner (2 friends)": {
        "coeffs": [[0, 0, 0, 0], [0, -1, 1, 0], [0, -1, -1, 0], [0, 0, 0, 0]],
        "bound": 2.0,
        "friends": (True, True),
    },
    "Genuine LF (2 friends)": {
        "coeffs": [[0, -1, -1, 0], [-1, -1, -2, 0], [-1, -2, 2, -1], [0, 0, -1, -1]],
        "bound": 6.0,
        "friends": (True, True),
    },
}

def _genuine_lf_seed(f: float) -> tuple:
    """Strategy from the proof of Proposition 1 (value 6 + f^4/(1+f^2) without other noise).

    For small f the violation is small and its basin of attraction is narrow, so
    the search is also started from this strategy.
    """
    cos_t = 1.0 / (1.0 + f * f)
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2.0)
    return (phi_plus, {2: -Z, 3: sin_t * X + cos_t * Z}, {2: -Z, 3: -sin_t * X + cos_t * Z})


TESTS["Genuine LF (2 friends)"]["seed"] = _genuine_lf_seed

# Published hardware run used for the calibration example:
# Laux and Cavalcanti, arXiv:2609.12527, Tables I and II, "Reflex" agent on
# ibm_marrakesh: S_LF = 0.465 +/- 0.009 above the bound 2, eps = 0.0175 +/- 0.0021.
PUBLISHED_RUN = {
    "label": "one-friend run of arXiv:2609.12527 (Reflex agent, ibm_marrakesh)",
    "value": 2.0 + 0.465,
    "value_err": 0.009,
    "eps": 0.0175,
    "eps_err": 0.0021,
}


# ---------------------------------------------------------------------------
# 1. Noise model
# ---------------------------------------------------------------------------

def visibilities(lam: float, eps: float, friend: bool, record_gates: int = 1,
                 reversal_gates: int = 2) -> Tuple[float, float, float, float]:
    """Visibility factors (identity, setting 1, setting 2, setting 3) of one laboratory."""
    if friend:
        eta1 = (1.0 - lam) ** record_gates * (1.0 - 2.0 * eps)
        eta2 = (1.0 - lam) ** reversal_gates * (1.0 - 2.0 * eps)
        return (1.0, eta1, eta2, eta2)
    eta0 = 1.0 - 2.0 * eps
    return (1.0, eta0, eta0, eta0)


def relaxation(coeffs: Sequence[Sequence[float]], eps_a: float, eps_b: float,
               friends: Tuple[bool, bool]) -> float:
    """Analytic upper bound on the loosening of an LF bound (coefficient sums).

    Valid for every LF inequality; tight for the one-friend test, not for the
    two-friend ones (see relaxed_lf_bound for the exact value).
    """
    kappa_a = sum(abs(coeffs[1][y]) for y in range(4))
    kappa_b = sum(abs(coeffs[x][1]) for x in range(4))
    return 2.0 * eps_a * kappa_a * friends[0] + 2.0 * eps_b * kappa_b * friends[1]


_SIGNS = (1, -1)
_OUTCOMES = list(itertools.product(_SIGNS, repeat=4))            # (a, b, c, d)
_SETTINGS = [(x, y) for x in (1, 2, 3) for y in (1, 2, 3)]


def _lp_index(x: int, y: int, outcome: Tuple[int, int, int, int]) -> int:
    return _SETTINGS.index((x, y)) * 16 + _OUTCOMES.index(outcome)


def _relaxed_lf_program(coeffs: Sequence[Sequence[float]], eps_a: float, eps_b: float,
                        friends: Tuple[bool, bool]):
    """Linear program whose maximum is the relaxed LF bound.

    Variables: P(a, b, c, d | x, y) >= 0.  Constraints: normalisation; Local
    Agency, i.e. P(a, c, d | x, y) independent of y and P(b, c, d | x, y)
    independent of x; and P(a != c | x = 1, y) <= eps_A for every y (if Alice
    has a friend), P(b != d | y = 1, x) <= eps_B for every x (if Bob has one).
    """
    n = 9 * 16
    objective = np.zeros(n)
    for x, y in _SETTINGS:
        for outcome in _OUTCOMES:
            a, b = outcome[0], outcome[1]
            value = coeffs[x][y] * a * b
            if y == 1:
                value += coeffs[x][0] * a          # marginal of A_x (independent of y)
            if x == 1:
                value += coeffs[0][y] * b          # marginal of B_y (independent of x)
            objective[_lp_index(x, y, outcome)] = value
    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    for x, y in _SETTINGS:
        row = np.zeros(n)
        for outcome in _OUTCOMES:
            row[_lp_index(x, y, outcome)] = 1.0
        a_eq.append(row)
        b_eq.append(1.0)
    for x, y in _SETTINGS:
        if y != 1:
            for a, c, d in itertools.product(_SIGNS, repeat=3):
                row = np.zeros(n)
                for b in _SIGNS:
                    row[_lp_index(x, y, (a, b, c, d))] += 1.0
                    row[_lp_index(x, 1, (a, b, c, d))] -= 1.0
                a_eq.append(row)
                b_eq.append(0.0)
        if x != 1:
            for b, c, d in itertools.product(_SIGNS, repeat=3):
                row = np.zeros(n)
                for a in _SIGNS:
                    row[_lp_index(x, y, (a, b, c, d))] += 1.0
                    row[_lp_index(1, y, (a, b, c, d))] -= 1.0
                a_eq.append(row)
                b_eq.append(0.0)
    if friends[0]:
        for y in (1, 2, 3):
            row = np.zeros(n)
            for outcome in _OUTCOMES:
                if outcome[0] != outcome[2]:
                    row[_lp_index(1, y, outcome)] = 1.0
            a_ub.append(row)
            b_ub.append(eps_a)
    if friends[1]:
        for x in (1, 2, 3):
            row = np.zeros(n)
            for outcome in _OUTCOMES:
                if outcome[1] != outcome[3]:
                    row[_lp_index(x, 1, outcome)] = 1.0
            a_ub.append(row)
            b_ub.append(eps_b)
    return objective, np.array(a_eq), np.array(b_eq), np.array(a_ub), np.array(b_ub)


@functools.lru_cache(maxsize=None)
def _relaxed_lf_bound_cached(coeff_key: tuple, eps_a: float, eps_b: float,
                             friends: Tuple[bool, bool]) -> Tuple[float, float]:
    objective, a_eq, b_eq, a_ub, b_ub = _relaxed_lf_program(coeff_key, eps_a, eps_b, friends)
    res = linprog(-objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"relaxed LF linear program failed: {res.message}")
    # Dual certificate: any dual-feasible point bounds the maximum from above,
    # independently of the solver's primal answer.
    u, v = res.eqlin.marginals, res.ineqlin.marginals
    reduced = -objective - a_eq.T @ u - a_ub.T @ v
    certified = -(b_eq @ u + b_ub @ v)
    if reduced.min() < -1e-7 or v.max() > 1e-7:
        raise RuntimeError("relaxed LF linear program returned an infeasible dual point")
    return float(-res.fun), float(certified)


def relaxed_lf_bound(coeffs: Sequence[Sequence[float]], eps_a: float, eps_b: float,
                     friends: Tuple[bool, bool]) -> float:
    """Exact LF bound when P(a != c | x = 1) <= eps_a and P(b != d | y = 1) <= eps_b."""
    key = tuple(tuple(float(v) for v in row) for row in coeffs)
    primal, certified = _relaxed_lf_bound_cached(key, float(eps_a), float(eps_b), tuple(friends))
    assert abs(primal - certified) < 1e-7, (primal, certified)
    return primal


def one_friend_closed_form(mu: float, lam: float, eps: float, f: float,
                           record_gates: int = 1, reversal_gates: int = 2) -> float:
    """Exact maximum of the one-friend Semi-Brukner functional in the noise model."""
    if reversal_gates < record_gates:
        raise ValueError(f"require reversal_gates >= record_gates, got {reversal_gates} < {record_gates}")
    _, eta1, eta2, _ = visibilities(lam, eps, True, record_gates=record_gates,
                                    reversal_gates=reversal_gates)
    eta0 = 1.0 - 2.0 * eps
    return 2.0 * mu * eta0 * math.sqrt(eta1 ** 2 + eta2 ** 2 * f ** 2)


def max_friend_size(lam: float, eps: float, f: float, record: str = "last",
                    mu: Optional[float] = None) -> int:
    """Largest fan-out size G with a violation of the relaxed one-friend inequality.

    The friend register has G qubits written by G CNOTs (g_rev = 2 G). Reading
    the last copy gives g_rec = G; reading the first copy gives g_rec = 1.
    """
    if record not in ("last", "first"):
        raise ValueError(f"record must be 'last' or 'first', got {record!r}")
    if lam <= 0.0:
        raise ValueError("lam must be positive to bound the friend size")
    mu_val = (1.0 - lam) if mu is None else mu
    bound = 2.0 + 4.0 * eps
    g = 0
    while True:
        g_next = g + 1
        g_rec = g_next if record == "last" else 1
        if one_friend_closed_form(mu_val, lam, eps, f, record_gates=g_rec,
                                  reversal_gates=2 * g_next) > bound:
            g = g_next
        else:
            return g


# ---------------------------------------------------------------------------
# 2. Seesaw over two-qubit strategies
# ---------------------------------------------------------------------------

def dephase(obs: np.ndarray, f: float) -> np.ndarray:
    """Heisenberg-picture action of a record leak with coherence retention f."""
    return (P_PLUS @ obs @ P_PLUS + P_MINUS @ obs @ P_MINUS
            + f * (P_PLUS @ obs @ P_MINUS + P_MINUS @ obs @ P_PLUS))


def matrix_sign(herm: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((herm + herm.conj().T) / 2.0)
    return (v * np.where(w >= 0, 1.0, -1.0)) @ v.conj().T


def noisy_observable(obs: np.ndarray, eta: float, friend: bool, f: float) -> np.ndarray:
    """Image of a chosen +-1 observable under the laboratory's noise.

    The traceless part (a measured Pauli direction) is dephased and multiplied
    by the visibility eta; the trace part (a constant output) is untouched.
    The map is linear and self-adjoint, so it also pulls gradients back.
    """
    trace_part = np.real(np.trace(obs)) / 2.0
    traceless = obs - trace_part * I2
    return trace_part * I2 + eta * (dephase(traceless, f) if friend else traceless)


def lab_observables(obs: Dict[int, np.ndarray], eta: Sequence[float], friend: bool,
                    f: float) -> List[np.ndarray]:
    """Noisy observables of one laboratory; index 0 = identity, 1 = friend's outcome."""
    if friend:
        return [I2, eta[1] * Z] + [noisy_observable(obs[k], eta[k], True, f) for k in (2, 3)]
    return [I2] + [noisy_observable(obs[k], eta[k], False, 1.0) for k in (1, 2, 3)]


def strategy_value(coeffs: np.ndarray, noisy_a: List[np.ndarray], noisy_b: List[np.ndarray],
                   psi: np.ndarray, mu: float) -> float:
    """Value of the functional on mu |psi><psi| + (1 - mu) 1/4."""
    bell = sum(coeffs[x, y] * np.kron(noisy_a[x], noisy_b[y])
               for x in range(4) for y in range(4) if coeffs[x, y])
    white = np.real(np.trace(bell)) / 4.0
    return float(mu * np.real(psi.conj() @ bell @ psi) + (1.0 - mu) * white)


def seesaw(coeffs: np.ndarray, friends: Tuple[bool, bool], mu: float,
           eta_a: Sequence[float], eta_b: Sequence[float], f_a: float, f_b: float,
           rng: np.random.Generator, restarts: int = 12, warm: Optional[tuple] = None,
           seeds: Sequence[tuple] = (), max_iter: int = 400,
           tol: float = 1e-12) -> Tuple[float, tuple]:
    """Maximize the noisy functional over two-qubit states and the free +-1 observables."""
    free_a = (2, 3) if friends[0] else (1, 2, 3)
    free_b = (2, 3) if friends[1] else (1, 2, 3)
    eye4 = np.eye(4)

    def random_obs() -> np.ndarray:
        if rng.random() < 0.25:         # a constant output
            return I2 * (1.0 if rng.random() < 0.5 else -1.0)
        h = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        h = h + h.conj().T
        return matrix_sign(h - np.trace(h) / 2.0 * I2)

    starts: List[Optional[tuple]] = (([warm] if warm is not None else []) + list(seeds)
                                     + [None] * restarts)
    best_val, best = -math.inf, None
    for start in starts:
        if start is None:
            obs_a = {x: random_obs() for x in free_a}
            obs_b = {y: random_obs() for y in free_b}
        else:
            obs_a = {x: start[1][x].copy() for x in free_a}
            obs_b = {y: start[2][y].copy() for y in free_b}
        prev, psi = -math.inf, None
        for iteration in range(max_iter):
            noisy_a = lab_observables(obs_a, eta_a, friends[0], f_a)
            noisy_b = lab_observables(obs_b, eta_b, friends[1], f_b)
            bell = sum(coeffs[x, y] * np.kron(noisy_a[x], noisy_b[y])
                       for x in range(4) for y in range(4) if coeffs[x, y])
            w, v = np.linalg.eigh(bell)
            psi = v[:, -1]
            val = float(mu * w[-1] + (1.0 - mu) * np.real(np.trace(bell)) / 4.0)
            if abs(val - prev) < tol and iteration > 5:
                break
            prev = val
            rho = (mu * np.outer(psi, psi.conj()) + (1.0 - mu) * eye4 / 4.0).reshape(2, 2, 2, 2)
            for x in free_a:
                partner = sum(coeffs[x, y] * noisy_b[y] for y in range(4))
                grad = np.einsum("abcd,db->ac", rho, partner)
                obs_a[x] = matrix_sign(noisy_observable(grad, eta_a[x], friends[0], f_a))
            noisy_a = lab_observables(obs_a, eta_a, friends[0], f_a)
            for y in free_b:
                partner = sum(coeffs[x, y] * noisy_a[x] for x in range(4))
                grad = np.einsum("abcd,ca->bd", rho, partner)
                obs_b[y] = matrix_sign(noisy_observable(grad, eta_b[y], friends[1], f_b))
        if prev > best_val:
            best_val = prev
            best = (psi.copy(), {k: m.copy() for k, m in obs_a.items()},
                    {k: m.copy() for k, m in obs_b.items()})
    return best_val, best


def noisy_optimum(name: str, lam: float, eps: float, f: float, rng: np.random.Generator,
                  mu: Optional[float] = None, restarts: int = 12,
                  warm: Optional[tuple] = None, record_gates: int = 1,
                  reversal_gates: int = 2) -> dict:
    """Best value found for one test at one noise point, with its relaxed LF bound."""
    test = TESTS[name]
    friends = test["friends"]
    mu = (1.0 - lam) if mu is None else mu
    eta_a = visibilities(lam, eps, friends[0], record_gates=record_gates,
                         reversal_gates=reversal_gates)
    eta_b = visibilities(lam, eps, friends[1], record_gates=record_gates,
                         reversal_gates=reversal_gates)
    f_a = f if friends[0] else 1.0
    f_b = f if friends[1] else 1.0
    seeds = [test["seed"](f)] if "seed" in test else []
    value, strategy = seesaw(np.array(test["coeffs"], dtype=float), friends, mu, eta_a, eta_b,
                             f_a, f_b, rng, restarts=restarts, warm=warm, seeds=seeds)
    bound = relaxed_lf_bound(test["coeffs"], eps, eps, friends)
    return {"value": value, "bound": bound, "margin": value - bound, "strategy": strategy,
            "mu": mu, "eta_a": eta_a, "eta_b": eta_b, "f_a": f_a, "f_b": f_b}


# ---------------------------------------------------------------------------
# 3. Behaviours: visibility factors versus explicit circuit simulation
# ---------------------------------------------------------------------------

def behaviour_effective(strategy: tuple, friends: Tuple[bool, bool], mu: float,
                        eta_a: Sequence[float], eta_b: Sequence[float],
                        f_a: float, f_b: float) -> np.ndarray:
    """4x4 table of <A_x B_y> (index 0 = identity) from the visibility factors."""
    psi, obs_a, obs_b = strategy
    noisy_a = lab_observables(obs_a, eta_a, friends[0], f_a)
    noisy_b = lab_observables(obs_b, eta_b, friends[1], f_b)
    table = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            unit = np.zeros((4, 4))
            unit[x, y] = 1.0
            table[x, y] = strategy_value(unit, noisy_a, noisy_b, psi, mu)
    return table


def _lab_channel(rho: np.ndarray, lab: int, setting: int, friend: bool,
                 lam: float, f: float) -> np.ndarray:
    """Apply one laboratory's gates and noise to a state on (S_A F_A S_B F_B)."""
    if not friend:
        return rho
    cnot = np.kron(P_PLUS, I2) + np.kron(P_MINUS, X)
    z_friend = np.kron(I2, Z)
    eye4 = np.eye(4, dtype=complex)

    def embed(op: np.ndarray) -> np.ndarray:
        return np.kron(op, eye4) if lab == 0 else np.kron(eye4, op)

    def depolarize(state: np.ndarray) -> np.ndarray:
        t = state.reshape(4, 4, 4, 4)
        if lab == 0:
            mixed = np.kron(eye4 / 4.0, np.einsum("abac->bc", t))
        else:
            mixed = np.kron(np.einsum("abcb->ac", t), eye4 / 4.0)
        return (1.0 - lam) * state + lam * mixed

    gate = embed(cnot)
    rho = depolarize(gate @ rho @ gate.conj().T)
    zf = embed(z_friend)
    rho = 0.5 * (1.0 + f) * rho + 0.5 * (1.0 - f) * (zf @ rho @ zf)
    if setting >= 2:
        rho = depolarize(gate @ rho @ gate.conj().T)
    return rho


def behaviour_circuit(strategy: tuple, friends: Tuple[bool, bool], mu: float, lam: float,
                      eps: float, f_a: float, f_b: float) -> np.ndarray:
    """Same table as behaviour_effective, from a density-matrix simulation of the circuit."""
    psi, obs_a, obs_b = strategy
    pair = mu * np.outer(psi, psi.conj()) + (1.0 - mu) * np.eye(4) / 4.0
    zero = np.outer([1, 0], [1, 0]).astype(complex)
    # (S_A S_B F_A F_B) -> (S_A F_A S_B F_B)
    full = np.kron(pair, np.kron(zero, zero)).reshape([2] * 8)
    full = full.transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)
    readout = 1.0 - 2.0 * eps

    def measured(lab: int, setting: int) -> np.ndarray:
        friend = friends[lab]
        obs = obs_a if lab == 0 else obs_b
        if setting == 0:
            return np.eye(4, dtype=complex)
        if friend and setting == 1:
            return readout * np.kron(I2, Z)
        trace_part = np.real(np.trace(obs[setting])) / 2.0
        traceless = obs[setting] - trace_part * I2
        return trace_part * np.eye(4, dtype=complex) + readout * np.kron(traceless, I2)

    table = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            rho = _lab_channel(full, 0, x, friends[0], lam, f_a)
            rho = _lab_channel(rho, 1, y, friends[1], lam, f_b)
            op = np.kron(measured(0, x), measured(1, y))
            table[x, y] = float(np.real(np.trace(rho @ op)))
    return table


_PAULIS = (I2, X, Y, Z)


def _single_qubit_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    factors = [op if idx == qubit else I2 for idx in range(n_qubits)]
    out = factors[0]
    for mat in factors[1:]:
        out = np.kron(out, mat)
    return out


def _cnot_op(control: int, target: int, n_qubits: int) -> np.ndarray:
    return (_single_qubit_op(P_PLUS, control, n_qubits)
            + _single_qubit_op(P_MINUS, control, n_qubits) @ _single_qubit_op(X, target, n_qubits))


def _depolarize_two_qubits(rho: np.ndarray, q0: int, q1: int, lam: float,
                           n_qubits: int) -> np.ndarray:
    if lam == 0.0:
        return rho
    twirled = np.zeros_like(rho)
    for p0 in _PAULIS:
        op0 = _single_qubit_op(p0, q0, n_qubits)
        for p1 in _PAULIS:
            op = op0 @ _single_qubit_op(p1, q1, n_qubits)
            twirled += op @ rho @ op.conj().T
    return (1.0 - lam) * rho + (lam / 16.0) * twirled


def behaviour_fanout_one_friend(strategy: tuple, mu: float, lam: float, eps: float,
                                f: float, g_friends: int, record_copy: int) -> np.ndarray:
    """One-friend circuit simulation with a G-qubit fan-out friend register.

    Alice's system S_A (qubit 0) writes G record qubits F_1..F_G (qubits 1..G) via
    CNOT(S_A -> F_1), ..., CNOT(S_A -> F_G), each followed by two-qubit depolarizing
    noise of strength lam. After a dephasing leak with coherence retention f on the
    record, setting 1 reads F_{record_copy}; settings 2 and 3 reverse the G CNOTs in
    reverse order (with the same depolarizing noise) and measure S_A. Bob (qubit G+1)
    has no friend.
    """
    if not (1 <= record_copy <= g_friends):
        raise ValueError(f"require 1 <= record_copy <= g_friends, got {record_copy} vs {g_friends}")
    psi, obs_a, obs_b = strategy
    n_qubits = g_friends + 2
    dim = 2 ** n_qubits
    eye = np.eye(dim, dtype=complex)
    pair = mu * np.outer(psi, psi.conj()) + (1.0 - mu) * np.eye(4, dtype=complex) / 4.0
    zero_friends = np.zeros((2 ** g_friends, 2 ** g_friends), dtype=complex)
    zero_friends[0, 0] = 1.0
    # Reorder (S_A, S_B, F_1..F_G) -> (S_A, F_1..F_G, S_B)
    raw = np.kron(pair, zero_friends).reshape([2] * (2 * n_qubits))
    perm = [0] + list(range(2, n_qubits)) + [1]
    rho_init = raw.transpose(perm + [p + n_qubits for p in perm]).reshape(dim, dim)

    cnots = {j: _cnot_op(0, j, n_qubits) for j in range(1, g_friends + 1)}
    z_leak = _single_qubit_op(Z, 1, n_qubits)
    readout = 1.0 - 2.0 * eps

    def alice_op(setting: int) -> np.ndarray:
        if setting == 0:
            return eye
        if setting == 1:
            return readout * _single_qubit_op(Z, record_copy, n_qubits)
        trace_part = np.real(np.trace(obs_a[setting])) / 2.0
        traceless = obs_a[setting] - trace_part * I2
        return trace_part * eye + readout * _single_qubit_op(traceless, 0, n_qubits)

    def bob_op(setting: int) -> np.ndarray:
        if setting == 0:
            return eye
        trace_part = np.real(np.trace(obs_b[setting])) / 2.0
        traceless = obs_b[setting] - trace_part * I2
        return trace_part * eye + readout * _single_qubit_op(traceless, n_qubits - 1, n_qubits)

    table = np.zeros((4, 4))
    for x in range(4):
        rho = rho_init.copy()
        for j in range(1, g_friends + 1):
            gate = cnots[j]
            rho = _depolarize_two_qubits(gate @ rho @ gate.conj().T, 0, j, lam, n_qubits)
        rho = 0.5 * (1.0 + f) * rho + 0.5 * (1.0 - f) * (z_leak @ rho @ z_leak)
        if x >= 2:
            for j in range(g_friends, 0, -1):
                gate = cnots[j]
                rho = _depolarize_two_qubits(gate @ rho @ gate.conj().T, 0, j, lam, n_qubits)
        op_a = alice_op(x)
        for y in range(4):
            table[x, y] = float(np.real(np.trace(rho @ (op_a @ bob_op(y)))))
    return table


def shots_for_significance(coeffs: Sequence[Sequence[float]], table: np.ndarray,
                           margin: float, sigmas: float = 5.0) -> float:
    """Shot-noise estimate of the total number of shots for a `sigmas`-sigma violation.

    Each marginal term is split evenly over the setting pairs that enter the
    functional and contain its setting; shots are allocated across setting
    pairs in proportion to the per-shot standard deviation (optimal for a
    fixed total).  Gaussian estimate, not a hypothesis test.
    """
    if margin <= 0:
        return math.inf
    pairs = [(x, y) for x in (1, 2, 3) for y in (1, 2, 3) if coeffs[x][y]]
    total_sd, total_mean = 0.0, 0.0
    for x, y in pairs:
        n_x = sum(1 for (p, _) in pairs if p == x)
        n_y = sum(1 for (_, q) in pairs if q == y)
        j, alpha, beta = coeffs[x][y], coeffs[x][0] / n_x, coeffs[0][y] / n_y
        mean_a, mean_b, corr = table[x, 0], table[0, y], table[x, y]
        mean = j * corr + alpha * mean_a + beta * mean_b
        second = (j * j + alpha * alpha + beta * beta + 2 * j * alpha * mean_b
                  + 2 * j * beta * mean_a + 2 * alpha * beta * corr)
        total_sd += math.sqrt(max(0.0, second - mean * mean))
        total_mean += mean
    assert abs(total_mean - float(np.sum(np.array(coeffs) * table))) < 1e-9
    return (sigmas * total_sd / margin) ** 2


# ---------------------------------------------------------------------------
# 4. Critical noise levels
# ---------------------------------------------------------------------------

def _bisect(func, lo: float, hi: float, tol: float) -> float:
    """Root of a function that is positive at lo and non-positive at hi."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if func(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_lambda(name: str, eps: float, f: float, rng: np.random.Generator,
                    step: float = 0.01, tol: float = 2e-4, restarts: int = 6,
                    record_gates: int = 1, reversal_gates: int = 2) -> float:
    """Largest depolarizing parameter per entangling gate at which the test is still violated.

    Returns 0.0 if the relaxed inequality is not violated even with perfect gates.
    """
    if name.startswith("Semi-Brukner"):
        def margin(lam: float) -> float:
            return one_friend_closed_form(1.0 - lam, lam, eps, f,
                                          record_gates=record_gates,
                                          reversal_gates=reversal_gates) - (2.0 + 4.0 * eps)
        if margin(0.0) <= 0:
            return 0.0
        return _bisect(margin, 0.0, 0.5, tol / 10)

    state = {"warm": None}

    def margin(lam: float, extra: int = 0) -> float:
        res = noisy_optimum(name, lam, eps, f, rng, restarts=restarts + extra,
                            warm=state["warm"], record_gates=record_gates,
                            reversal_gates=reversal_gates)
        state["warm"] = res["strategy"]
        return res["margin"]

    if margin(0.0, extra=10) <= 1e-9:
        return 0.0
    lo, hi = 0.0, step
    while margin(hi) > 1e-9:
        lo, hi = hi, hi + step
        if hi > 0.5:
            return 0.5
    lam_star = _bisect(lambda lam: margin(lam) - 1e-9, lo, hi, tol)
    # A warm-started search can stay on a losing branch: confirm with fresh starts.
    for _ in range(3):
        if margin(lam_star + 2 * tol, extra=14) <= 1e-9:
            break
        lo, hi = lam_star + 2 * tol, lam_star + 2 * tol + step
        while margin(hi, extra=6) > 1e-9:
            lo, hi = hi, hi + step
        lam_star = _bisect(lambda lam: margin(lam) - 1e-9, lo, hi, tol)
    return lam_star


def critical_coherence(name: str, lam: float, eps: float, rng: np.random.Generator,
                       tol: float = 5e-4, restarts: int = 10) -> float:
    """Smallest coherence retention f at which the test is violated (inf if never)."""
    if name.startswith("Semi-Brukner"):
        def margin(f: float) -> float:
            return one_friend_closed_form(1.0 - lam, lam, eps, f) - (2.0 + 4.0 * eps)
        if margin(1.0) <= 0:
            return math.inf
        if margin(0.0) > 0:
            return 0.0
        return 1.0 - _bisect(lambda g: margin(1.0 - g), 0.0, 1.0, tol / 10)

    state = {"warm": None}

    def margin(f: float, extra: int = 0) -> float:
        res = noisy_optimum(name, lam, eps, f, rng, restarts=restarts + extra, warm=state["warm"])
        state["warm"] = res["strategy"]
        return res["margin"]

    if margin(1.0, extra=10) <= 1e-9:
        return math.inf
    gap = _bisect(lambda g: margin(1.0 - g) - 1e-9, 0.0, 1.0, tol)
    return 1.0 - gap


def critical_eps(name: str, f: float, rng: np.random.Generator, tol: float = 1e-4,
                 restarts: int = 10) -> float:
    """Largest readout error at which the test is violated with perfect gates (lambda = 0)."""
    state = {"warm": None}

    def margin(eps: float) -> float:
        if name.startswith("Semi-Brukner"):
            return one_friend_closed_form(1.0, 0.0, eps, f) - (2.0 + 4.0 * eps)
        res = noisy_optimum(name, 0.0, eps, f, rng, restarts=restarts, warm=state["warm"])
        state["warm"] = res["strategy"]
        return res["margin"] - 1e-9

    if margin(0.0) <= 0:
        return 0.0
    return _bisect(margin, 0.0, 0.25, tol)


def monotone_envelope(values: Sequence[float], increasing: bool) -> List[float]:
    """Tighten a list of certified thresholds using monotonicity in the noise.

    A violation found at some noise level is also available at any lower noise
    level (noise can be added by post-processing), so a tolerance found at a
    noisier grid point is valid at every less noisy one.
    """
    out = list(values)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1]) if increasing else max(out[i], out[i + 1])
    return out


def calibrate_lambda(value: float, eps: float, f: float = 1.0,
                     record_gates: int = 1, reversal_gates: int = 2) -> float:
    """Depolarizing parameter at which the one-friend model reproduces an observed value."""
    def excess(lam: float) -> float:
        return one_friend_closed_form(1.0 - lam, lam, eps, f,
                                      record_gates=record_gates,
                                      reversal_gates=reversal_gates) - value
    if excess(0.0) <= 0:
        return 0.0
    return _bisect(excess, 0.0, 0.5, 1e-7)


# ---------------------------------------------------------------------------
# 5. Self-test
# ---------------------------------------------------------------------------

def _random_involution(dim: int, rng: np.random.Generator) -> np.ndarray:
    h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    w, v = np.linalg.eigh(h + h.conj().T)
    signs = np.where(rng.random(dim) < 0.5, 1.0, -1.0)
    return (v * signs) @ v.conj().T


def sos_identity_residual(dim_a: int, dim_b: int, eta1: float, eta2: float, f: float,
                          rng: np.random.Generator) -> float:
    """Residual of 4(eta1^2 + eta2^2 f^2) 1 - K^2 = S1* S1 + S2* S2 on random involutions."""
    a1, n = _random_involution(dim_a, rng), _random_involution(dim_a, rng)
    b2, b3 = _random_involution(dim_b, rng), _random_involution(dim_b, rng)
    eye_a, eye_b = np.eye(dim_a), np.eye(dim_b)
    n_even = 0.5 * (n + a1 @ n @ a1)
    n_odd = 0.5 * (n - a1 @ n @ a1)
    d_minus = np.kron(eye_a, b3 - b2)
    d_plus = np.kron(eye_a, b3 + b2)
    a1_full = np.kron(a1, eye_b)
    k = eta1 * a1_full @ d_minus - eta2 * np.kron(n_even + f * n_odd, eye_b) @ d_plus
    s1 = eta2 * f * d_minus + eta1 * a1_full @ np.kron(n_odd, eye_b) @ d_plus
    s2 = math.sqrt(eta1 ** 2 - (1.0 - f ** 2) * eta2 ** 2) * np.kron(n_even, eye_b) @ d_plus
    lhs = 4.0 * (eta1 ** 2 + eta2 ** 2 * f ** 2) * np.eye(dim_a * dim_b) - k @ k
    return float(np.max(np.abs(lhs - s1.conj().T @ s1 - s2.conj().T @ s2)))


def self_test(rng: np.random.Generator) -> dict:
    report = {}

    # (a) visibility factors reproduce the simulated circuit: optimal strategies and
    #     random ones, the latter with some settings replaced by constant outputs
    worst = 0.0
    for name, test in TESTS.items():
        friends = test["friends"]
        coeffs = np.array(test["coeffs"], dtype=float)
        for mu, lam, eps, f in ((0.97, 0.03, 0.015, 0.9), (1.0, 0.08, 0.0, 0.6), (0.9, 0.0, 0.04, 1.0)):
            res = noisy_optimum(name, lam, eps, f, rng, mu=mu, restarts=4)
            strategies = [res["strategy"]]
            for _ in range(3):
                psi = rng.normal(size=4) + 1j * rng.normal(size=4)
                psi /= np.linalg.norm(psi)
                obs = []
                for _lab in range(2):
                    lab_obs = {}
                    for k in (1, 2, 3):
                        h = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                        h = h + h.conj().T
                        lab_obs[k] = matrix_sign(h - np.trace(h) / 2.0 * I2)
                        if rng.random() < 0.3:
                            lab_obs[k] = I2 * (1.0 if rng.random() < 0.5 else -1.0)
                    obs.append(lab_obs)
                strategies.append((psi, obs[0], obs[1]))
            for strategy in strategies:
                eff = behaviour_effective(strategy, friends, mu, res["eta_a"], res["eta_b"],
                                          res["f_a"], res["f_b"])
                sim = behaviour_circuit(strategy, friends, mu, lam, eps, res["f_a"], res["f_b"])
                worst = max(worst, float(np.max(np.abs(eff - sim))))
            sim = behaviour_circuit(res["strategy"], friends, mu, lam, eps, res["f_a"], res["f_b"])
            worst = max(worst, abs(float(np.sum(coeffs * sim)) - res["value"]))
    report["circuit_vs_visibility_max_abs_diff"] = worst
    assert worst < 1e-10, worst

    # (b) one-friend closed form against the seesaw. Where the closed form is below the
    #     LF bound 2 the search must not exceed 2 (constant outputs reach 2 without noise).
    worst = 0.0
    for mu, lam, eps, f in ((1.0, 0.02, 0.01, 0.9), (0.97, 0.03, 0.02, 0.7), (1.0, 0.0, 0.0, 0.5),
                            (0.99, 0.01, 0.005, 0.3), (0.95, 0.05, 0.02, 0.7), (0.9, 0.1, 0.03, 1.0)):
        res = noisy_optimum("Semi-Brukner (1 friend)", lam, eps, f, rng, mu=mu, restarts=20)
        closed = one_friend_closed_form(mu, lam, eps, f)
        if closed > 2.0:
            worst = max(worst, abs(res["value"] - closed))
        else:
            assert res["value"] <= 2.0 + 1e-9, (res["value"], closed)
    report["one_friend_closed_form_max_abs_diff"] = worst
    assert worst < 1e-8, worst

    # (c) the sum-of-squares identity behind the closed form, in several dimensions
    worst = 0.0
    for dim_a, dim_b in ((2, 2), (3, 2), (4, 3), (5, 5)):
        for eta1, eta2, f in ((1.0, 1.0, 0.6), (0.95, 0.9, 0.8), (0.7, 0.7, 0.0), (0.9, 0.5, 1.0)):
            worst = max(worst, sos_identity_residual(dim_a, dim_b, eta1, eta2, f, rng))
    report["sos_identity_max_residual"] = worst
    assert worst < 1e-10, worst

    # (d) relaxed bounds: the linear program against its closed forms and against the
    #     analytic coefficient-sum bound, on a grid of readout errors
    closed = {
        "Semi-Brukner (1 friend)": lambda ea, eb: 2.0 + 4.0 * ea,
        "Brukner (2 friends)": lambda ea, eb: 2.0 + 4.0 * min(ea, eb),
        "Genuine LF (2 friends)": lambda ea, eb: 6.0 + 4.0 * (ea + eb),
    }
    grid = (0.0, 0.005, 0.0175, 0.05, 0.1, 0.25)
    worst, min_slack = 0.0, math.inf
    for name, test in TESTS.items():
        for ea in grid:
            for eb in grid:
                exact = relaxed_lf_bound(test["coeffs"], ea, eb, test["friends"])
                worst = max(worst, abs(exact - closed[name](ea, eb)))
                analytic = test["bound"] + relaxation(test["coeffs"], ea, eb, test["friends"])
                min_slack = min(min_slack, analytic - exact)
    report["relaxed_bound_lp_vs_closed_form_max_abs_diff"] = worst
    report["relaxed_bound_analytic_minus_lp_min"] = min_slack
    assert worst < 1e-8, worst
    assert min_slack > -1e-8, min_slack
    # Brukner needs only one of the two identifications
    brukner = TESTS["Brukner (2 friends)"]
    assert abs(relaxed_lf_bound(brukner["coeffs"], 1.0, 0.0, brukner["friends"]) - 2.0) < 1e-8
    assert abs(relaxed_lf_bound(brukner["coeffs"], 0.0, 1.0, brukner["friends"]) - 2.0) < 1e-8

    # (e) noiseless limits
    ideal = {name: noisy_optimum(name, 0.0, 0.0, 1.0, rng, restarts=30)["value"] for name in TESTS}
    assert abs(ideal["Semi-Brukner (1 friend)"] - 2 * math.sqrt(2)) < 1e-9
    assert abs(ideal["Brukner (2 friends)"] - 2 * math.sqrt(2)) < 1e-9
    assert abs(ideal["Genuine LF (2 friends)"] - 7.344912) < 1e-5
    report["noiseless_values"] = ideal

    # (f) multi-gate fan-out friend (G = 1, 2, 3, reading copy 1 and copy G):
    #     use an independent RNG so default downstream tables/curves keep identical seeds.
    fan_rng = np.random.default_rng(17)
    worst_fanout, worst_g1 = 0.0, 0.0
    for g_friends in (1, 2, 3):
        for record_copy in sorted({1, g_friends}):
            for mu, lam, eps, f in ((0.97, 0.03, 0.015, 0.9),
                                    (1.0, 0.08, 0.0, 0.6),
                                    (0.92, 0.04, 0.02, 1.0)):
                res = noisy_optimum("Semi-Brukner (1 friend)", lam, eps, f, fan_rng, mu=mu,
                                    restarts=4, record_gates=record_copy,
                                    reversal_gates=2 * g_friends)
                strategies = [res["strategy"]]
                for _ in range(2):
                    psi = fan_rng.normal(size=4) + 1j * fan_rng.normal(size=4)
                    psi /= np.linalg.norm(psi)
                    obs_pair = []
                    for _lab in range(2):
                        lab_obs = {}
                        for k in (1, 2, 3):
                            h = fan_rng.normal(size=(2, 2)) + 1j * fan_rng.normal(size=(2, 2))
                            h = h + h.conj().T
                            lab_obs[k] = matrix_sign(h - np.trace(h) / 2.0 * I2)
                            if fan_rng.random() < 0.3:
                                lab_obs[k] = I2 * (1.0 if fan_rng.random() < 0.5 else -1.0)
                        obs_pair.append(lab_obs)
                    strategies.append((psi, obs_pair[0], obs_pair[1]))
                for strategy in strategies:
                    eff = behaviour_effective(strategy, (True, False), mu, res["eta_a"],
                                              res["eta_b"], res["f_a"], res["f_b"])
                    sim = behaviour_fanout_one_friend(strategy, mu, lam, eps, f,
                                                      g_friends, record_copy)
                    worst_fanout = max(worst_fanout, float(np.max(np.abs(eff - sim))))
                    if g_friends == 1:
                        circ = behaviour_circuit(strategy, (True, False), mu, lam, eps, f, 1.0)
                        worst_g1 = max(worst_g1, float(np.max(np.abs(sim - circ))))
    report["fanout_vs_visibility_max_abs_diff"] = worst_fanout
    report["fanout_g1_vs_circuit_max_abs_diff"] = worst_g1
    assert worst_fanout < 1e-10, worst_fanout
    assert worst_g1 < 1e-10, worst_g1
    return report


# ---------------------------------------------------------------------------
# 6. Reporting
# ---------------------------------------------------------------------------

def device_report(lam: float, eps: float, f: float, mu: Optional[float],
                  rng: np.random.Generator, restarts: int = 30,
                  record_gates: int = 1, reversal_gates: int = 2) -> List[dict]:
    rows = []
    for name, test in TESTS.items():
        res = noisy_optimum(name, lam, eps, f, rng, mu=mu, restarts=restarts,
                            record_gates=record_gates, reversal_gates=reversal_gates)
        table = behaviour_effective(res["strategy"], test["friends"], res["mu"], res["eta_a"],
                                    res["eta_b"], res["f_a"], res["f_b"])
        rows.append({
            "test": name,
            "value": res["value"],
            "relaxed_bound": res["bound"],
            "margin": res["margin"],
            "shots_5_sigma": shots_for_significance(test["coeffs"], table, res["margin"]),
        })
    return rows


def fit_agents(path: str) -> dict:
    """Compare the gate-count visibility model against the four agents of arXiv:2609.12527.

    Skips cleanly while any agent's transpiled Heisenberg gate counts (record_gates,
    reversal_gates) are null. When populated:
      * Test 1 (prediction): calibrate lambda on Reflex alone and report predicted
        S_LF = S_SB - 2 and residuals (in standard errors) for the other three agents.
      * Test 2 (consistency): fit a single lambda across all four agents by weighted
        least squares and report lambda, chi-squared, and degrees of freedom.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    agents = raw["agents"] if isinstance(raw, dict) else raw
    missing = [a["name"] for a in agents
               if a.get("record_gates") is None or a.get("reversal_gates") is None]
    if missing:
        print(f"  Skipped ({path}): record_gates/reversal_gates are null for "
              f"{', '.join(missing)} (awaiting transpiled Heisenberg gate counts).")
        return {"status": "skipped", "missing_gate_counts": missing, "path": path}

    def pred_s_lf(lam: float, agent: dict) -> float:
        return one_friend_closed_form(
            1.0 - lam, lam, float(agent["eps"]), 1.0,
            record_gates=int(agent["record_gates"]),
            reversal_gates=int(agent["reversal_gates"]),
        ) - 2.0

    reflex = next((a for a in agents if a["name"].lower().startswith("reflex")), agents[0])
    lam_reflex = calibrate_lambda(
        2.0 + float(reflex["s_lf"]), float(reflex["eps"]), 1.0,
        record_gates=int(reflex["record_gates"]),
        reversal_gates=int(reflex["reversal_gates"]),
    )
    predictions = []
    print(f"  Test 1 (calibrated on {reflex['name']}: lambda_eff = {lam_reflex:.5f}):")
    print(f"    {'agent':<14} {'g_rec':>5} {'g_rev':>5} {'observed':>10} {'predicted':>10} {'residual':>10} {'z (sigma)':>10}")
    for agent in agents:
        obs = float(agent["s_lf"])
        err = float(agent["s_lf_err"])
        pred = pred_s_lf(lam_reflex, agent)
        res = obs - pred
        z = res / err
        predictions.append({
            "name": agent["name"],
            "record_gates": int(agent["record_gates"]),
            "reversal_gates": int(agent["reversal_gates"]),
            "observed": obs,
            "observed_err": err,
            "predicted": pred,
            "residual": res,
            "residual_sigma": z,
        })
        print(f"    {agent['name']:<14} {int(agent['record_gates']):5d} {int(agent['reversal_gates']):5d} "
              f"{obs:10.4f} {pred:10.4f} {res:+10.4f} {z:+10.2f}")

    def chi2_fn(lam: float) -> float:
        return sum(((float(a["s_lf"]) - pred_s_lf(lam, a)) / float(a["s_lf_err"])) ** 2
                   for a in agents)

    opt = minimize_scalar(chi2_fn, bounds=(0.0, 0.5), method="bounded", options={"xatol": 1e-10})
    lam_wls = float(opt.x)
    chi2_val = float(opt.fun)
    dof = max(0, len(agents) - 1)
    print(f"  Test 2 (weighted least squares across all {len(agents)} agents): "
          f"lambda = {lam_wls:.5f}, chi^2 = {chi2_val:.2f} for {dof} dof.")
    return {
        "status": "ok",
        "path": path,
        "reflex_calibration": {"agent": reflex["name"], "lambda_eff": lam_reflex,
                               "predictions": predictions},
        "wls_fit": {"lambda": lam_wls, "chi2": chi2_val, "dof": dof},
    }


def print_device_report(rows: List[dict]) -> None:
    print(f"  {'test':<26} {'S_max':>9} {'relaxed bound':>14} {'margin':>9} {'shots for 5 sigma':>18}")
    for row in rows:
        shots = "no violation" if math.isinf(row["shots_5_sigma"]) else f"{row['shots_5_sigma']:,.0f}"
        print(f"  {row['test']:<26} {row['value']:9.4f} {row['relaxed_bound']:14.4f} "
              f"{row['margin']:+9.4f} {shots:>18}")


def make_figure(curves: dict, calibration: dict, prefix: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARN] matplotlib is not installed; skipping figure generation.")
        return
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    styles = {
        "Semi-Brukner (1 friend)": ("#1f77b4", "Semi-Brukner (1 friend)"),
        "Brukner (2 friends)": ("#ff7f0e", "Brukner (2 friends)"),
        "Genuine LF (2 friends)": ("#d62728", "Genuine LF (2 friends)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    eps_grid = np.array(curves["eps_grid"])
    for name, (color, label) in styles.items():
        for f_key, ls in (("1.0", "-"), ("0.9", "--")):
            lam = np.array(curves["critical_lambda"][name][f_key])
            eps_max = curves["critical_eps"][name][f_key]
            keep = (lam > 0) & (eps_grid < eps_max)
            xs = np.append(eps_grid[keep], eps_max)
            ys = np.append(lam[keep], 0.0)
            ax.plot(100 * xs, 100 * 0.75 * ys, color=color, linestyle=ls, lw=2.0,
                    label=f"{label}, $f = {float(f_key):g}$")
    ax.plot([100 * calibration["eps"]], [100 * 0.75 * calibration["lambda_eff"]], marker="*",
            color="k", ms=13, linestyle="none", label="effective noise of a published\none-friend run (see text)")
    ax.set_xlabel(r"Readout assignment error $\varepsilon$ (%)")
    ax.set_ylabel(r"Critical two-qubit gate infidelity $r^\ast = 3\lambda^\ast/4$ (%)")
    ax.set_title("(a) Violated below each curve")
    ax.set_xlim(0, 100 * eps_grid[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.6, loc="upper right")

    ax = axes[1]
    r_grid = np.array(curves["r_grid"])
    for name, (color, label) in styles.items():
        for eps_key, ls in (("0.0", "-"), ("0.01", "--")):
            f_crit = np.array(curves["critical_f"][name][eps_key], dtype=float)
            r_end = 0.75 * curves["critical_lambda_f1"][name][eps_key]
            ok = np.isfinite(f_crit) & (r_grid < r_end)
            xs = np.append(r_grid[ok], r_end)
            ys = np.append(f_crit[ok], 1.0)
            ax.plot(100 * xs, ys, color=color, linestyle=ls, lw=2.0,
                    label=f"{label}, $\\varepsilon = {100 * float(eps_key):g}\\%$")
    ax.set_xlabel(r"Two-qubit gate infidelity $r = 3\lambda/4$ (%)")
    ax.set_ylabel(r"Minimum coherence retention $f_{\mathrm{crit}}$")
    ax.set_title("(b) Violated above each curve")
    ax.set_xlim(0, 100 * r_grid[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.6, loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{prefix}.pdf")
    fig.savefig(f"{prefix}.png", dpi=220)
    print(f"\n[OK] Saved figure: {prefix}.pdf and {prefix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise budget for circuit implementations of LF tests")
    parser.add_argument("--plot", action="store_true", help="Export lf_noise_budget.pdf/.png")
    parser.add_argument("--save-json", type=str, default="", help="Optional path for a JSON summary")
    parser.add_argument("--r2q", type=float, default=None,
                        help="Device lookup: average two-qubit gate infidelity (e.g. 0.004)")
    parser.add_argument("--eps", type=float, default=0.01, help="Device lookup: readout assignment error")
    parser.add_argument("--f", type=float, default=1.0, help="Device lookup: coherence retention of the record")
    parser.add_argument("--mu", type=float, default=None,
                        help="Device lookup: visibility of the entangled pair (default 1 - 4 r / 3)")
    parser.add_argument("--friend-gates", type=int, default=None,
                        help="Device lookup: entangling gates G of the friend's measurement "
                             "(sets reversal_gates = 2 G)")
    parser.add_argument("--record-gates", type=int, default=None,
                        help="Device lookup: entangling gates g_rec on the record readout "
                             "(default: G when --friend-gates is given, else 1)")
    parser.add_argument("--agents-json", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "manuscript", "data", "agents_laux2026.json"),
                        help="Path to multi-agent JSON table for Stage B comparison")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the random restarts")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.time()

    print("=" * 78)
    print("Noise budget for circuit implementations of Local Friendliness tests")
    print("=" * 78)

    if args.r2q is not None:
        if args.friend_gates is not None and args.friend_gates < 1:
            parser.error("--friend-gates must be at least 1")
        rev_gates = 2 * args.friend_gates if args.friend_gates is not None else 2
        default_rec = args.friend_gates if args.friend_gates is not None else 1
        rec_gates = default_rec if args.record_gates is None else args.record_gates
        if not (1 <= rec_gates <= rev_gates):
            parser.error(f"--record-gates must satisfy 1 <= k <= {rev_gates}")
        lam = 4.0 * args.r2q / 3.0
        print(f"\nDevice lookup: r_2q = {args.r2q:g} (lambda = {lam:.5f}), eps = {args.eps:g}, "
              f"f = {args.f:g}, mu = {'1 - lambda' if args.mu is None else args.mu}")
        if (rec_gates, rev_gates) != (1, 2):
            print(f"  Friend gate counts: g_rec = {rec_gates}, g_rev = {rev_gates}")
        print_device_report(device_report(lam, args.eps, args.f, args.mu, rng,
                                          record_gates=rec_gates, reversal_gates=rev_gates))
        if (rec_gates, rev_gates) != (1, 2):
            print("\n  Two-friend rows are model output assuming identical multi-gate friends in "
                  "both laboratories (explicit two-qubit strategies, lower bounds).")
        else:
            print("\n  Two-friend rows are values of explicit two-qubit strategies (lower bounds).")
        return

    print("\n[1] Self-test")
    report = self_test(rng)
    print(f"  circuit simulation vs visibility factors: max |diff| = "
          f"{report['circuit_vs_visibility_max_abs_diff']:.1e}")
    print(f"  fan-out simulation (G <= 3) vs visibilities: max |diff| = "
          f"{report['fanout_vs_visibility_max_abs_diff']:.1e} "
          f"(G = 1 vs circuit: {report['fanout_g1_vs_circuit_max_abs_diff']:.1e})")
    print(f"  one-friend closed form vs seesaw:         max |diff| = "
          f"{report['one_friend_closed_form_max_abs_diff']:.1e}")
    print(f"  sum-of-squares identity, random matrices: max residual = "
          f"{report['sos_identity_max_residual']:.1e}")
    print(f"  relaxed bounds, linear program vs closed forms: max |diff| = "
          f"{report['relaxed_bound_lp_vs_closed_form_max_abs_diff']:.1e}")
    print("    2 + 4 eps_A (Semi-Brukner, 1 friend), 2 + 4 min(eps_A, eps_B) (Brukner),"
          " 6 + 4 (eps_A + eps_B) (Genuine LF)")

    print("\n[2] Critical two-qubit gate infidelity r* = 3 lambda*/4 (pair prepared with one such gate)")
    eps_table = (0.0, 0.005, 0.01, 0.02, 0.03)
    f_table = (1.0, 0.95, 0.9)
    table: Dict[str, Dict[str, List[float]]] = {name: {} for name in TESTS}
    print(f"  {'f':>5} {'eps':>6} | " + " | ".join(f"{name:>24}" for name in TESTS))
    for f in f_table:
        for eps in eps_table:
            row = []
            for name in TESTS:
                lam_star = critical_lambda(name, eps, f, rng)
                table[name].setdefault(f"{f:g}", []).append(lam_star)
                row.append("        not violated" if lam_star == 0.0 else f"{100 * 0.75 * lam_star:19.2f}%")
            print(f"  {f:5.2f} {eps:6.3f} | " + " | ".join(f"{cell:>24}" for cell in row), flush=True)

    print("\n[3] Calibration example")
    run = PUBLISHED_RUN
    lam_eff = calibrate_lambda(run["value"], run["eps"])
    corners = [(run["value"] + sv * run["value_err"], run["eps"] + se * run["eps_err"])
               for sv in (-1, 1) for se in (-1, 1)]
    corner_fits = [(calibrate_lambda(value, eps), eps) for value, eps in corners]
    lam_lo, lam_hi = min(c[0] for c in corner_fits), max(c[0] for c in corner_fits)
    print(f"  {run['label']}:")
    print(f"  S = {run['value']:.3f} +/- {run['value_err']:.3f}, eps = {run['eps']:.4f} +/- {run['eps_err']:.4f}")
    print(f"  Reproduced by the one-friend model at f = 1 with lambda_eff = {lam_eff:.4f} "
          f"(r_eff = {100 * 0.75 * lam_eff:.2f}%; {100 * 0.75 * lam_lo:.2f}%-{100 * 0.75 * lam_hi:.2f}% "
          "over the quoted uncertainties).")
    print("  lambda_eff absorbs every loss of visibility that is not readout error; it is an")
    print("  equivalent noise level in this model, not a measured gate error.")
    print("  The three tests at that noise level:")
    calib_rows = device_report(lam_eff, run["eps"], 1.0, None, rng)
    print_device_report(calib_rows)
    calib_range = {}
    for name in TESTS:
        margins = [noisy_optimum(name, lam_val, eps_val, 1.0, rng, restarts=20)["margin"]
                   for lam_val, eps_val in corner_fits]
        calib_range[name] = [min(margins), max(margins)]
        print(f"    {name:<26} margin {min(margins):+.3f} to {max(margins):+.3f} over the quoted uncertainties")

    print("\n[4] Reference devices (f = 1 unless noted; shots are a shot-noise estimate)")
    devices = (
        ("r = 0.1%, eps = 0.5%", 0.001, 0.005, 1.0),
        ("r = 0.3%, eps = 1%", 0.003, 0.01, 1.0),
        ("r = 0.5%, eps = 1%", 0.005, 0.01, 1.0),
        ("r = 1%,   eps = 2%", 0.01, 0.02, 1.0),
        ("r = 0.3%, eps = 1%, f = 0.9", 0.003, 0.01, 0.9),
        # Google Quantum AI, "Willow Spec Sheet" (published 9 Dec 2024, Chip 1);
        # nominal published error rates, not fitted to an LF run:
        ("Willow spec sheet, nominal: r = 0.33%, eps = 0.77%", 0.0033, 0.0077, 1.0),
        ("Willow spec sheet, lower:   r = 0.15%, eps = 0.56%", 0.0015, 0.0056, 1.0),
        ("Willow spec sheet, higher:  r = 0.51%, eps = 0.98%", 0.0051, 0.0098, 1.0),
        ("Willow spec sheet, nominal, f = 0.95", 0.0033, 0.0077, 0.95),
    )
    device_rows = {}
    for label, r2q, eps, f in devices:
        print(f"\n  {label}")
        rows = device_report(4.0 * r2q / 3.0, eps, f, None, rng)
        device_rows[label] = rows
        print_device_report(rows)

    print("\n[5] Largest fan-out friend size G with a one-friend violation (mu = 1 - lambda, g_rev = 2 G)")
    size_specs = (
        ("r = 0.3%, eps = 1%", 4.0 * 0.003 / 3.0, 0.01),
        ("r = 0.33%, eps = 0.77% (Willow nominal)", 4.0 * 0.0033 / 3.0, 0.0077),
        ("r = 0.1%, eps = 0.5%", 4.0 * 0.001 / 3.0, 0.005),
        (f"lambda = {lam_eff:.4f}, eps = {run['eps']:.4f} (effective, IBM run)", lam_eff, run["eps"]),
    )
    print(f"  {'device parameters':<46} {'f=1, last':>10} {'f=1, first':>11} {'f=0.9, last':>12} {'f=0.9, first':>13}")
    friend_size_limits = []
    for label, lam_val, eps_val in size_specs:
        entry = {
            "label": label,
            "lambda": lam_val,
            "eps": eps_val,
            "f1_last": max_friend_size(lam_val, eps_val, 1.0, record="last"),
            "f1_first": max_friend_size(lam_val, eps_val, 1.0, record="first"),
            "f09_last": max_friend_size(lam_val, eps_val, 0.9, record="last"),
            "f09_first": max_friend_size(lam_val, eps_val, 0.9, record="first"),
        }
        friend_size_limits.append(entry)
        print(f"  {label:<46} {entry['f1_last']:10d} {entry['f1_first']:11d} "
              f"{entry['f09_last']:12d} {entry['f09_first']:13d}")
    if args.agents_json and os.path.exists(args.agents_json):
        fit_agents(args.agents_json)

    curves = None
    if args.plot or args.save_json:
        print("\n[6] Curves for the figure", flush=True)
        eps_grid = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        r_grid = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09]
        curves = {"eps_grid": eps_grid, "r_grid": r_grid, "critical_lambda": {}, "critical_eps": {},
                  "critical_f": {}, "critical_lambda_f1": {}}
        for name in TESTS:
            curves["critical_lambda"][name] = {
                f"{f:.1f}": monotone_envelope([critical_lambda(name, eps, f, rng) for eps in eps_grid],
                                              increasing=False) for f in (1.0, 0.9)}
            curves["critical_eps"][name] = {f"{f:.1f}": critical_eps(name, f, rng) for f in (1.0, 0.9)}
            curves["critical_f"][name] = {}
            curves["critical_lambda_f1"][name] = {}
            for eps, key in ((0.0, "0.0"), (0.01, "0.01")):
                curves["critical_f"][name][key] = monotone_envelope(
                    [critical_coherence(name, 4.0 * r / 3.0, eps, rng) for r in r_grid], increasing=True)
                curves["critical_lambda_f1"][name][key] = curves["critical_lambda"][name]["1.0"][
                    eps_grid.index(eps)]
            print(f"  {name}: done", flush=True)

    if args.plot and curves is not None:
        make_figure(curves, {"eps": run["eps"], "lambda_eff": lam_eff}, "lf_noise_budget")

    if args.save_json:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, float) and math.isinf(obj):
                return None
            return obj
        payload = {
            "model": "CNOT + two-qubit depolarizing (lambda) per entangling gate, record dephasing f, "
                     "symmetric readout error eps, pair visibility mu = 1 - lambda",
            "self_test": report,
            "critical_lambda_table": {"eps": list(eps_table), "f": list(f_table), "values": table},
            "calibration": {"run": run, "lambda_eff": lam_eff, "lambda_range": [lam_lo, lam_hi],
                            "tests": calib_rows, "margin_range": calib_range},
            "reference_devices": device_rows,
            "friend_size_limits": friend_size_limits,
            "curves": curves,
        }
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(clean(payload), handle, indent=2)
        print(f"[OK] Saved JSON summary: {args.save_json}")

    print(f"\nDone in {time.time() - started:.0f} s.")


if __name__ == "__main__":
    main()
