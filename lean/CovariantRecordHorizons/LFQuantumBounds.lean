import Mathlib.Tactic

/-!
  1. Quartic Genuine LF Quantum Violation Witness:
     For every coherence retention f > 0, choosing collinear friend axes
     (A1 = B1 = Z, A2 = B2 = -Z) and transverse settings with cos(theta) = 1 / (1 + f^2),
     sin(theta) = f * sqrt(2 + f^2) / (1 + f^2) is a valid normalized Bloch strategy
     achieving S_GLF(f) = 6 + f^4 / (1 + f^2) > 6.
-/

theorem genuineLF_bloch_normalization (f : ℝ) :
    let c := 1 / (1 + f ^ 2)
    let s := f * Real.sqrt (2 + f ^ 2) / (1 + f ^ 2)
    c ^ 2 + s ^ 2 = 1 := by
  dsimp
  have hpos : (0 : ℝ) ≤ 2 + f ^ 2 := by positivity
  have hne : 1 + f ^ 2 ≠ 0 := by positivity
  field_simp [hne]
  rw [Real.sq_sqrt hpos]
  ring

theorem genuineLF_quartic_excess (f : ℝ) (hf : 0 < f) :
    let c := 1 / (1 + f ^ 2)
    let s := f * Real.sqrt (2 + f ^ 2) / (1 + f ^ 2)
    let s_glf := 5 + 2 * c - (-(f * s) * (f * s) + c ^ 2)
    s_glf = 6 + f ^ 4 / (1 + f ^ 2) ∧ 6 < s_glf := by
  dsimp
  have hpos : (0 : ℝ) ≤ 2 + f ^ 2 := by positivity
  have hne : 1 + f ^ 2 ≠ 0 := by positivity
  have h_eq :
      5 + 2 * (1 / (1 + f ^ 2)) -
        (-(f * (f * Real.sqrt (2 + f ^ 2) / (1 + f ^ 2))) *
          (f * (f * Real.sqrt (2 + f ^ 2) / (1 + f ^ 2))) +
          (1 / (1 + f ^ 2)) ^ 2)
      = 6 + f ^ 4 / (1 + f ^ 2) := by
    field_simp [hne]
    rw [Real.sq_sqrt hpos]
    ring
  refine ⟨h_eq, ?_⟩
  rw [h_eq]
  have hquart : 0 < f ^ 4 / (1 + f ^ 2) := by positivity
  linarith
