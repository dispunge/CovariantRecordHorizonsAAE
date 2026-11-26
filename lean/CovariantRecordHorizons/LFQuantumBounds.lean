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

/-!
  2. Dimension-Independent Noncommutative `⋆`-Algebra Sum-of-Squares Identity
     (Theorem 1 and Generalized Noisy Semi-Brukner Identity, Eq. `eq:sos_noisy`):
     Over any unital noncommutative `⋆`-ring `A` that is a `⋆`-algebra over `ℝ`,
     given self-adjoint involutions `A1, N, B2, B3 : A` with commuting Alice/Bob
     operators (`[A1, B_y] = [N, B_y] = 0`) and real parameters `f, η1, η2, g`
     satisfying `g^2 = η1^2 - (1 - f^2) * η2^2`, the operator
       `K_eta = η1 • (A1 * (B3 - B2)) - η2 • ((N_+ + f • N_-) * (B3 + B2))`
     satisfies the exact operator Sum-of-Squares identity:
       `4 • (η1^2 + η2^2 * f^2) • 1 - K_eta * K_eta = star S1 * S1 + star S2 * S2`.
-/

section NoncommutativeSOS

variable {A : Type*} [Ring A] [StarRing A] [Algebra ℝ A] [StarModule ℝ A]

/-- Generalized noncommutative Sum-of-Squares certificate (Eq. `eq:sos_noisy`). -/
theorem noisy_semiBrukner_sos_identity
    (A1 N B2 B3 : A)
    (hA1_star : star A1 = A1) (hN_star : star N = N)
    (hB2_star : star B2 = B2) (hB3_star : star B3 = B3)
    (hA1_sq : A1 * A1 = 1) (hN_sq : N * N = 1)
    (hB2_sq : B2 * B2 = 1) (hB3_sq : B3 * B3 = 1)
    (hA1B2 : Commute A1 B2) (hA1B3 : Commute A1 B3)
    (hNB2 : Commute N B2) (hNB3 : Commute N B3)
    (f η1 η2 g : ℝ)
    (hg : g ^ 2 = η1 ^ 2 - (1 - f ^ 2) * η2 ^ 2) :
    let N_plus : A := (1 / 2 : ℝ) • (N + A1 * N * A1)
    let N_minus : A := (1 / 2 : ℝ) • (N - A1 * N * A1)
    let D_minus : A := B3 - B2
    let D_plus : A := B3 + B2
    let K_eta : A := η1 • (A1 * D_minus) - η2 • ((N_plus + f • N_minus) * D_plus)
    let S1 : A := η2 • f • D_minus + η1 • (A1 * N_minus * D_plus)
    let S2 : A := g • (N_plus * D_plus)
    4 • (η1 ^ 2 + η2 ^ 2 * f ^ 2) • (1 : A) - K_eta * K_eta =
      star S1 * S1 + star S2 * S2 := by
  intro N_plus N_minus D_minus D_plus K_eta S1 S2
  -- Commutativity of Alice observables with D_minus and D_plus
  have hA1Dm : Commute A1 D_minus := hA1B3.sub_right hA1B2
  have hA1Dp : Commute A1 D_plus := hA1B3.add_right hA1B2
  have hNDm : Commute N D_minus := hNB3.sub_right hNB2
  have hNDp : Commute N D_plus := hNB3.add_right hNB2
  have hANADm : Commute (A1 * N * A1) D_minus := (hA1Dm.mul_left hNDm).mul_left hA1Dm
  have hANADp : Commute (A1 * N * A1) D_plus := (hA1Dp.mul_left hNDp).mul_left hA1Dp
  have hNpDm : Commute N_plus D_minus := (hNDm.add_left hANADm).smul_left (1 / 2 : ℝ)
  have hNpDp : Commute N_plus D_plus := (hNDp.add_left hANADp).smul_left (1 / 2 : ℝ)
  have hNmDm : Commute N_minus D_minus := (hNDm.sub_left hANADm).smul_left (1 / 2 : ℝ)
  have hNmDp : Commute N_minus D_plus := (hNDp.sub_left hANADp).smul_left (1 / 2 : ℝ)
  have hA1NmDp : Commute (A1 * N_minus) D_plus := hA1Dp.mul_left hNmDp
  -- Bob involution identities: D_minus^2 + D_plus^2 = 4 and D_plus * D_minus = - D_minus * D_plus
  have hfour_one : (1 : A) + 1 + 1 + 1 = (4 : ℝ) • (1 : A) := by
    calc (1 : A) + 1 + 1 + 1
        = ((1 : ℝ) + 1 + 1 + 1) • (1 : A) := by simp [add_smul]
      _ = (4 : ℝ) • (1 : A) := by norm_num
  have hBob_sum : D_minus * D_minus + D_plus * D_plus = (4 : ℝ) • (1 : A) := by
    have hexpand : (B3 - B2) * (B3 - B2) + (B3 + B2) * (B3 + B2) =
        B3 * B3 + B2 * B2 + B3 * B3 + B2 * B2 := by noncomm_ring
    change (B3 - B2) * (B3 - B2) + (B3 + B2) * (B3 + B2) = (4 : ℝ) • (1 : A)
    rw [hexpand, hB2_sq, hB3_sq, hfour_one]
  have hDpDm : D_plus * D_minus = -(D_minus * D_plus) := by
    change (B3 + B2) * (B3 - B2) = -((B3 - B2) * (B3 + B2))
    have hanti : (B3 - B2) * (B3 + B2) + (B3 + B2) * (B3 - B2) = 0 := by
      calc (B3 - B2) * (B3 + B2) + (B3 + B2) * (B3 - B2)
          = B3 * B3 - B2 * B2 + (B3 * B3 - B2 * B2) := by noncomm_ring
        _ = 0 := by rw [hB2_sq, hB3_sq]; noncomm_ring
    exact eq_neg_of_add_eq_zero_right hanti
  -- Alice parity identities on N_plus and N_minus
  have hANA_sq : (A1 * N * A1) * (A1 * N * A1) = 1 := by
    calc (A1 * N * A1) * (A1 * N * A1)
        = A1 * N * (A1 * A1) * N * A1 := by noncomm_ring
      _ = A1 * (N * N) * A1 := by rw [hA1_sq]; noncomm_ring
      _ = 1 := by rw [hN_sq, mul_one, hA1_sq]
  have hNp_A1 : N_plus * A1 = A1 * N_plus := by
    dsimp [N_plus]
    rw [smul_mul_assoc, mul_smul_comm]
    congr 1
    calc (N + A1 * N * A1) * A1
        = N * A1 + A1 * N * (A1 * A1) := by noncomm_ring
      _ = N * A1 + A1 * N := by rw [hA1_sq]; noncomm_ring
      _ = A1 * N + (A1 * A1) * N * A1 := by rw [hA1_sq]; noncomm_ring
      _ = A1 * (N + A1 * N * A1) := by noncomm_ring
  have hNm_A1 : N_minus * A1 = -(A1 * N_minus) := by
    dsimp [N_minus]
    rw [smul_mul_assoc, mul_smul_comm, ← smul_neg]
    congr 1
    calc (N - A1 * N * A1) * A1
        = N * A1 - A1 * N * (A1 * A1) := by noncomm_ring
      _ = N * A1 - A1 * N := by rw [hA1_sq]; noncomm_ring
      _ = -(A1 * N - (A1 * A1) * N * A1) := by rw [hA1_sq]; noncomm_ring
      _ = -(A1 * (N - A1 * N * A1)) := by noncomm_ring
  have hNp_Nm_anti : N_plus * N_minus + N_minus * N_plus = 0 := by
    dsimp [N_plus, N_minus]
    simp only [smul_mul_assoc, mul_smul_comm, smul_smul, ← smul_add]
    suffices h : (N + A1 * N * A1) * (N - A1 * N * A1) +
        (N - A1 * N * A1) * (N + A1 * N * A1) = 0 by
      rw [h, smul_zero]
    calc (N + A1 * N * A1) * (N - A1 * N * A1) + (N - A1 * N * A1) * (N + A1 * N * A1)
        = (N * N - (A1 * N * A1) * (A1 * N * A1)) +
          (N * N - (A1 * N * A1) * (A1 * N * A1)) := by noncomm_ring
      _ = 0 := by rw [hN_sq, hANA_sq]; noncomm_ring
  have hNp_sq_Nm_sq : N_plus * N_plus + N_minus * N_minus = 1 := by
    dsimp [N_plus, N_minus]
    simp only [smul_mul_assoc, mul_smul_comm, smul_smul, ← smul_add]
    have hinner : (N + A1 * N * A1) * (N + A1 * N * A1) +
        (N - A1 * N * A1) * (N - A1 * N * A1) = (4 : ℝ) • (1 : A) := by
      calc (N + A1 * N * A1) * (N + A1 * N * A1) + (N - A1 * N * A1) * (N - A1 * N * A1)
          = N * N + (A1 * N * A1) * (A1 * N * A1) +
            N * N + (A1 * N * A1) * (A1 * N * A1) := by noncomm_ring
        _ = (1 : A) + 1 + 1 + 1 := by rw [hN_sq, hANA_sq]
        _ = (4 : ℝ) • (1 : A) := hfour_one
    rw [hinner, smul_smul]
    norm_num
  have hA1Nm_sq : (A1 * N_minus) * (A1 * N_minus) = -(N_minus * N_minus) := by
    calc (A1 * N_minus) * (A1 * N_minus)
        = A1 * (N_minus * A1) * N_minus := by noncomm_ring
      _ = A1 * (-(A1 * N_minus)) * N_minus := by rw [hNm_A1]
      _ = -((A1 * A1) * (N_minus * N_minus)) := by noncomm_ring
      _ = -(N_minus * N_minus) := by rw [hA1_sq, one_mul]
  -- Star evaluations on D_minus, D_plus, N_plus, N_minus, S1, S2
  have hDm_star : star D_minus = D_minus := by
    dsimp [D_minus]; rw [star_sub, hB3_star, hB2_star]
  have hDp_star : star D_plus = D_plus := by
    dsimp [D_plus]; rw [star_add, hB3_star, hB2_star]
  have hANA_star : star (A1 * N * A1) = A1 * N * A1 := by
    rw [star_mul, star_mul, hA1_star, hN_star, mul_assoc]
  have hNp_star : star N_plus = N_plus := by
    dsimp [N_plus]; rw [star_smul, star_trivial, star_add, hN_star, hANA_star]
  have hNm_star : star N_minus = N_minus := by
    dsimp [N_minus]; rw [star_smul, star_trivial, star_sub, hN_star, hANA_star]
  have hS1_star : star S1 = (η2 * f) • D_minus - η1 • (A1 * N_minus * D_plus) := by
    dsimp [S1]
    rw [star_add, star_smul, star_smul, star_trivial, star_trivial, smul_smul, hDm_star]
    rw [star_smul, star_trivial, star_mul, star_mul, hDp_star, hNm_star, hA1_star]
    rw [hNm_A1, mul_neg, hA1NmDp.symm.eq, smul_neg, sub_eq_add_neg]
  have hS2_star : star S2 = g • (N_plus * D_plus) := by
    dsimp [S2]
    rw [star_smul, star_trivial, star_mul, hDp_star, hNp_star, hNpDp.symm.eq]
  -- Make N_plus, N_minus, D_minus, D_plus opaque now that their algebraic identities are proved
  dsimp only [K_eta, S1, S2] at hS1_star hS2_star ⊢
  clear_value N_plus N_minus D_minus D_plus K_eta S1 S2
  -- Expand into common basis blocks X1, X2, X3, C
  set X1 : A := D_minus * D_minus with hX1_def
  set X2 : A := (N_plus * N_plus) * (D_plus * D_plus) with hX2_def
  set X3 : A := (N_minus * N_minus) * (D_plus * D_plus) with hX3_def
  set C : A := (A1 * N_minus) * (D_minus * D_plus - D_plus * D_minus) with hC_def
  have hUnit_decomp : (4 : ℝ) • (1 : A) = X1 + X2 + X3 := by
    rw [hX1_def, hX2_def, hX3_def, add_assoc, ← add_mul, hNp_sq_Nm_sq, one_mul, hBob_sum]
  have hLHS_unit :
      4 • (η1 ^ 2 + η2 ^ 2 * f ^ 2) • (1 : A) =
        (η1 ^ 2 + η2 ^ 2 * f ^ 2) • (X1 + X2 + X3) := by
    rw [← Nat.cast_smul_eq_nsmul ℝ 4, Nat.cast_ofNat, smul_comm, hUnit_decomp]
  -- Sub-product helper lemmas for K_eta^2 and S1^* S1
  have hADm_sq : (A1 * D_minus) * (A1 * D_minus) = X1 := by
    calc (A1 * D_minus) * (A1 * D_minus)
        = A1 * (D_minus * A1) * D_minus := by noncomm_ring
      _ = (A1 * A1) * (D_minus * D_minus) := by rw [hA1Dm.symm.eq]; noncomm_ring
      _ = X1 := by rw [hA1_sq, one_mul, hX1_def]
  have hNp_fNm_sq :
      ((N_plus + f • N_minus) * D_plus) * ((N_plus + f • N_minus) * D_plus) =
        X2 + (f ^ 2) • X3 := by
    have hcomm : Commute (N_plus + f • N_minus) D_plus :=
      hNpDp.add_left (hNmDp.smul_left f)
    have hmid : (N_plus + f • N_minus) * (N_plus + f • N_minus) =
        N_plus * N_plus + (f ^ 2) • (N_minus * N_minus) := by
      have hexp : (N_plus + f • N_minus) * (N_plus + f • N_minus) =
          N_plus * N_plus + f • (N_plus * N_minus + N_minus * N_plus) +
            (f ^ 2) • (N_minus * N_minus) := by
        simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, smul_add, smul_smul]
        module
      rw [hexp, hNp_Nm_anti, smul_zero, add_zero]
    calc ((N_plus + f • N_minus) * D_plus) * ((N_plus + f • N_minus) * D_plus)
        = (N_plus + f • N_minus) * (D_plus * (N_plus + f • N_minus)) * D_plus := by noncomm_ring
      _ = ((N_plus + f • N_minus) * (N_plus + f • N_minus)) * (D_plus * D_plus) := by
            rw [hcomm.symm.eq]; noncomm_ring
      _ = X2 + (f ^ 2) • X3 := by
            rw [hmid, add_mul, smul_mul_assoc, ← hX2_def, ← hX3_def]
  have hCross_Np :
      (A1 * D_minus) * (N_plus * D_plus) + (N_plus * D_plus) * (A1 * D_minus) = 0 := by
    have h1 : (A1 * D_minus) * (N_plus * D_plus) = (A1 * N_plus) * (D_minus * D_plus) := by
      calc (A1 * D_minus) * (N_plus * D_plus)
          = A1 * (D_minus * N_plus) * D_plus := by noncomm_ring
        _ = (A1 * N_plus) * (D_minus * D_plus) := by rw [hNpDm.symm.eq]; noncomm_ring
    have h2 : (N_plus * D_plus) * (A1 * D_minus) = (A1 * N_plus) * (D_plus * D_minus) := by
      calc (N_plus * D_plus) * (A1 * D_minus)
          = N_plus * (D_plus * A1) * D_minus := by noncomm_ring
        _ = (N_plus * A1) * (D_plus * D_minus) := by rw [hA1Dp.symm.eq]; noncomm_ring
        _ = (A1 * N_plus) * (D_plus * D_minus) := by rw [hNp_A1]
    rw [h1, h2, ← mul_add, hDpDm, add_neg_cancel, mul_zero]
  have hCross_Nm :
      (A1 * D_minus) * (N_minus * D_plus) + (N_minus * D_plus) * (A1 * D_minus) = C := by
    have h1 : (A1 * D_minus) * (N_minus * D_plus) = (A1 * N_minus) * (D_minus * D_plus) := by
      calc (A1 * D_minus) * (N_minus * D_plus)
          = A1 * (D_minus * N_minus) * D_plus := by noncomm_ring
        _ = (A1 * N_minus) * (D_minus * D_plus) := by rw [hNmDm.symm.eq]; noncomm_ring
    have h2 : (N_minus * D_plus) * (A1 * D_minus) = -((A1 * N_minus) * (D_plus * D_minus)) := by
      calc (N_minus * D_plus) * (A1 * D_minus)
          = N_minus * (D_plus * A1) * D_minus := by noncomm_ring
        _ = (N_minus * A1) * (D_plus * D_minus) := by rw [hA1Dp.symm.eq]; noncomm_ring
        _ = -((A1 * N_minus) * (D_plus * D_minus)) := by rw [hNm_A1, neg_mul]
    rw [h1, h2, hC_def, mul_sub, sub_eq_add_neg]
  have hCross_K :
      (A1 * D_minus) * ((N_plus + f • N_minus) * D_plus) +
        ((N_plus + f • N_minus) * D_plus) * (A1 * D_minus) = f • C := by
    have hexp :
        (A1 * D_minus) * ((N_plus + f • N_minus) * D_plus) +
          ((N_plus + f • N_minus) * D_plus) * (A1 * D_minus) =
        ((A1 * D_minus) * (N_plus * D_plus) + (N_plus * D_plus) * (A1 * D_minus)) +
          f • ((A1 * D_minus) * (N_minus * D_plus) + (N_minus * D_plus) * (A1 * D_minus)) := by
      simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, smul_add]
      module
    rw [hexp, hCross_Np, zero_add, hCross_Nm]
  have hK_sq :
      (η1 • (A1 * D_minus) - η2 • ((N_plus + f • N_minus) * D_plus)) *
        (η1 • (A1 * D_minus) - η2 • ((N_plus + f • N_minus) * D_plus)) =
      (η1 ^ 2) • X1 + (η2 ^ 2) • X2 + (η2 ^ 2 * f ^ 2) • X3 - (η1 * η2 * f) • C := by
    have hexp :
        (η1 • (A1 * D_minus) - η2 • ((N_plus + f • N_minus) * D_plus)) *
          (η1 • (A1 * D_minus) - η2 • ((N_plus + f • N_minus) * D_plus)) =
        (η1 ^ 2) • ((A1 * D_minus) * (A1 * D_minus)) +
          (η2 ^ 2) • (((N_plus + f • N_minus) * D_plus) * ((N_plus + f • N_minus) * D_plus)) -
          (η1 * η2) • ((A1 * D_minus) * ((N_plus + f • N_minus) * D_plus) +
            ((N_plus + f • N_minus) * D_plus) * (A1 * D_minus)) := by
      simp only [sub_mul, mul_sub, smul_mul_assoc, mul_smul_comm, smul_sub, smul_smul]
      module
    rw [hexp, hADm_sq, hNp_fNm_sq, hCross_K]
    module
  have hA1NmDp_sq : (A1 * N_minus * D_plus) * (A1 * N_minus * D_plus) = -X3 := by
    calc (A1 * N_minus * D_plus) * (A1 * N_minus * D_plus)
        = (A1 * N_minus) * (D_plus * (A1 * N_minus)) * D_plus := by noncomm_ring
      _ = ((A1 * N_minus) * (A1 * N_minus)) * (D_plus * D_plus) := by
          rw [hA1NmDp.symm.eq]; noncomm_ring
      _ = -X3 := by rw [hX3_def, hA1Nm_sq, neg_mul]
  have hS1_cross :
      D_minus * (A1 * N_minus * D_plus) - (A1 * N_minus * D_plus) * D_minus = C := by
    have hA1NmDm : Commute (A1 * N_minus) D_minus := hA1Dm.mul_left hNmDm
    have h1 : D_minus * (A1 * N_minus * D_plus) = (A1 * N_minus) * (D_minus * D_plus) := by
      calc D_minus * (A1 * N_minus * D_plus)
          = (D_minus * (A1 * N_minus)) * D_plus := by noncomm_ring
        _ = (A1 * N_minus) * (D_minus * D_plus) := by rw [hA1NmDm.symm.eq]; noncomm_ring
    have h2 : (A1 * N_minus * D_plus) * D_minus = (A1 * N_minus) * (D_plus * D_minus) := by
      noncomm_ring
    rw [h1, h2, hC_def, mul_sub]
  have hS1_prod :
      star (η2 • f • D_minus + η1 • (A1 * N_minus * D_plus)) *
        (η2 • f • D_minus + η1 • (A1 * N_minus * D_plus)) =
      (η2 ^ 2 * f ^ 2) • X1 + (η1 ^ 2) • X3 + (η1 * η2 * f) • C := by
    rw [hS1_star]
    have hexp :
        ((η2 * f) • D_minus - η1 • (A1 * N_minus * D_plus)) *
          (η2 • f • D_minus + η1 • (A1 * N_minus * D_plus)) =
        (η2 ^ 2 * f ^ 2) • (D_minus * D_minus) -
          (η1 ^ 2) • ((A1 * N_minus * D_plus) * (A1 * N_minus * D_plus)) +
          (η1 * η2 * f) • (D_minus * (A1 * N_minus * D_plus) -
            (A1 * N_minus * D_plus) * D_minus) := by
      simp only [sub_mul, mul_add, smul_mul_assoc, mul_smul_comm, smul_sub, smul_smul]
      module
    rw [hexp, ← hX1_def, hA1NmDp_sq, hS1_cross]
    module
  have hS2_prod :
      star (g • (N_plus * D_plus)) * (g • (N_plus * D_plus)) = (g ^ 2) • X2 := by
    rw [hS2_star, smul_mul_assoc, mul_smul_comm, smul_smul, ← sq]
    congr 1
    calc (N_plus * D_plus) * (N_plus * D_plus)
        = N_plus * (D_plus * N_plus) * D_plus := by noncomm_ring
      _ = X2 := by rw [hX2_def, hNpDp.symm.eq]; noncomm_ring
  -- Combine all four pieces and discharge by ℝ-module normalization
  rw [hLHS_unit, hK_sq, hS1_prod, hS2_prod, hg]
  module

/-- Unit-visibility specialization (`η1 = 1, η2 = 1, g = f`) proving the exact
    noncommutative Sum-of-Squares identity of Theorem 1 (`eq:sos_identity`). -/
theorem theorem1_sos_identity
    (A1 N B2 B3 : A)
    (hA1_star : star A1 = A1) (hN_star : star N = N)
    (hB2_star : star B2 = B2) (hB3_star : star B3 = B3)
    (hA1_sq : A1 * A1 = 1) (hN_sq : N * N = 1)
    (hB2_sq : B2 * B2 = 1) (hB3_sq : B3 * B3 = 1)
    (hA1B2 : Commute A1 B2) (hA1B3 : Commute A1 B3)
    (hNB2 : Commute N B2) (hNB3 : Commute N B3)
    (f : ℝ) :
    let N_plus : A := (1 / 2 : ℝ) • (N + A1 * N * A1)
    let N_minus : A := (1 / 2 : ℝ) • (N - A1 * N * A1)
    let D_minus : A := B3 - B2
    let D_plus : A := B3 + B2
    let K : A := A1 * D_minus - (N_plus + f • N_minus) * D_plus
    let S1 : A := f • D_minus + A1 * N_minus * D_plus
    let S2 : A := f • (N_plus * D_plus)
    4 • (1 + f ^ 2) • (1 : A) - K * K = star S1 * S1 + star S2 * S2 := by
  have hg : f ^ 2 = (1 : ℝ) ^ 2 - (1 - f ^ 2) * (1 : ℝ) ^ 2 := by ring
  have hgen := noisy_semiBrukner_sos_identity A1 N B2 B3
    hA1_star hN_star hB2_star hB3_star hA1_sq hN_sq hB2_sq hB3_sq
    hA1B2 hA1B3 hNB2 hNB3 f 1 1 f hg
  dsimp only at hgen ⊢
  have hscalar : (1 : ℝ) ^ 2 + (1 : ℝ) ^ 2 * f ^ 2 = 1 + f ^ 2 := by ring
  simpa only [one_smul, hscalar] using hgen

end NoncommutativeSOS

/-!
  3. Multi-Gate Friend-Size Threshold and SOS Certificate Admissibility
     (Sec. IV, Eqs. `eq:sos_noisy`, `eq:eta_gates`, and `eq:golden_threshold`):
     * `sos_hypothesis_of_le`: whenever `0 ≤ f ≤ 1` and `0 ≤ η2 ≤ η1` (in particular
       for `g_rev ≥ g_rec`), `0 ≤ η1^2 - (1 - f^2) * η2^2`, so `g := √(η1^2 - (1 - f^2) * η2^2)`
       satisfies the real hypothesis `hg` of `noisy_semiBrukner_sos_identity`.
     * `golden_threshold`: in the ideal-pair, noiseless-readout, zero-leakage fan-out case
       `S_SB = 2 v √(1 + v^2)`, the violation condition `1 < v^2 + v^4` holds iff
       `(√5 - 1) / 2 < v^2`.
-/

/-- Admissibility of the generalized noisy SOS certificate when `0 ≤ η2 ≤ η1` and `0 ≤ f ≤ 1`. -/
theorem sos_hypothesis_of_le (f η1 η2 : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1)
    (h2 : 0 ≤ η2) (h12 : η2 ≤ η1) :
    0 ≤ η1 ^ 2 - (1 - f ^ 2) * η2 ^ 2 := by
  have _ : 0 ≤ f ∧ f ≤ 1 := ⟨hf0, hf1⟩
  have h_eta_sq : η2 ^ 2 ≤ η1 ^ 2 := by nlinarith
  have h_prod : (1 - f ^ 2) * η2 ^ 2 ≤ η2 ^ 2 := by nlinarith [sq_nonneg f, sq_nonneg η2]
  linarith

/-- Golden-ratio visibility threshold (`eq:golden_threshold`): `1 < v^2 + v^4 ↔ (√5 - 1) / 2 < v^2`. -/
theorem golden_threshold (v : ℝ) :
    1 < v ^ 2 + v ^ 4 ↔ (Real.sqrt 5 - 1) / 2 < v ^ 2 := by
  have h5_nonneg : (0 : ℝ) ≤ 5 := by norm_num
  have hsqrt5_sq : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt h5_nonneg
  have hsqrt5_nonneg : 0 ≤ Real.sqrt 5 := Real.sqrt_nonneg 5
  have hv_sq : 0 ≤ v ^ 2 := sq_nonneg v
  constructor
  · intro h
    by_contra hnot
    have hle : v ^ 2 ≤ (Real.sqrt 5 - 1) / 2 := le_of_not_gt hnot
    nlinarith
  · intro h
    nlinarith

