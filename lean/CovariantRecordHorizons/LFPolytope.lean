/-
  LFPolytope.lean
  Zero-dependency Lean 4 kernel verification of the (3,3,2,2) Bell (LHV) and
  Two-Friend Local Friendliness (LF) polytope facets from Bong et al. (2020).
-/

structure Corr15 where
  a1 : Int
  a2 : Int
  a3 : Int
  b1 : Int
  b2 : Int
  b3 : Int
  e11 : Int
  e12 : Int
  e13 : Int
  e21 : Int
  e22 : Int
  e23 : Int
  e31 : Int
  e32 : Int
  e33 : Int
  deriving DecidableEq, Repr

/-- Cartesian product helper avoiding minor-version library name drift. -/
def cprod {α β : Type} (xs : List α) (ys : List β) : List (α × β) :=
  xs.foldr (fun x acc => ys.map (fun y => (x, y)) ++ acc) []

def signs : List Int := [-1, 1]
def signs2 : List (Int × Int) := cprod signs signs
def signs3 : List (Int × Int × Int) :=
  (cprod signs signs2).map fun (x, (y, z)) => (x, y, z)
def signs4 : List (Int × Int × Int × Int) :=
  (cprod signs2 signs2).map fun ((w, x), (y, z)) => (w, x, y, z)

/-- Deterministic Local Hidden Variable (LHV) vertex in R^15. -/
def mkLHV (a b : Int × Int × Int) : Corr15 :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  { a1, a2, a3, b1, b2, b3,
    e11 := a1 * b1, e12 := a1 * b2, e13 := a1 * b3,
    e21 := a2 * b1, e22 := a2 * b2, e23 := a2 * b3,
    e31 := a3 * b1, e32 := a3 * b2, e33 := a3 * b3 }

def lhvVertices : List Corr15 :=
  (cprod signs3 signs3).map fun (a, b) => mkLHV a b

/-- The 8 Popescu-Rohrlich (PR) non-signaling sub-boxes on settings {2,3} x {2,3}. -/
def prSubBoxes : List (Int × Int × Int × Int) :=
  signs4.filter fun (e22, e23, e32, e33) => e22 * e23 * e32 * e33 == -1

/-- Extreme Two-Friend LF vertex combining deterministic friend outcomes (c, d) with a PR box. -/
def mkPRLF (cd : Int × Int) (pr : Int × Int × Int × Int) : Corr15 :=
  let (c, d) := cd
  let (e22, e23, e32, e33) := pr
  { a1 := c, a2 := 0, a3 := 0,
    b1 := d, b2 := 0, b3 := 0,
    e11 := c * d, e12 := 0, e13 := 0,
    e21 := 0,     e22,      e23,
    e31 := 0,     e32,      e33 }

def lfVertices : List Corr15 :=
  lhvVertices ++ (cprod signs2 prSubBoxes).map fun (cd, pr) => mkPRLF cd pr

/-! ### Canonical Bong et al. (2020) Table I Functionals -/

def semiBrukner (p : Corr15) : Int := -p.e12 + p.e13 - p.e32 - p.e33
def brukner     (p : Corr15) : Int := -p.e11 + p.e12 - p.e21 - p.e22
def bellNonLF   (p : Corr15) : Int :=  p.e22 - p.e23 - p.e32 - p.e33

def i3322LF (p : Corr15) : Int :=
  -p.a1 + p.a2 + p.b1 - p.b2
  + p.e11 - p.e12 - p.e13 - p.e21 + p.e22 - p.e23 - p.e31 - p.e32

def genuineLF (p : Corr15) : Int :=
  -p.a1 - p.a2 - p.b1 - p.b2
  - p.e11 - 2 * p.e12 - 2 * p.e21 + 2 * p.e22 - p.e23 - p.e32 - p.e33

/-! ### Exact Fraction-Free Homogeneous Matrix Rank over Q (via Int) -/

def toHomRow (p : Corr15) : List Int :=
  [1, p.a1, p.a2, p.a3, p.b1, p.b2, p.b3,
   p.e11, p.e12, p.e13, p.e21, p.e22, p.e23, p.e31, p.e32, p.e33]

def primRow (w : List Int) : List Int :=
  let g : Nat := w.foldl (fun acc x => Nat.gcd acc x.natAbs) 0
  if g ≤ 1 then w else w.map (· / Int.ofNat g)

def rankAux : Nat → List (List Int) → Nat
  | 0, _ => 0
  | fuel + 1, rows =>
    match rows.find? (fun r => match r with | x :: _ => x != 0 | [] => false) with
    | none =>
      let tails := rows.filterMap (fun r => match r with | _ :: xs => some xs | [] => none)
      if tails.isEmpty then 0 else rankAux fuel tails
    | some (p :: ps) =>
      let reduced := rows.filterMap fun r =>
        match r with
        | [] => none
        | x :: xs =>
          let w := primRow (List.zipWith (fun xi pi => p * xi - x * pi) xs ps)
          if w.all (· == 0) then none else some w
      1 + rankAux fuel reduced
    | some [] => 0

def tightHomRank (verts : List Corr15) (f : Corr15 → Int) (bound : Int) : Nat :=
  let tightRows := (verts.filter (fun p => f p == bound)).map toHomRow
  rankAux 16 tightRows

/-! ### Machine-Checked Kernel Theorems (Zero `sorry`) -/

theorem vertex_counts : lhvVertices.length = 64 ∧ lfVertices.length = 96 := by decide

theorem valid_lf_inequalities :
    lfVertices.all (fun p => semiBrukner p ≤ 2) = true ∧
    lfVertices.all (fun p => brukner p ≤ 2) = true ∧
    lfVertices.all (fun p => i3322LF p ≤ 4) = true ∧
    lfVertices.all (fun p => genuineLF p ≤ 6) = true := by decide

theorem bellNonLF_separates_LHV_and_LF :
    lhvVertices.all (fun p => bellNonLF p ≤ 2) = true ∧
    lfVertices.any (fun p => bellNonLF p == 4) = true := by decide

/-- Genuine LF has 14 tight LHV vertices (homogeneous LHV rank 12 < 15,
    NOT a Bell facet), but achieves full facet rank 15 on the 96-vertex LF polytope. -/
theorem genuineLF_is_strict_LF_facet :
    (lhvVertices.filter (fun p => genuineLF p == 6)).length = 14 ∧
    tightHomRank lhvVertices genuineLF 6 = 12 ∧
    tightHomRank lfVertices genuineLF 6 = 15 := by decide
