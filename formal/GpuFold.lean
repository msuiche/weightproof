import Std.Tactic.BVDecide

set_option maxRecDepth 100000000
set_option maxHeartbeats 20000000

namespace Weightproof

/-- Goldilocks modulus. The extension equations below are proved over the
integer lift, before reduction modulo this prime. -/
def goldilocksModulus : Nat := 0xffffffff00000001

@[ext]
structure Ext where
  c0 : Int
  c1 : Int
  c2 : Int
  deriving DecidableEq, Repr

namespace Ext

def zero : Ext := ⟨0, 0, 0⟩

def add (a b : Ext) : Ext :=
  ⟨a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2⟩

def sub (a b : Ext) : Ext :=
  ⟨a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2⟩

/-- Multiplication in the integer lift of E = F[X]/(X^3 - 2). -/
def mul (a b : Ext) : Ext :=
  ⟨a.c0 * b.c0 + 2 * (a.c1 * b.c2 + a.c2 * b.c1),
   a.c0 * b.c1 + a.c1 * b.c0 + 2 * (a.c2 * b.c2),
   a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0⟩

/-- Independent reference specification of one multilinear fold. -/
def foldReference (zero one challenge : Ext) : Ext :=
  add zero (mul challenge (sub one zero))

/-- Component-level equations emitted by the fused ICICLE program. -/
def foldFused (zero one challenge : Ext) : Ext :=
  let d0 := one.c0 - zero.c0
  let d1 := one.c1 - zero.c1
  let d2 := one.c2 - zero.c2
  ⟨zero.c0 + d0 * challenge.c0 + 2 * (d1 * challenge.c2 + d2 * challenge.c1),
   zero.c1 + d0 * challenge.c1 + d1 * challenge.c0 + 2 * (d2 * challenge.c2),
   zero.c2 + d0 * challenge.c2 + d1 * challenge.c1 + d2 * challenge.c0⟩

/-- Universal correctness: the fused component equations implement exactly the
reference fold for every input and challenge. -/
theorem foldFused_correct (zero one challenge : Ext) :
    foldFused zero one challenge = foldReference zero one challenge := by
  cases zero
  cases one
  cases challenge
  apply Ext.ext <;>
    simp [foldFused, foldReference, add, sub, mul,
      Int.mul_sub, Int.sub_mul, Int.mul_add] <;>
    ac_rfl

/-- Non-vacuity: a concrete non-zero execution of the same fused equations. -/
theorem foldFused_nonvacuous :
    foldFused ⟨1, 2, 3⟩ ⟨4, 5, 6⟩ ⟨7, 11, 13⟩ ≠ zero := by
  decide

/-- Quadratic round polynomial for one pair of multilinear evaluations. -/
structure DotRound where
  q0 : Int
  q1 : Int
  q2 : Int

def dotRound (a0 a1 b0 b1 : Int) : DotRound :=
  let q0 := a0 * b0
  let q2 := (a1 - a0) * (b1 - b0)
  ⟨q0, a1 * b1 - q0 - q2, q2⟩

def DotRound.eval (q : DotRound) (challenge : Int) : Int :=
  q.q0 + q.q1 * challenge + q.q2 * challenge * challenge

def dotProductAt (a0 a1 b0 b1 challenge : Int) : Int :=
  (a0 + (a1 - a0) * challenge) * (b0 + (b1 - b0) * challenge)

/-- Universal correctness of the coefficient extraction emitted by the Rust
multilinear dot-product prover. -/
theorem dotRound_correct (a0 a1 b0 b1 challenge : Int) :
    (dotRound a0 a1 b0 b1).eval challenge = dotProductAt a0 a1 b0 b1 challenge := by
  simp [DotRound.eval, dotRound, dotProductAt,
    Int.sub_eq_add_neg, Int.mul_add,
    Int.mul_neg, Int.neg_add, Int.neg_neg,
    Int.mul_comm, Int.mul_left_comm] <;>
    omega

theorem dotRound_nonvacuous :
    (dotRound 2 5 7 11).eval 13 ≠ 0 := by
  decide

/-- Coefficients for the following round, computed from the folded values in
`dot_round_kernel` after the current challenge is applied. -/
def nextDotRound (a0 a1 b0 b1 challenge : Int) : DotRound :=
  dotRound
    (a0 + (a1 - a0) * challenge)
    (a1 + (a1 - a0) * challenge)
    (b0 + (b1 - b0) * challenge)
    (b1 + (b1 - b0) * challenge)

/-- The fused fold-then-coefficient phase matches direct evaluation on the
folded state for every input, fold challenge, and evaluation point. -/
theorem nextDotRound_correct (a0 a1 b0 b1 foldChallenge evalChallenge : Int) :
    (nextDotRound a0 a1 b0 b1 foldChallenge).eval evalChallenge =
      dotProductAt
        (a0 + (a1 - a0) * foldChallenge)
        (a1 + (a1 - a0) * foldChallenge)
        (b0 + (b1 - b0) * foldChallenge)
        (b1 + (b1 - b0) * foldChallenge)
        evalChallenge := by
  exact dotRound_correct _ _ _ _ _

theorem nextDotRound_nonvacuous :
    (nextDotRound 2 5 7 11 13).eval 17 ≠ 0 := by
  decide

#print axioms nextDotRound_correct
#print axioms nextDotRound_nonvacuous

#print axioms dotRound_correct
#print axioms dotRound_nonvacuous

#print axioms foldFused_correct
#print axioms foldFused_nonvacuous

end Ext

namespace Kernel

/-- The modulus as a positive integer, for canonical field representatives. -/
def modulus : Int := 0xffffffff00000001

structure Field where
  value : Int
  nonneg : 0 ≤ value
  lt : value < modulus
  deriving DecidableEq, Repr

namespace Field

@[ext]
theorem ext {a b : Field} (h : a.value = b.value) : a = b := by
  cases a
  cases b
  cases h
  rfl

def ofInt (x : Int) : Field :=
  ⟨x % modulus,
   Int.emod_nonneg _ (by decide),
   Int.emod_lt_of_pos _ (by decide)⟩

def add (a b : Field) : Field := ofInt (a.value + b.value)
def sub (a b : Field) : Field := ofInt (a.value - b.value)
def mul (a b : Field) : Field := ofInt (a.value * b.value)

theorem ofInt_value (a : Field) : ofInt a.value = a := by
  apply Field.ext
  exact Int.emod_eq_of_lt a.nonneg a.lt

theorem ofInt_add (a b : Int) : add (ofInt a) (ofInt b) = ofInt (a + b) := by
  apply Field.ext
  simp [add, ofInt, Int.add_emod]

theorem ofInt_sub (a b : Int) : sub (ofInt a) (ofInt b) = ofInt (a - b) := by
  apply Field.ext
  simp [sub, ofInt, Int.sub_emod]

theorem ofInt_mul (a b : Int) : mul (ofInt a) (ofInt b) = ofInt (a * b) := by
  apply Field.ext
  simp [mul, ofInt, Int.mul_emod]

end Field

@[ext]
structure FieldExt where
  c0 : Field
  c1 : Field
  c2 : Field
  deriving DecidableEq, Repr

namespace FieldExt

def zero : FieldExt := ⟨Field.ofInt 0, Field.ofInt 0, Field.ofInt 0⟩
def two : Field := Field.ofInt 2

/-- Interpret an integer-lift value as canonical Goldilocks coordinates. -/
def ofInt (a : Weightproof.Ext) : FieldExt :=
  ⟨Field.ofInt a.c0, Field.ofInt a.c1, Field.ofInt a.c2⟩

/-- Every canonical field element is the image of its integer value. -/
def integerLift (a : FieldExt) : Weightproof.Ext :=
  ⟨a.c0.value, a.c1.value, a.c2.value⟩

def add (a b : FieldExt) : FieldExt :=
  ⟨Field.add a.c0 b.c0, Field.add a.c1 b.c1, Field.add a.c2 b.c2⟩

def sub (a b : FieldExt) : FieldExt :=
  ⟨Field.sub a.c0 b.c0, Field.sub a.c1 b.c1, Field.sub a.c2 b.c2⟩

def mul (a b : FieldExt) : FieldExt :=
  ⟨Field.add (Field.mul a.c0 b.c0)
      (Field.mul two (Field.add (Field.mul a.c1 b.c2) (Field.mul a.c2 b.c1))),
   Field.add (Field.add (Field.mul a.c0 b.c1) (Field.mul a.c1 b.c0))
      (Field.mul two (Field.mul a.c2 b.c2)),
   Field.add (Field.mul a.c0 b.c2)
      (Field.add (Field.mul a.c1 b.c1) (Field.mul a.c2 b.c0))⟩

def foldReference (zero one challenge : FieldExt) : FieldExt :=
  add zero (mul challenge (sub one zero))

/-- Exact canonical-field counterpart of `fold_value` in `cuda/fold.cu`. -/
def foldFused (zero one challenge : FieldExt) : FieldExt :=
  let d0 := Field.sub one.c0 zero.c0
  let d1 := Field.sub one.c1 zero.c1
  let d2 := Field.sub one.c2 zero.c2
  ⟨Field.add zero.c0 (Field.add (Field.mul d0 challenge.c0)
      (Field.mul two (Field.add (Field.mul d1 challenge.c2) (Field.mul d2 challenge.c1)))),
   Field.add zero.c1 (Field.add (Field.add (Field.mul d0 challenge.c1)
      (Field.mul d1 challenge.c0)) (Field.mul two (Field.mul d2 challenge.c2))),
   Field.add zero.c2 (Field.add (Field.mul d0 challenge.c2)
      (Field.add (Field.mul d1 challenge.c1) (Field.mul d2 challenge.c0)))⟩

theorem ofInt_integerLift (a : FieldExt) : ofInt (integerLift a) = a := by
  apply FieldExt.ext <;> exact Field.ofInt_value _

/-- Canonical field reduction preserves the integer-lift equations. -/
theorem foldFused_ofInt_correct (zero one challenge : Weightproof.Ext) :
    foldFused (ofInt zero) (ofInt one) (ofInt challenge) =
      foldReference (ofInt zero) (ofInt one) (ofInt challenge) := by
  apply FieldExt.ext <;>
    simp [foldFused, foldReference, add, sub, mul, ofInt, two,
      Field.ofInt_add, Field.ofInt_sub, Field.ofInt_mul] <;>
    ac_rfl

/-- Universal correctness for every canonical field input and challenge. -/
theorem foldFused_correct (zero one challenge : FieldExt) :
    foldFused zero one challenge = foldReference zero one challenge := by
  rw [← ofInt_integerLift zero, ← ofInt_integerLift one, ← ofInt_integerLift challenge]
  exact foldFused_ofInt_correct _ _ _

theorem foldFused_nonvacuous :
    foldFused
      (ofInt ⟨1, 2, 3⟩) (ofInt ⟨4, 5, 6⟩) (ofInt ⟨7, 11, 13⟩) ≠ zero := by
  decide

end FieldExt

/-! Exact natural-number model of the unsigned machine-word operations in
`cuda/fold.cu`. -/
namespace Word

def base : Nat := 0x10000000000000000
def epsilon : Nat := 0xffffffff
def modulus : Nat := 0xffffffff00000001

def wrapSub (a b : Nat) : Nat := (a + base - b) % base

theorem wrapSub_of_le {a b : Nat} (_ha : a < base) (hb : b ≤ a) :
    wrapSub a b = a - b := by
  have h : a + base - b = a - b + base := by omega
  simp [wrapSub, h, Nat.add_mod_right, Nat.mod_eq_of_lt (show a - b < base by omega)]

theorem wrapSub_of_lt {a b : Nat} (_ha : a < base) (hb : b < base) (h : a < b) :
    wrapSub a b = base - (b - a) := by
  simp [wrapSub, Nat.mod_eq_of_lt (show a + base - b < base by omega)]
  omega

theorem wrapSub_lt (a b : Nat) : wrapSub a b < base := by
  exact Nat.mod_lt _ (by decide)

def add (a b : Nat) : Nat :=
  let notB := wrapSub modulus b
  let difference := wrapSub a notB
  if a < notB then wrapSub difference epsilon else difference

def sub (a b : Nat) : Nat :=
  let difference := wrapSub a b
  if a < b then wrapSub difference epsilon else difference

theorem add_correct {a b : Nat} (ha : a < modulus) (hb : b < modulus) :
    add a b = (a + b) % modulus := by
  have haBase : a < base := by simp [base, modulus] at *; omega
  have hbBase : b < base := by simp [base, modulus] at *; omega
  have hnotBase : modulus - b < base := by simp [base, modulus] at *; omega
  have hnot : wrapSub modulus b = modulus - b :=
    wrapSub_of_le (by simp [base, modulus]) (by omega)
  by_cases h : a < modulus - b
  · have hd : wrapSub a (modulus - b) = base - (modulus - b - a) :=
      wrapSub_of_lt haBase hnotBase h
    have hdBase : wrapSub a (modulus - b) < base := wrapSub_lt _ _
    have he : epsilon ≤ wrapSub a (modulus - b) := by
      simp [base, modulus, epsilon] at *
      omega
    simp only [add, hnot, h, ↓reduceIte]
    rw [wrapSub_of_le hdBase he, hd]
    rw [Nat.mod_eq_of_lt]
    · simp [base, modulus, epsilon] at *
      omega
    · simp [modulus] at *
      omega
  · have hle : modulus - b ≤ a := Nat.le_of_not_gt h
    have hd : wrapSub a (modulus - b) = a - (modulus - b) :=
      wrapSub_of_le haBase hle
    simp only [add, hnot, h, ↓reduceIte, hd]
    rw [Nat.mod_eq_sub_mod (show a + b ≥ modulus by simp [modulus] at *; omega)]
    rw [Nat.mod_eq_of_lt]
    · simp [modulus] at *
      omega
    · simp [modulus] at *
      omega

theorem sub_correct {a b : Nat} (ha : a < modulus) (hb : b < modulus) :
    sub a b = (a + modulus - b) % modulus := by
  have haBase : a < base := by simp [base, modulus] at *; omega
  have hbBase : b < base := by simp [base, modulus] at *; omega
  by_cases h : a < b
  · have hd : wrapSub a b = base - (b - a) := wrapSub_of_lt haBase hbBase h
    have hdBase : wrapSub a b < base := wrapSub_lt _ _
    have he : epsilon ≤ wrapSub a b := by
      simp [base, modulus, epsilon] at *
      omega
    simp only [sub, h, ↓reduceIte]
    rw [wrapSub_of_le hdBase he, hd]
    rw [Nat.mod_eq_of_lt]
    · simp [base, modulus, epsilon] at *
      omega
    · simp [modulus] at *
      omega
  · have hle : b ≤ a := Nat.le_of_not_gt h
    have hd : wrapSub a b = a - b := wrapSub_of_le haBase hle
    simp only [sub, h, ↓reduceIte, hd]
    have hraw : a + modulus - b = modulus + (a - b) := by omega
    rw [hraw, Nat.add_mod_left, Nat.mod_eq_of_lt (show a - b < modulus by omega)]

def canonical (x : Nat) : Nat := if x ≥ modulus then x - modulus else x

theorem canonical_correct {x : Nat} (hx : x < 2 * modulus) :
    canonical x = x % modulus := by
  by_cases h : x ≥ modulus
  · simp (config := { decide := true }) only [canonical, h, ↓reduceIte]
    rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt (by omega)]
  · have hx' : x < modulus := Nat.lt_of_not_ge h
    simp [canonical, h, Nat.mod_eq_of_lt hx']

def highWord (hi : Nat) : Nat := hi / 0x100000000
def lowWord (hi : Nat) : Nat := hi % 0x100000000

def reduceDifference (hi lo : Nat) : Nat :=
  wrapSub (wrapSub lo (highWord hi)) (if lo < highWord hi then epsilon else 0)

theorem highWord_lt {hi : Nat} (hhi : hi < modulus) : highWord hi < 0x100000000 := by
  have hhiBase : hi < base := by simp [base, modulus] at *; omega
  exact (Nat.div_lt_iff_lt_mul (by decide)).2 (by simpa [highWord, base] using hhiBase)

theorem reduceDifference_mod {hi lo : Nat}
    (hhi : hi < modulus) (hlo : lo < base) (m : Nat) :
    (reduceDifference hi lo + m) % modulus =
      (lo + modulus - highWord hi + m) % modulus := by
  have hhighSmall := highWord_lt hhi
  have hhighBase : highWord hi < base := by simp [base] at *; omega
  by_cases h : lo < highWord hi
  · have hd : wrapSub lo (highWord hi) = base - (highWord hi - lo) :=
      wrapSub_of_lt hlo hhighBase h
    have hdBase : wrapSub lo (highWord hi) < base := wrapSub_lt _ _
    have he : epsilon ≤ wrapSub lo (highWord hi) := by
      simp [base, modulus, epsilon] at *
      omega
    simp only [reduceDifference, h, ↓reduceIte]
    rw [wrapSub_of_le hdBase he, hd]
    have hraw : lo + modulus - highWord hi + m = modulus - (highWord hi - lo) + m := by omega
    rw [hraw]
    have hbase : base - (highWord hi - lo) - epsilon = modulus - (highWord hi - lo) := by
      simp [base, modulus, epsilon]
      omega
    rw [hbase]
  · have hle : highWord hi ≤ lo := Nat.le_of_not_gt h
    have hd : wrapSub lo (highWord hi) = lo - highWord hi :=
      wrapSub_of_le hlo hle
    have hdBase : wrapSub lo (highWord hi) < base := wrapSub_lt _ _
    simp only [reduceDifference, h, ↓reduceIte]
    rw [wrapSub_of_le hdBase (Nat.zero_le _), hd]
    have hraw : lo + modulus - highWord hi + m = modulus + (lo - highWord hi + m) := by omega
    rw [hraw, Nat.add_mod_left]
    simp only [Nat.sub_zero]

def adjust (d m : Nat) : Nat :=
  ((d + m) % base + if (d + m) % base < m then epsilon else 0) % base

def reduce128 (hi lo : Nat) : Nat :=
  canonical (adjust (reduceDifference hi lo) (lowWord hi * 0x100000000 - lowWord hi))

theorem lowWord_mul (hi : Nat) :
    lowWord hi * 0x100000000 - lowWord hi = lowWord hi * epsilon := by
  simp [epsilon]
  omega

theorem reduce128_overflow {hi lo : Nat}
    (hdBase : reduceDifference hi lo < base)
    (hmLt : lowWord hi * 0x100000000 - lowWord hi < modulus)
    (hover : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) ≥ base) :
    reduce128 hi lo =
      (reduceDifference hi lo + (lowWord hi * 0x100000000 - lowWord hi)) % modulus := by
  have hsumBase : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) < 2 * base := by
    have hpbase : modulus < base := by decide
    omega
  have hr : (reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi)) % base =
      reduceDifference hi lo + (lowWord hi * 0x100000000 - lowWord hi) - base := by
    rw [Nat.mod_eq_sub_mod hover, Nat.mod_eq_of_lt (by omega)]
  have hrm : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) - base <
      lowWord hi * 0x100000000 - lowWord hi := by omega
  simp only [reduce128, adjust, hr, hrm, ↓reduceIte]
  have hsumStrong : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) < base + modulus := by
    have hpbase : modulus < base := by decide
    omega
  have hbaseEq : base = modulus + epsilon := by decide
  have harg : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) - base + epsilon < base := by
    omega
  have hraw : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) - base + epsilon =
      reduceDifference hi lo + (lowWord hi * 0x100000000 - lowWord hi) - modulus := by
    omega
  rw [Nat.mod_eq_of_lt harg, hraw]
  have hbound : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) - modulus < 2 * modulus := by
    have hbase : 2 * base < 3 * modulus := by decide
    omega
  have hge : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) ≥ modulus := by
    have hpbase : modulus < base := by decide
    omega
  exact (canonical_correct hbound).trans (Nat.mod_eq_sub_mod hge).symm

theorem canonical_adjust_noOverflow {d m : Nat} (hlt : d + m < base) :
    canonical (adjust d m) = (d + m) % modulus := by
  have hr : (d + m) % base = d + m := Nat.mod_eq_of_lt hlt
  have hrm : ¬d + m < m := by omega
  simp only [adjust, hr, hrm, ↓reduceIte, Nat.add_zero]
  exact canonical_correct (by
    have hbase : base < 2 * modulus := by decide
    omega)

theorem reduce128_noOverflow {hi lo : Nat}
    (hlt : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) < base) :
    reduce128 hi lo =
      (reduceDifference hi lo + (lowWord hi * 0x100000000 - lowWord hi)) % modulus :=
  canonical_adjust_noOverflow hlt

theorem reduce128_core {hi lo : Nat} (_hhi : hi < modulus) (_hlo : lo < base) :
    reduce128 hi lo =
      (reduceDifference hi lo + (lowWord hi * 0x100000000 - lowWord hi)) % modulus := by
  have hdBase : reduceDifference hi lo < base := wrapSub_lt _ _
  have hlow : lowWord hi < 0x100000000 := Nat.mod_lt _ (by decide)
  have hm : lowWord hi * 0x100000000 - lowWord hi = lowWord hi * epsilon := lowWord_mul hi
  have hmLt : lowWord hi * 0x100000000 - lowWord hi < modulus := by
    rw [hm]
    have hlowEps : lowWord hi ≤ epsilon := by simp [epsilon]; omega
    have hle : lowWord hi * epsilon ≤ epsilon * epsilon :=
      Nat.mul_le_mul_right _ hlowEps
    have heps : epsilon * epsilon < modulus := by decide
    omega
  by_cases hover : reduceDifference hi lo +
      (lowWord hi * 0x100000000 - lowWord hi) ≥ base
  · exact reduce128_overflow hdBase hmLt hover
  · exact reduce128_noOverflow (Nat.lt_of_not_ge hover)

theorem reduce128_correct {hi lo : Nat} (hhi : hi < modulus) (hlo : lo < base) :
    reduce128 hi lo = (hi * base + lo) % modulus := by
  rw [reduce128_core hhi hlo, reduceDifference_mod hhi hlo]
  have hdecomp := Nat.div_add_mod hi 0x100000000
  simp only [highWord, lowWord, base, modulus] at *
  omega

def mul (a b : Nat) : Nat := reduce128 ((a * b) / base) ((a * b) % base)

theorem productHigh_lt {a b : Nat} (ha : a < modulus) (hb : b < modulus) :
    (a * b) / base < modulus := by
  by_cases ha0 : a = 0
  · simp [ha0, modulus]
  · have hbBase : b < base := Nat.lt_trans hb (by decide)
    have haPos : 0 < a := by omega
    have hproduct : a * b < a * base :=
      (Nat.mul_lt_mul_left haPos).2 hbBase
    have hdiv : (a * b) / base < a :=
      (Nat.div_lt_iff_lt_mul (by simp [base])).2 hproduct
    exact Nat.lt_trans hdiv ha

theorem mul_correct {a b : Nat} (ha : a < modulus) (hb : b < modulus) :
    mul a b = (a * b) % modulus := by
  unfold mul
  have hdecomp : (a * b) / base * base + (a * b) % base = a * b := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod _ _
  calc
    reduce128 ((a * b) / base) ((a * b) % base) =
        ((a * b) / base * base + (a * b) % base) % modulus :=
      reduce128_correct (productHigh_lt ha hb) (Nat.mod_lt _ (by decide))
    _ = (a * b) % modulus := by rw [hdecomp]

theorem add_nonvacuous : add 3 5 = 8 := by decide
theorem sub_nonvacuous : sub 3 5 = modulus - 2 := by decide
theorem reduce128_nonvacuous : reduce128 1 2 = epsilon + 2 := by decide
theorem mul_nonvacuous : mul 3 5 = 15 := by decide

end Word

namespace Field

/-- Natural machine-word representative of a canonical field element. -/
def toWord (a : Field) : Nat := a.value.toNat

/-- Reinterpret a natural word in the canonical field. -/
def ofWord (x : Nat) : Field := ofInt (x : Int)

theorem ofWord_toWord (a : Field) : ofWord (toWord a) = a := by
  apply Field.ext
  simp [ofWord, Field.ofInt, toWord, Int.toNat_of_nonneg a.nonneg,
    Int.emod_eq_of_lt a.nonneg a.lt]

theorem toWord_lt_modulus (a : Field) : toWord a < Word.modulus := by
  apply (Int.toNat_lt a.nonneg).2
  simpa [Word.modulus, modulus] using a.lt

theorem ofWord_mod (x : Nat) : ofWord (x % Word.modulus) = ofWord x := by
  apply Field.ext
  simp [ofWord, Field.ofInt, Word.modulus, modulus]

theorem toWord_ofWord {x : Nat} (hx : x < Word.modulus) : toWord (ofWord x) = x := by
  simp only [toWord, ofWord, Field.ofInt]
  rw [Int.toNat_emod (x := (x : Int)) (y := modulus)
    (Int.natCast_nonneg x) (by decide)]
  simpa [Word.modulus, modulus] using (Nat.mod_eq_of_lt hx)

theorem word_add_lt_modulus {a b : Nat} (ha : a < Word.modulus) (hb : b < Word.modulus) :
    Word.add a b < Word.modulus := by
  rw [Word.add_correct ha hb]
  exact Nat.mod_lt _ (by decide)

theorem word_sub_lt_modulus {a b : Nat} (ha : a < Word.modulus) (hb : b < Word.modulus) :
    Word.sub a b < Word.modulus := by
  rw [Word.sub_correct ha hb]
  exact Nat.mod_lt _ (by decide)

theorem word_mul_lt_modulus {a b : Nat} (ha : a < Word.modulus) (hb : b < Word.modulus) :
    Word.mul a b < Word.modulus := by
  rw [Word.mul_correct ha hb]
  exact Nat.mod_lt _ (by decide)

/-- CUDA addition implements the canonical field addition already used by the
fold and launch proofs. -/
theorem word_add_correct (a b : Field) :
    ofWord (Word.add (toWord a) (toWord b)) = Field.add a b := by
  rw [Word.add_correct (toWord_lt_modulus a) (toWord_lt_modulus b), ofWord_mod]
  apply Field.ext
  simp [ofWord, Field.ofInt, Field.add, toWord,
    Int.toNat_of_nonneg a.nonneg, Int.toNat_of_nonneg b.nonneg]

/-- CUDA subtraction implements canonical field subtraction. -/
theorem word_sub_correct (a b : Field) :
    ofWord (Word.sub (toWord a) (toWord b)) = Field.sub a b := by
  rw [Word.sub_correct (toWord_lt_modulus a) (toWord_lt_modulus b), ofWord_mod]
  apply Field.ext
  have hb_le : toWord b ≤ toWord a + Word.modulus := by
    have hb := toWord_lt_modulus b
    omega
  simp only [ofWord, Field.ofInt, Field.sub]
  rw [Int.ofNat_sub hb_le]
  simp [toWord, Word.modulus, modulus,
    Int.toNat_of_nonneg a.nonneg, Int.toNat_of_nonneg b.nonneg, Int.sub_emod]

/-- The exact 128-bit reduction implements canonical field multiplication. -/
theorem word_mul_correct (a b : Field) :
    ofWord (Word.mul (toWord a) (toWord b)) = Field.mul a b := by
  rw [Word.mul_correct (toWord_lt_modulus a) (toWord_lt_modulus b), ofWord_mod]
  apply Field.ext
  simp [ofWord, Field.ofInt, Field.mul, toWord,
    Int.toNat_of_nonneg a.nonneg, Int.toNat_of_nonneg b.nonneg]

theorem toWord_word_add (a b : Field) :
    Word.add (toWord a) (toWord b) = toWord (Field.add a b) := by
  rw [← Field.word_add_correct a b,
    toWord_ofWord (word_add_lt_modulus (toWord_lt_modulus a) (toWord_lt_modulus b))]

theorem toWord_word_sub (a b : Field) :
    Word.sub (toWord a) (toWord b) = toWord (Field.sub a b) := by
  rw [← Field.word_sub_correct a b,
    toWord_ofWord (word_sub_lt_modulus (toWord_lt_modulus a) (toWord_lt_modulus b))]

theorem toWord_word_mul (a b : Field) :
    Word.mul (toWord a) (toWord b) = toWord (Field.mul a b) := by
  rw [← Field.word_mul_correct a b,
    toWord_ofWord (word_mul_lt_modulus (toWord_lt_modulus a) (toWord_lt_modulus b))]

theorem toWord_word_mul_two (a : Field) :
    Word.mul 2 (toWord a) = toWord (Field.mul FieldExt.two a) := by
  have htwo : toWord FieldExt.two = 2 := by decide
  rw [← htwo]
  exact toWord_word_mul FieldExt.two a

end Field

/-- The fixed kernel's public launch parameters and initialized input. -/
structure Launch where
  input : Array FieldExt
  outputCount : Nat
  lowHalf : Nat
  challenge : FieldExt
  inputBase : Nat
  outputBase : Nat

namespace Launch

/-- Host and FFI preconditions from `wp_fold_cuda`. The bitwise condition is
exactly `low_half & (low_half - 1) == 0`. -/
structure Valid (launch : Launch) : Prop where
  inputSize : launch.input.size = 2 * launch.outputCount
  outputPositive : 0 < launch.outputCount
  lowHalfPositive : 0 < launch.lowHalf
  powerOfTwo : launch.lowHalf &&& (launch.lowHalf - 1) = 0
  lowHalfLe : launch.lowHalf ≤ launch.outputCount
  divides : launch.lowHalf ∣ launch.outputCount
  inputAddress : launch.inputBase + 24 * launch.input.size < Word.base
  outputAddress : launch.outputBase + 24 * launch.outputCount < Word.base

theorem lowHalf_lt_base (launch : Launch) (valid : Valid launch) :
    launch.lowHalf < Word.base := by
  have houtput := valid.outputAddress
  have hle := valid.lowHalfLe
  have hpositive := valid.outputPositive
  omega

def zeroIndex (launch : Launch) (index : Nat) : Nat :=
  index % launch.lowHalf + 2 * launch.lowHalf * (index / launch.lowHalf)

def oneIndex (launch : Launch) (index : Nat) : Nat :=
  zeroIndex launch index + launch.lowHalf

/-- The high input load is in bounds for every active thread. -/
theorem oneIndex_lt_twice_output (launch : Launch) (valid : Valid launch)
    {index : Nat} (hi : index < launch.outputCount) :
    oneIndex launch index < 2 * launch.outputCount := by
  have hdiv : launch.lowHalf * (launch.outputCount / launch.lowHalf) =
      launch.outputCount := Nat.mul_div_cancel' valid.divides
  have hindex : index <
      (launch.outputCount / launch.lowHalf) * launch.lowHalf := by
    rw [Nat.mul_comm, hdiv]
    exact hi
  have hblock : index / launch.lowHalf < launch.outputCount / launch.lowHalf :=
    (Nat.div_lt_iff_lt_mul valid.lowHalfPositive).2 hindex
  have hspan :
      2 * launch.lowHalf * (index / launch.lowHalf + 1) ≤ 2 * launch.outputCount := by
    calc
      2 * launch.lowHalf * (index / launch.lowHalf + 1) ≤
          2 * launch.lowHalf * (launch.outputCount / launch.lowHalf) :=
        Nat.mul_le_mul_left _ (Nat.succ_le_of_lt hblock)
      _ = 2 * launch.outputCount := by rw [Nat.mul_assoc, hdiv]
  have hexpand :
      2 * launch.lowHalf * (index / launch.lowHalf + 1) =
        2 * launch.lowHalf * (index / launch.lowHalf) + 2 * launch.lowHalf := by
    simp [Nat.mul_add]
  have hoffset : index % launch.lowHalf < launch.lowHalf :=
    Nat.mod_lt _ valid.lowHalfPositive
  simp only [oneIndex, zeroIndex]
  omega

theorem zeroIndex_lt_input (launch : Launch) (valid : Valid launch)
    {index : Nat} (hi : index < launch.outputCount) :
    zeroIndex launch index < launch.input.size := by
  have hone := oneIndex_lt_twice_output launch valid hi
  rw [valid.inputSize]
  simp only [oneIndex] at hone
  omega

theorem oneIndex_lt_input (launch : Launch) (valid : Valid launch)
    {index : Nat} (hi : index < launch.outputCount) :
    oneIndex launch index < launch.input.size := by
  rw [valid.inputSize]
  exact oneIndex_lt_twice_output launch valid hi

/-- One active CUDA thread, including the kernel's early-return guards and two
checked global loads. A `some` result is the single store at `output[index]`. -/
def runThread (launch : Launch) (index : Nat) : Option FieldExt :=
  if launch.lowHalf = 0 then
    none
  else if index ≥ launch.outputCount then
    none
  else do
    let zero ← launch.input[zeroIndex launch index]?
    let one ← launch.input[oneIndex launch index]?
    pure (FieldExt.foldFused zero one launch.challenge)

/-- Every accepted active thread computes the independent reference fold at the
same source indices. This also rules out partial execution from failed loads. -/
theorem runThread_correct (launch : Launch) (valid : Valid launch)
    {index : Nat} (hi : index < launch.outputCount) :
    runThread launch index =
      some (FieldExt.foldReference
        (launch.input[zeroIndex launch index]'(zeroIndex_lt_input launch valid hi))
        (launch.input[oneIndex launch index]'(oneIndex_lt_input launch valid hi))
        launch.challenge) := by
  have hlow : ¬launch.lowHalf = 0 := Nat.ne_of_gt valid.lowHalfPositive
  have hactive : ¬index ≥ launch.outputCount := Nat.not_le_of_lt hi
  simp [runThread, hlow, hactive,
    Array.getElem?_eq_getElem (zeroIndex_lt_input launch valid hi),
    Array.getElem?_eq_getElem (oneIndex_lt_input launch valid hi),
    FieldExt.foldFused_correct]

/-- Out-of-range threads return before any load or store. -/
theorem runThread_inactive (launch : Launch) {index : Nat}
    (hi : index ≥ launch.outputCount) : runThread launch index = none := by
  by_cases hlow : launch.lowHalf = 0 <;> simp [runThread, hlow, hi]

end Launch

/-! Exact semantics of the PTX instruction subset emitted for `fold_kernel`.
The operations use natural machine words and the same wrapping arithmetic as
ROCetta's SMT-LIB PTX semantics. -/
namespace Ptx

structure State where
  r64 : Nat → Nat
  r32 : Nat → Nat
  predicate : Nat → Bool
  output : Nat → Nat

structure Thread where
  blockIndex : Nat
  blockDim : Nat
  threadIndex : Nat

def Thread.index (thread : Thread) : Nat :=
  thread.blockIndex * thread.blockDim + thread.threadIndex

theorem Thread.index_mod (thread : Thread) (h32 : thread.index < 0x100000000) :
    (thread.blockIndex * thread.blockDim + thread.threadIndex) % 0x100000000 =
      thread.index := Nat.mod_eq_of_lt h32

def initial : State := {
  r64 := fun _ => 0
  r32 := fun _ => 0
  predicate := fun _ => false
  output := fun _ => 0
}

def get64 (state : State) (register : Nat) : Nat := state.r64 register
def get32 (state : State) (register : Nat) : Nat := state.r32 register
def getPredicate (state : State) (register : Nat) : Bool := state.predicate register

inductive Value where
  | register : Nat → Value
  | immediate : Nat → Value

instance : OfNat Value 0 where ofNat := .immediate 0
instance : OfNat Value 4294967295 where ofNat := .immediate 4294967295
instance (n : Nat) [NeZero n] : OfNat Value n where ofNat := .register n

def eval64 (state : State) (value : Value) : Nat :=
  match value with
  | .register register => get64 state register
  | .immediate immediate => immediate

def eval32 (state : State) (value : Value) : Nat :=
  match value with
  | .register register => get32 state register
  | .immediate immediate => immediate

def set64 (state : State) (register value : Nat) : State :=
  { state with r64 := fun r => if r = register then value else state.r64 r }

def set32 (state : State) (register value : Nat) : State :=
  { state with r32 := fun r => if r = register then value else state.r32 r }

def setPredicate (state : State) (register : Nat) (value : Bool) : State :=
  { state with predicate := fun r => if r = register then value else state.predicate r }

def store (state : State) (base address value : Nat) : State :=
  { state with output := fun i => if i = (address - base) / 8 then value else state.output i }

def inputWord (launch : Launch) (wordIndex : Nat) : Nat :=
  match launch.input[wordIndex / 3]? with
  | none => 0
  | some ext =>
    match wordIndex % 3 with
    | 0 => Field.toWord ext.c0
    | 1 => Field.toWord ext.c1
    | _ => Field.toWord ext.c2

def load (launch : Launch) (base address : Nat) : Nat :=
  inputWord launch ((address - base) / 8)

def ldGlobal (state : State) (launch : Launch) (destination address : Nat) : State :=
  set64 state destination (load launch launch.inputBase (get64 state address))

def ldGlobalOffset (state : State) (launch : Launch) (destination address offset : Nat) : State :=
  set64 state destination (load launch launch.inputBase (get64 state address + offset))

def stGlobal (state : State) (launch : Launch) (address source : Nat) : State :=
  store state launch.outputBase (get64 state address) (get64 state source)

def stGlobalOffset (state : State) (launch : Launch) (address offset source : Nat) : State :=
  store state launch.outputBase (get64 state address + offset) (get64 state source)

def mov64 (state : State) (destination value : Nat) : State := set64 state destination value
def mov32 (state : State) (destination value : Nat) : State := set32 state destination value
def cvt64 (state : State) (destination source : Nat) : State :=
  set64 state destination (get32 state source)
def cvt32 (state : State) (destination source : Nat) : State :=
  set32 state destination (get64 state source % 0x100000000)
def cvta (state : State) (destination source : Nat) : State :=
  set64 state destination (get64 state source)

def add64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination ((eval64 state left + eval64 state right) % Word.base)
def sub64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination (Word.wrapSub (eval64 state left) (eval64 state right))
def mulLo64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination ((eval64 state left * eval64 state right) % Word.base)
def mulHi64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination ((eval64 state left * eval64 state right) / Word.base)
def madLo64 (state : State) (destination : Nat) (left right addend : Value) : State :=
  set64 state destination ((eval64 state left * eval64 state right + eval64 state addend) % Word.base)
def sub32 (state : State) (destination : Nat) (left right : Value) : State :=
  set32 state destination ((eval32 state left + 0x100000000 - eval32 state right) % 0x100000000)
def mulLo32 (state : State) (destination : Nat) (left right : Value) : State :=
  set32 state destination ((eval32 state left * eval32 state right) % 0x100000000)
def madLo32 (state : State) (destination : Nat) (left right addend : Value) : State :=
  set32 state destination ((eval32 state left * eval32 state right + eval32 state addend) % 0x100000000)
def div64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination (eval64 state left / eval64 state right)
def div32 (state : State) (destination : Nat) (left right : Value) : State :=
  set32 state destination (eval32 state left / eval32 state right)
def shl64 (state : State) (destination : Nat) (value shift : Value) : State :=
  set64 state destination ((eval64 state value * 2 ^ eval64 state shift) % Word.base)
def shr64 (state : State) (destination : Nat) (value shift : Value) : State :=
  set64 state destination (eval64 state value / 2 ^ eval64 state shift)
def madLo64ImmRight (state : State) (destination left right addend : Nat) : State :=
  set64 state destination ((get64 state left * right + get64 state addend) % Word.base)
def shl64Imm (state : State) (destination value shift : Nat) : State :=
  set64 state destination ((get64 state value * 2 ^ shift) % Word.base)
def shr64Imm (state : State) (destination value shift : Nat) : State :=
  set64 state destination (get64 state value / 2 ^ shift)
def and64ImmRight (state : State) (destination left right : Nat) : State :=
  set64 state destination (get64 state left &&& right)
def and64 (state : State) (destination : Nat) (left right : Value) : State :=
  set64 state destination (eval64 state left &&& eval64 state right)
def selp64 (state : State) (destination : Nat) (left right : Value) (predicate : Nat) : State :=
  set64 state destination (if getPredicate state predicate then eval64 state left else eval64 state right)
def setpEq (state : State) (destination : Nat) (left right : Value) : State :=
  setPredicate state destination (eval64 state left = eval64 state right)
def setpNe (state : State) (destination : Nat) (left right : Value) : State :=
  setPredicate state destination (eval64 state left != eval64 state right)
def setpLt (state : State) (destination : Nat) (left right : Value) : State :=
  setPredicate state destination (eval64 state left < eval64 state right)
def setpLe (state : State) (destination : Nat) (left right : Value) : State :=
  setPredicate state destination (eval64 state left ≤ eval64 state right)
def setpGt (state : State) (destination : Nat) (left right : Value) : State :=
  setPredicate state destination (eval64 state left > eval64 state right)

def subField (state : State) (base predicate left right : Nat) : State :=
  let s := sub64 state base (Value.register left) (Value.register right)
  let s := setPredicate s predicate (get64 s left < get64 s right)
  let s := add64 s (base + 1) (Value.register base) (Value.immediate 18446744069414584321)
  selp64 s (base + 2) (Value.register (base + 1)) (Value.register base) predicate

def addField (state : State) (base predicate left right : Nat) : State :=
  let s := sub64 state base (Value.immediate 18446744069414584321) (Value.register right)
  let s := sub64 s (base + 1) (Value.register left) (Value.register base)
  let s := setPredicate s predicate (get64 s left < get64 s base)
  let s := add64 s (base + 2) (Value.register (base + 1)) (Value.immediate 18446744069414584321)
  selp64 s (base + 3) (Value.register (base + 2)) (Value.register (base + 1)) predicate

def mulField (state : State) (base predicate left right : Nat) : State :=
  let s := mulHi64 state base (Value.register left) (Value.register right)
  let s := mulLo64 s (base + 1) (Value.register left) (Value.register right)
  let s := shr64Imm s (base + 2) base 32
  let s := and64ImmRight s (base + 3) base 4294967295
  let s := sub64 s (base + 4) (Value.register (base + 1)) (Value.register (base + 2))
  let s := setPredicate s predicate (get64 s (base + 1) < get64 s (base + 2))
  let s := selp64 s (base + 5) (Value.immediate 18446744069414584321) 0 predicate
  let s := shl64Imm s (base + 6) base 32
  let s := sub64 s (base + 7) (Value.register (base + 6)) (Value.register (base + 3))
  let s := add64 s (base + 8) (Value.register (base + 4)) (Value.register (base + 7))
  let s := add64 s (base + 9) (Value.register (base + 8)) (Value.register (base + 5))
  let s := setPredicate s (predicate + 1) (get64 s (base + 9) < get64 s (base + 7))
  let s := selp64 s (base + 10) 4294967295 0 (predicate + 1)
  let s := add64 s (base + 11) (Value.register (base + 10)) (Value.register (base + 9))
  let s := setPredicate s (predicate + 2)
    (get64 s (base + 11) > 18446744069414584320)
  let s := add64 s (base + 12) (Value.register (base + 11)) 4294967295
  selp64 s (base + 13) (Value.register (base + 12)) (Value.register (base + 11)) (predicate + 2)

def doubleField (state : State) (base predicate source : Nat) : State :=
  let s := mulHi64 state base 84 (Value.register source)
  let s := shl64Imm s (base + 1) source 1
  let s := shr64Imm s (base + 2) base 32
  let s := and64ImmRight s (base + 3) base 4294967295
  let s := sub64 s (base + 4) (Value.register (base + 1)) (Value.register (base + 2))
  let s := setPredicate s predicate (get64 s (base + 1) < get64 s (base + 2))
  let s := selp64 s (base + 5) (Value.immediate 18446744069414584321) 0 predicate
  let s := shl64Imm s (base + 6) base 32
  let s := sub64 s (base + 7) (Value.register (base + 6)) (Value.register (base + 3))
  let s := add64 s (base + 8) (Value.register (base + 4)) (Value.register (base + 7))
  let s := add64 s (base + 9) (Value.register (base + 8)) (Value.register (base + 5))
  let s := setPredicate s (predicate + 1) (get64 s (base + 9) < get64 s (base + 7))
  let s := selp64 s (base + 10) 4294967295 0 (predicate + 1)
  let s := add64 s (base + 11) (Value.register (base + 10)) (Value.register (base + 9))
  let s := setPredicate s (predicate + 2)
    (get64 s (base + 11) > 18446744069414584320)
  let s := add64 s (base + 12) (Value.register (base + 11)) 4294967295
  selp64 s (base + 13) (Value.register (base + 12)) (Value.register (base + 11)) (predicate + 2)

def outputExt (state : State) (index : Nat) : FieldExt :=
  ⟨Field.ofWord (state.output (3 * index)),
   Field.ofWord (state.output (3 * index + 1)),
   Field.ofWord (state.output (3 * index + 2))⟩

theorem get64_set64 (state : State) (register value r : Nat) :
    get64 (set64 state register value) r =
      if r = register then value else get64 state r := rfl

theorem get32_set32 (state : State) (register value r : Nat) :
    get32 (set32 state register value) r =
      if r = register then value else get32 state r := rfl

theorem getPredicate_setPredicate (state : State) (register : Nat) (value : Bool) (r : Nat) :
    getPredicate (setPredicate state register value) r =
      if r = register then value else getPredicate state r := rfl

theorem get64_mov64_ne (state : State) (destination value r : Nat) (h : r ≠ destination) :
    get64 (mov64 state destination value) r = get64 state r := by
  simp [mov64, get64_set64, h]

theorem get64_setPredicate (state : State) (register : Nat) (value : Bool) (r : Nat) :
    get64 (setPredicate state register value) r = get64 state r := rfl

theorem get32_setPredicate (state : State) (register : Nat) (value : Bool) (r : Nat) :
    get32 (setPredicate state register value) r = get32 state r := rfl

theorem getPredicate_set64 (state : State) (register value r : Nat) :
    getPredicate (set64 state register value) r = getPredicate state r := rfl

theorem getPredicate_set32 (state : State) (register value r : Nat) :
    getPredicate (set32 state register value) r = getPredicate state r := rfl

theorem get64_set32 (state : State) (register value r : Nat) :
    get64 (set32 state register value) r = get64 state r := rfl

theorem get32_set64 (state : State) (register value r : Nat) :
    get32 (set64 state register value) r = get32 state r := rfl

theorem output_store (state : State) (base address value i : Nat) :
    (store state base address value).output i =
      if i = (address - base) / 8 then value else state.output i := rfl

theorem get64_store (state : State) (base address value r : Nat) :
    get64 (store state base address value) r = get64 state r := rfl

theorem get32_store (state : State) (base address value r : Nat) :
    get32 (store state base address value) r = get32 state r := rfl

theorem getPredicate_store (state : State) (base address value r : Nat) :
    getPredicate (store state base address value) r = getPredicate state r := rfl

theorem get64_initial (r : Nat) : get64 initial r = 0 := rfl

theorem get32_initial (r : Nat) : get32 initial r = 0 := rfl

theorem getPredicate_initial (r : Nat) : getPredicate initial r = false := rfl

theorem output_initial (i : Nat) : initial.output i = 0 := rfl

theorem inputWord_c0 (launch : Launch) (index : Nat) (hi : index < launch.input.size) :
    inputWord launch (3 * index) = Field.toWord (launch.input[index]'hi).c0 := by
  simp [inputWord, Array.getElem?_eq_getElem hi]

theorem inputWord_c1 (launch : Launch) (index : Nat) (hi : index < launch.input.size) :
    inputWord launch (3 * index + 1) = Field.toWord (launch.input[index]'hi).c1 := by
  simp [inputWord, Nat.add_div, Nat.add_mod, Array.getElem?_eq_getElem hi]

theorem inputWord_c2 (launch : Launch) (index : Nat) (hi : index < launch.input.size) :
    inputWord launch (3 * index + 2) = Field.toWord (launch.input[index]'hi).c2 := by
  simp [inputWord, Nat.add_div, Nat.add_mod, Array.getElem?_eq_getElem hi]

end Ptx

def ptxPrepare (launch : Launch) (thread : Ptx.Thread) : Ptx.State :=
  let s := Ptx.initial
  let s := Ptx.set64 s 11 launch.inputBase
  let s := Ptx.set64 s 1 (Field.toWord launch.challenge.c2)
  let s := Ptx.set64 s 2 (Field.toWord launch.challenge.c1)
  let s := Ptx.set64 s 3 (Field.toWord launch.challenge.c0)
  let s := Ptx.set64 s 12 launch.outputBase
  let s := Ptx.set64 s 13 launch.outputCount
  let s := Ptx.set64 s 14 launch.lowHalf
  let s := Ptx.set32 s 1 thread.blockIndex
  let s := Ptx.set32 s 2 thread.blockDim
  let s := Ptx.set32 s 3 thread.threadIndex
  let s := Ptx.madLo32 s 4 (Ptx.Value.register 1) (Ptx.Value.register 2) (Ptx.Value.register 3)
  Ptx.cvt64 s 4 4

/-- Both PTX address paths compute the same block and offset. The 32-bit path
is used when the stride has no set bits above bit 31. -/
def ptxAddress (state : Ptx.State) : Ptx.State :=
  let s := Ptx.and64 state 15 (Ptx.Value.register 14) (Ptx.Value.immediate 18446744069414584320)
  if Ptx.get64 s 15 != 0 then
    let s := Ptx.div64 s 231 (Ptx.Value.register 4) (Ptx.Value.register 14)
    let s := Ptx.mulLo64 s 16 (Ptx.Value.register 231) (Ptx.Value.register 14)
    Ptx.sub64 s 232 (Ptx.Value.register 4) (Ptx.Value.register 16)
  else
    let s := Ptx.cvt32 s 5 14
    let s := Ptx.cvt32 s 6 4
    let s := Ptx.div32 s 7 (Ptx.Value.register 6) (Ptx.Value.register 5)
    let s := Ptx.mulLo32 s 8 (Ptx.Value.register 7) (Ptx.Value.register 5)
    let s := Ptx.sub32 s 9 (Ptx.Value.register 6) (Ptx.Value.register 8)
    let s := Ptx.cvt64 s 231 7
    Ptx.cvt64 s 232 9

def ptxArithmeticAddress (state : Ptx.State) : Ptx.State :=
  let s := Ptx.mulLo64 state 17 14 231
  let s := Ptx.shl64Imm s 18 17 1
  let s := Ptx.add64 s 19 232 18
  let s := Ptx.cvta s 20 11
  let s := Ptx.madLo64ImmRight s 21 19 24 20
  Ptx.madLo64ImmRight s 22 14 24 21

def ptxArithmeticLoad0 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.ldGlobalOffset state launch 23 22 0
  let s := Ptx.ldGlobalOffset s launch 24 21 0
  Ptx.subField s 25 4 23 24

def ptxArithmeticLoad1 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.ldGlobalOffset state launch 28 22 8
  let s := Ptx.ldGlobalOffset s launch 29 21 8
  Ptx.subField s 30 5 28 29

def ptxArithmeticLoad2 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.ldGlobalOffset state launch 33 22 16
  let s := Ptx.ldGlobalOffset s launch 34 21 16
  Ptx.subField s 35 6 33 34

def ptxArithmeticInputs (launch : Launch) (state : Ptx.State) : Ptx.State :=
  ptxArithmeticLoad2 launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch
    (ptxArithmeticAddress state)))

def ptxArithmeticProducts (state : Ptx.State) : Ptx.State :=
  let s := Ptx.mulField state 38 7 27 3
  let s := Ptx.mulField s 52 10 32 1
  Ptx.mulField s 66 13 37 2

def ptxArithmeticBase (launch : Launch) (state : Ptx.State) : Ptx.State :=
  ptxArithmeticProducts (ptxArithmeticInputs launch state)

def ptxArithmeticC0 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.addField state 80 16 65 79
  let s := Ptx.mov64 s 84 2
  let s := Ptx.doubleField s 85 17 83
  Ptx.addField s 99 20 51 98

def ptxArithmeticC1 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.mulField state 103 21 27 2
  let s := Ptx.mulField s 117 24 32 3
  let s := Ptx.addField s 131 27 116 130
  let s := Ptx.mulField s 135 28 37 1
  let s := Ptx.doubleField s 149 29 148
  Ptx.addField s 163 34 134 162

def ptxArithmeticC2 (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := Ptx.mulField state 167 35 27 1
  let s := Ptx.mulField s 181 38 32 2
  let s := Ptx.mulField s 195 41 37 3
  let s := Ptx.addField s 209 44 194 208
  Ptx.addField s 213 45 180 212

def ptxArithmeticOutputBase (launch : Launch) (state : Ptx.State)
    (base predicate left right : Nat) : Ptx.State :=
  let s := Ptx.addField state base predicate left right
  let s := Ptx.cvta s 229 12
  Ptx.madLo64ImmRight s 230 4 24 229

def ptxArithmeticOutput (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s0 := ptxArithmeticOutputBase launch state 217 46 24 102
  let s1 := ptxArithmeticOutputBase launch state 221 47 29 166
  let s2 := ptxArithmeticOutputBase launch state 225 48 34 216
  let t0 := Ptx.store s0 launch.outputBase (Ptx.get64 s0 230 + 0) (Ptx.get64 s0 220)
  let t1 := Ptx.store t0 launch.outputBase (Ptx.get64 s1 230 + 8) (Ptx.get64 s1 224)
  Ptx.store t1 launch.outputBase (Ptx.get64 s2 230 + 16) (Ptx.get64 s2 228)

def ptxArithmetic (launch : Launch) (state : Ptx.State) : Ptx.State :=
  ptxArithmeticOutput launch (ptxArithmeticC2 launch (ptxArithmeticC1 launch
    (ptxArithmeticC0 launch (ptxArithmeticBase launch state))))

def ptxRunThread (launch : Launch) (thread : Ptx.Thread) : Option FieldExt :=
  if launch.lowHalf = 0 then
    none
  else if launch.outputCount ≤ thread.index then
    none
  else
    let state := ptxPrepare launch thread
    let state := ptxAddress state
    let state := ptxArithmetic launch state
    some (Ptx.outputExt state thread.index)

def ptxZeroAddress (launch : Launch) (thread : Ptx.Thread) : Nat :=
  ((thread.index % launch.lowHalf + launch.lowHalf * (thread.index / launch.lowHalf) * 2) %
      Word.base * 24 + launch.inputBase) % Word.base

def ptxOneAddress (launch : Launch) (thread : Ptx.Thread) : Nat :=
  (launch.lowHalf * 24 + ptxZeroAddress launch thread) % Word.base

theorem ptxZeroAddress_fold (launch : Launch) (thread : Ptx.Thread) :
    ((thread.index % launch.lowHalf + launch.lowHalf * (thread.index / launch.lowHalf) * 2) %
        Word.base * 24 + launch.inputBase) % Word.base = ptxZeroAddress launch thread := rfl

theorem ptxOneAddress_fold (launch : Launch) (thread : Ptx.Thread) :
    (launch.lowHalf * 24 + ptxZeroAddress launch thread) % Word.base =
      ptxOneAddress launch thread := rfl

private theorem ptxZeroAddress_raw_fold (launch : Launch) (thread : Ptx.Thread) :
    ((thread.index % launch.lowHalf +
          launch.lowHalf * (thread.index / launch.lowHalf) % Word.base * 2 ^ 1 % Word.base) %
        Word.base * 24 + launch.inputBase) % Word.base = ptxZeroAddress launch thread := by
  simp only [ptxZeroAddress, Nat.pow_one, Nat.add_mod, Nat.mul_mod, Nat.mod_mod]

private theorem ptxOneAddress_raw_fold (launch : Launch) (thread : Ptx.Thread) :
    (launch.lowHalf * 24 +
        ((thread.index % launch.lowHalf +
            launch.lowHalf * (thread.index / launch.lowHalf) % Word.base * 2 ^ 1 % Word.base) %
          Word.base * 24 + launch.inputBase) % Word.base) % Word.base =
      ptxOneAddress launch thread := by
  rw [ptxZeroAddress_raw_fold]
  rfl

def ptxZeroAddressPTX (launch : Launch) (thread : Ptx.Thread) : Nat :=
  if launch.lowHalf &&& 18446744069414584320 = 0 then
    (((thread.index + 0x100000000 -
        (thread.index / (launch.lowHalf % 0x100000000) *
          (launch.lowHalf % 0x100000000)) % 0x100000000) % 0x100000000 +
        launch.lowHalf * (thread.index / (launch.lowHalf % 0x100000000)) * 2) %
      Word.base * 24 + launch.inputBase) % Word.base
  else
    ((Word.wrapSub thread.index ((thread.index / launch.lowHalf * launch.lowHalf) % Word.base) +
        launch.lowHalf * (thread.index / launch.lowHalf) * 2) %
      Word.base * 24 + launch.inputBase) % Word.base

def ptxOneAddressPTX (launch : Launch) (thread : Ptx.Thread) : Nat :=
  (launch.lowHalf * 24 + ptxZeroAddressPTX launch thread) % Word.base

theorem and_high_zero_lt_32 {x : Nat} (hx : x < Word.base)
    (h : x &&& 18446744069414584320 = 0) : x < 0x100000000 := by
  have hx64 : x < 2 ^ 64 := by
    simpa [Word.base] using hx
  change x < 2 ^ 32
  apply Nat.lt_pow_two_of_testBit
  intro i hi
  by_cases hi64 : i ≥ 64
  · apply Nat.testBit_lt_two_pow
    exact Nat.lt_of_lt_of_le hx64 (Nat.pow_le_pow_right Nat.zero_lt_two hi64)
  · have hsub : i - 32 < 32 := by omega
    have hmask : (18446744069414584320 : Nat).testBit i = true := by
      change ((2 ^ 32 - 1) * 2 ^ 32).testBit i = true
      rw [Nat.testBit_mul_two_pow, Nat.testBit_two_pow_sub_one]
      simp [hi, hsub]
    have hb := congrArg (fun y : Nat => y.testBit i) h
    simpa [Nat.testBit_and, hmask] using hb

theorem ptxAddress32_remainder (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000)
    (haddr : launch.lowHalf &&& 18446744069414584320 = 0) :
    (thread.index + 0x100000000 -
        (thread.index / (launch.lowHalf % 0x100000000) *
          (launch.lowHalf % 0x100000000)) % 0x100000000) % 0x100000000 =
      thread.index % launch.lowHalf := by
  have hlow := and_high_zero_lt_32 (Launch.lowHalf_lt_base launch valid) haddr
  rw [Nat.mod_eq_of_lt hlow]
  have hdiv : thread.index / launch.lowHalf * launch.lowHalf ≤ thread.index :=
    Nat.div_mul_le_self _ _
  have hprod : thread.index / launch.lowHalf * launch.lowHalf < 0x100000000 := by
    omega
  rw [Nat.mod_eq_of_lt hprod]
  rw [Nat.mul_comm (thread.index / launch.lowHalf) launch.lowHalf]
  have hmod := Nat.mod_add_div thread.index launch.lowHalf
  omega

theorem ptxAddress64_remainder (launch : Launch) (thread : Ptx.Thread)
    (h32 : thread.index < 0x100000000) :
    Word.wrapSub thread.index ((thread.index / launch.lowHalf * launch.lowHalf) % Word.base) =
      thread.index % launch.lowHalf := by
  have hdiv : thread.index / launch.lowHalf * launch.lowHalf ≤ thread.index :=
    Nat.div_mul_le_self _ _
  have hprod : thread.index / launch.lowHalf * launch.lowHalf < Word.base := by
    simp only [Word.base]
    omega
  rw [Nat.mod_eq_of_lt hprod]
  rw [Word.wrapSub_of_le (by simp only [Word.base]; omega) hdiv]
  rw [Nat.mul_comm (thread.index / launch.lowHalf) launch.lowHalf]
  have hmod := Nat.mod_add_div thread.index launch.lowHalf
  omega

theorem ptxZeroAddressPTX_eq (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    ptxZeroAddressPTX launch thread = ptxZeroAddress launch thread := by
  unfold ptxZeroAddressPTX ptxZeroAddress
  by_cases haddr : launch.lowHalf &&& 18446744069414584320 = 0
  · rw [ptxAddress32_remainder launch valid thread h32 haddr,
      Nat.mod_eq_of_lt (and_high_zero_lt_32 (Launch.lowHalf_lt_base launch valid) haddr), haddr]
    rfl
  · simp [haddr, ptxAddress64_remainder launch thread h32]

theorem ptxOneAddressPTX_eq (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    ptxOneAddressPTX launch thread = ptxOneAddress launch thread := by
  simp [ptxOneAddressPTX, ptxOneAddress, ptxZeroAddressPTX_eq launch valid thread h32]

def ptxArithmeticSSA (launch : Launch) (thread : Ptx.Thread) : FieldExt :=
  let zero0 := Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8)
  let zero1 := Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8)
  let zero2 := Ptx.inputWord launch ((ptxZeroAddress launch thread + 16 - launch.inputBase) / 8)
  let one0 := Ptx.inputWord launch ((ptxOneAddress launch thread + 0 - launch.inputBase) / 8)
  let one1 := Ptx.inputWord launch ((ptxOneAddress launch thread + 8 - launch.inputBase) / 8)
  let one2 := Ptx.inputWord launch ((ptxOneAddress launch thread + 16 - launch.inputBase) / 8)
  let d0 := Word.sub one0 zero0
  let d1 := Word.sub one1 zero1
  let d2 := Word.sub one2 zero2
  let x00 := Word.mul d0 (Field.toWord launch.challenge.c0)
  let x01 := Word.mul d0 (Field.toWord launch.challenge.c1)
  let x02 := Word.mul d0 (Field.toWord launch.challenge.c2)
  let x10 := Word.mul d1 (Field.toWord launch.challenge.c0)
  let x11 := Word.mul d1 (Field.toWord launch.challenge.c1)
  let x12 := Word.mul d1 (Field.toWord launch.challenge.c2)
  let x20 := Word.mul d2 (Field.toWord launch.challenge.c0)
  let x21 := Word.mul d2 (Field.toWord launch.challenge.c1)
  let x22 := Word.mul d2 (Field.toWord launch.challenge.c2)
  ⟨Field.ofWord (Word.add zero0 (Word.add x00 (Word.mul 2 (Word.add x12 x21)))),
   Field.ofWord (Word.add zero1 (Word.add (Word.add x01 x10) (Word.mul 2 x22))),
   Field.ofWord (Word.add zero2 (Word.add x02 (Word.add x11 x20)))⟩

theorem ptxZeroAddress_eq (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    ptxZeroAddress launch thread = launch.inputBase + 24 * Launch.zeroIndex launch thread.index := by
  have hindex := Launch.zeroIndex_lt_input launch valid hi
  have hbound : launch.inputBase + 24 * Launch.zeroIndex launch thread.index < Word.base := by
    have hbase := valid.inputAddress
    omega
  have hsmall : Launch.zeroIndex launch thread.index < Word.base := by omega
  have hsmall' : thread.index % launch.lowHalf +
      launch.lowHalf * (thread.index / launch.lowHalf) * 2 < Word.base := by
    simpa [Launch.zeroIndex, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hsmall
  have hbound' : launch.inputBase + 24 * (thread.index % launch.lowHalf +
      launch.lowHalf * (thread.index / launch.lowHalf) * 2) < Word.base := by
    simpa [Launch.zeroIndex, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hbound
  have hbound'' : (thread.index % launch.lowHalf +
      launch.lowHalf * (thread.index / launch.lowHalf) * 2) * 24 + launch.inputBase < Word.base := by
    simpa [Nat.add_comm, Nat.mul_comm] using hbound'
  rw [ptxZeroAddress, Nat.mod_eq_of_lt hsmall', Nat.mod_eq_of_lt hbound'']
  simp [Launch.zeroIndex]
  ac_rfl

theorem ptxOneAddress_eq (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    ptxOneAddress launch thread = launch.inputBase + 24 * Launch.oneIndex launch thread.index := by
  have hindex := Launch.oneIndex_lt_input launch valid hi
  have hbound : launch.inputBase + 24 * Launch.oneIndex launch thread.index < Word.base := by
    have hbase := valid.inputAddress
    omega
  have hbound' : launch.lowHalf * 24 + (launch.inputBase +
      24 * Launch.zeroIndex launch thread.index) < Word.base := by
    simpa [Launch.oneIndex, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm,
      Nat.mul_add] using hbound
  rw [ptxOneAddress, ptxZeroAddress_eq launch valid thread hi h32,
    Nat.mod_eq_of_lt hbound']
  simp [Launch.oneIndex, Nat.mul_add]
  ac_rfl

theorem ptxZeroWordIndex (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxZeroAddress launch thread + 0 - launch.inputBase) / 8 =
      3 * Launch.zeroIndex launch thread.index := by
  rw [ptxZeroAddress_eq launch valid thread hi h32]
  omega

theorem ptxZeroWordIndex8 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxZeroAddress launch thread + 8 - launch.inputBase) / 8 =
      3 * Launch.zeroIndex launch thread.index + 1 := by
  rw [ptxZeroAddress_eq launch valid thread hi h32]
  omega

theorem ptxZeroWordIndex16 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxZeroAddress launch thread + 16 - launch.inputBase) / 8 =
      3 * Launch.zeroIndex launch thread.index + 2 := by
  rw [ptxZeroAddress_eq launch valid thread hi h32]
  omega

theorem ptxOneWordIndex (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxOneAddress launch thread + 0 - launch.inputBase) / 8 =
      3 * Launch.oneIndex launch thread.index := by
  rw [ptxOneAddress_eq launch valid thread hi h32]
  omega

theorem ptxOneWordIndex8 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxOneAddress launch thread + 8 - launch.inputBase) / 8 =
      3 * Launch.oneIndex launch thread.index + 1 := by
  rw [ptxOneAddress_eq launch valid thread hi h32]
  omega

theorem ptxOneWordIndex16 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxOneAddress launch thread + 16 - launch.inputBase) / 8 =
      3 * Launch.oneIndex launch thread.index + 2 := by
  rw [ptxOneAddress_eq launch valid thread hi h32]
  omega

theorem ptxOutputAddress_eq (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) :
    (thread.index * 24 + launch.outputBase) % Word.base =
      launch.outputBase + 24 * thread.index := by
  have hbound : thread.index * 24 + launch.outputBase < Word.base := by
    have houtput := valid.outputAddress
    omega
  rw [Nat.mod_eq_of_lt hbound]
  ac_rfl

theorem ptxOutputWordIndex (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) :
    ((thread.index * 24 + launch.outputBase) % Word.base + 0 - launch.outputBase) / 8 =
      3 * thread.index := by
  rw [ptxOutputAddress_eq launch valid thread hi]
  omega

theorem ptxOutputWordIndex8 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) :
    ((thread.index * 24 + launch.outputBase) % Word.base + 8 - launch.outputBase) / 8 =
      3 * thread.index + 1 := by
  rw [ptxOutputAddress_eq launch valid thread hi]
  omega

theorem ptxOutputWordIndex16 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) :
    ((thread.index * 24 + launch.outputBase) % Word.base + 16 - launch.outputBase) / 8 =
      3 * thread.index + 2 := by
  rw [ptxOutputAddress_eq launch valid thread hi]
  omega

theorem ptxZeroWordIndex0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxZeroAddress launch thread - launch.inputBase) / 8 =
      3 * Launch.zeroIndex launch thread.index := by
  simpa only [Nat.add_zero] using ptxZeroWordIndex launch valid thread hi h32

theorem ptxOneWordIndex0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    (ptxOneAddress launch thread - launch.inputBase) / 8 =
      3 * Launch.oneIndex launch thread.index := by
  simpa only [Nat.add_zero] using ptxOneWordIndex launch valid thread hi h32

theorem ptxOutputWordIndex0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) :
    ((thread.index * 24 + launch.outputBase) % Word.base - launch.outputBase) / 8 =
      3 * thread.index := by
  simpa only [Nat.add_zero] using ptxOutputWordIndex launch valid thread hi

theorem word_highWord_pow_fold (hi : Nat) :
    hi / 2 ^ 32 = Word.highWord hi := by
  rfl

theorem word_lowWord_mask_fold (hi : Nat) :
    hi &&& 0xffffffff = Word.lowWord hi := by
  simpa [Word.lowWord] using (Nat.and_two_pow_sub_one_eq_mod hi 32)

theorem word_mul_fold (a b : Nat) :
    Word.reduce128 ((a * b) / Word.base) ((a * b) % Word.base) = Word.mul a b := rfl

theorem word_mul_two_fold (a : Nat) :
    Word.reduce128 ((2 * a) / Word.base) ((a * 2 ^ 1) % Word.base) = Word.mul 2 a := by
  simpa [Nat.mul_comm] using word_mul_fold 2 a

def ptxReduce128 (hi lo : Nat) : Nat :=
  let high := hi / 2 ^ 32
  let low := hi &&& 0xffffffff
  let t := Word.wrapSub lo high
  let borrow := if lo < high then Word.modulus else 0
  let m := Word.wrapSub ((hi * 2 ^ 32) % Word.base) low
  let r := (t + m) % Word.base
  let r := (r + borrow) % Word.base
  let adjusted := (r + if r < m then Word.epsilon else 0) % Word.base
  if adjusted > Word.modulus - 1 then (adjusted + Word.epsilon) % Word.base else adjusted

theorem ptxReduce128_fold (hi lo : Nat) :
    (let high := hi / 2 ^ 32
     let low := hi &&& 0xffffffff
     let t := Word.wrapSub lo high
     let borrow := if lo < high then Word.modulus else 0
     let m := Word.wrapSub ((hi * 2 ^ 32) % Word.base) low
     let r := (t + m) % Word.base
     let r := (r + borrow) % Word.base
     let adjusted := (r + if r < m then Word.epsilon else 0) % Word.base
     if adjusted > Word.modulus - 1 then (adjusted + Word.epsilon) % Word.base else adjusted) =
      ptxReduce128 hi lo := rfl

theorem ptx_shift_mod (hi : Nat) :
    (hi * 2 ^ 32) % Word.base =
      (hi % 0x100000000 * 0x100000000) % Word.base := by
  have h := Nat.div_add_mod hi 0x100000000
  have hq32 : (2 ^ 32 : Nat) = 0x100000000 := by decide
  have hbase : 0x100000000 * 0x100000000 = Word.base := by decide
  simp only [hq32]
  calc
    (hi * 2 ^ 32) % Word.base =
        ((0x100000000 * (hi / 0x100000000) + hi % 0x100000000) * 2 ^ 32) % Word.base := by
      rw [h]
    _ = (hi % 0x100000000 * 0x100000000 +
        (0x100000000 * (hi / 0x100000000)) * 0x100000000) % Word.base := by
      simp only [hq32, Nat.add_mul]
      ac_rfl
    _ = (hi % 0x100000000 * 0x100000000 + Word.base * (hi / 0x100000000)) % Word.base := by
      have hq : (0x100000000 * (hi / 0x100000000)) * 0x100000000 =
          Word.base * (hi / 0x100000000) := by
        calc
          (0x100000000 * (hi / 0x100000000)) * 0x100000000 =
              (0x100000000 * 0x100000000) * (hi / 0x100000000) := by ac_rfl
          _ = Word.base * (hi / 0x100000000) := by rw [hbase]
      rw [hq]
    _ = (hi % 0x100000000 * 0x100000000) % Word.base :=
      Nat.add_mul_mod_self_left _ _ _

theorem ptx_reduce_m (hi : Nat) :
    Word.wrapSub ((hi * 2 ^ 32) % Word.base) (hi % 0x100000000) =
      hi % 0x100000000 * 0x100000000 - hi % 0x100000000 := by
  rw [ptx_shift_mod]
  have hlow : hi % 0x100000000 < 0x100000000 := Nat.mod_lt _ (by decide)
  have hprod : hi % 0x100000000 * 0x100000000 < Word.base := by
    simp only [Word.base]
    exact (Nat.mul_lt_mul_right (by decide)).2 hlow
  rw [Nat.mod_eq_of_lt hprod]
  exact Word.wrapSub_of_le hprod (by omega)

theorem ptx_reduce_m_word (hi : Nat) :
    Word.wrapSub ((hi * 2 ^ 32) % Word.base) (Word.lowWord hi) =
      Word.lowWord hi * 0x100000000 - Word.lowWord hi := by
  exact ptx_reduce_m hi

theorem ptx_borrow_add (t m : Nat) :
    (((t + m) % Word.base + Word.modulus) % Word.base) =
      (Word.wrapSub t Word.epsilon + m) % Word.base := by
  simp only [Word.wrapSub, Word.modulus, Word.base, Word.epsilon]
  omega

theorem word_canonical_ptx_add_fold (x : Nat) (hx : x < Word.base) :
    (if x > Word.modulus - 1 then (x + Word.epsilon) % Word.base else x) =
      Word.canonical x := by
  by_cases h : x > Word.modulus - 1
  · have hge : x ≥ Word.modulus := by omega
    simp only [h, Word.canonical, hge, ↓reduceIte]
    simp only [Word.modulus, Word.base, Word.epsilon] at *
    have hcarry : x + 0xffffffff ≥ 0x10000000000000000 := by omega
    rw [Nat.mod_eq_sub_mod hcarry]
    omega
  · have hnot : ¬x ≥ Word.modulus := by
      simp only [Word.modulus] at *
      omega
    simp [Word.canonical, h, hnot]

def ptxReduce128BV (hi lo : BitVec 64) : BitVec 64 :=
  let high := hi >>> 32
  let low := hi &&& 0xffffffff#64
  let t := lo - high
  let borrow := if lo < high then 0xffffffff00000001#64 else 0#64
  let m := (hi <<< 32) - low
  let r := t + m + borrow
  let adjusted := r + if r < m then 0xffffffff#64 else 0#64
  if adjusted > 0xffffffff00000000#64 then adjusted + 0xffffffff#64 else adjusted

def wordReduce128BV (hi lo : BitVec 64) : BitVec 64 :=
  let high := hi >>> 32
  let low := hi &&& 0xffffffff#64
  let d := if lo < high then (lo - high) - 0xffffffff#64 else lo - high
  let m := (low <<< 32) - low
  let r := d + m
  let adjusted := r + if r < m then 0xffffffff#64 else 0#64
  if adjusted ≥ 0xffffffff00000001#64 then adjusted - 0xffffffff00000001#64 else adjusted

theorem ptxReduce128BV_eq (hi lo : BitVec 64) :
    ptxReduce128BV hi lo = wordReduce128BV hi lo := by
  unfold ptxReduce128BV wordReduce128BV
  bv_decide

theorem toNat_sub_source (x y : BitVec 64) :
    (x - y).toNat = (x.toNat + 2 ^ 64 - y.toNat) % 2 ^ 64 := by
  rw [BitVec.toNat_sub, ← Nat.sub_add_comm (Nat.le_of_lt (BitVec.isLt y))]
  ac_rfl

private theorem ptxReduce128_eq_borrow (hi lo : Nat) (h : lo < Word.highWord hi) :
    ptxReduce128 hi lo = Word.reduce128 hi lo := by
  simp only [ptxReduce128, word_highWord_pow_fold, word_lowWord_mask_fold,
    ptx_reduce_m_word, h, ↓reduceIte]
  rw [ptx_borrow_add]
  refine (word_canonical_ptx_add_fold _ (Nat.mod_lt _ (by decide))).trans ?_
  simp only [Word.reduce128, Word.reduceDifference, Word.adjust, h, ↓reduceIte,
    Word.wrapSub, Nat.sub_zero, Nat.add_zero, Nat.mod_mod, Nat.add_mod_right]
  rfl

private theorem word_wrapSub_zero_of_lt {a : Nat} (ha : a < Word.base) :
    Word.wrapSub a 0 = a := by
  simp [Word.wrapSub, Nat.mod_eq_of_lt ha]

private theorem ptx_no_borrow_add (t m : Nat) :
    (((t + m) % Word.base + 0) % Word.base) = (t + m) % Word.base := by
  rw [Nat.add_zero, Nat.mod_mod]

private theorem ptxReduce128_eq_no_borrow (hi lo : Nat) (h : ¬lo < Word.highWord hi) :
    ptxReduce128 hi lo = Word.reduce128 hi lo := by
  simp only [ptxReduce128, word_highWord_pow_fold, word_lowWord_mask_fold,
    ptx_reduce_m_word, h, ↓reduceIte]
  rw [ptx_no_borrow_add]
  refine (word_canonical_ptx_add_fold _ (Nat.mod_lt _ (by decide))).trans ?_
  have hd : Word.reduceDifference hi lo = Word.wrapSub lo (Word.highWord hi) := by
    simp only [Word.reduceDifference, h, ↓reduceIte]
    exact word_wrapSub_zero_of_lt (Word.wrapSub_lt _ _)
  change Word.canonical (Word.adjust (Word.wrapSub lo (Word.highWord hi))
    (Word.lowWord hi * 0x100000000 - Word.lowWord hi)) = Word.reduce128 hi lo
  rw [← hd]
  rfl

theorem ptxReduce128_eq (hi lo : Nat) :
    ptxReduce128 hi lo = Word.reduce128 hi lo := by
  by_cases h : lo < Word.highWord hi
  · exact ptxReduce128_eq_borrow hi lo h
  · exact ptxReduce128_eq_no_borrow hi lo h

theorem word_sub_ptx_fold (a b : Nat) :
    (if a < b then (Word.wrapSub a b + Word.modulus) % Word.base
      else Word.wrapSub a b) = Word.sub a b := by
  by_cases h : a < b
  · simp (config := { decide := true }) only [Word.sub, Word.wrapSub, Word.modulus, Word.base, Word.epsilon,
      h, ↓reduceIte]
    omega
  · simp (config := { decide := true }) only [Word.sub, Word.wrapSub, Word.modulus, Word.base, Word.epsilon,
      h, ↓reduceIte]

theorem word_wrapSub_epsilon_ptx (x : Nat) :
    (x + Word.modulus) % Word.base = Word.wrapSub x Word.epsilon := by
  simp only [Word.wrapSub, Word.modulus, Word.base, Word.epsilon]
  omega

theorem word_add_ptx_fold (a b : Nat) :
    (if a < Word.wrapSub Word.modulus b then
        (Word.wrapSub a (Word.wrapSub Word.modulus b) + Word.modulus) % Word.base
      else Word.wrapSub a (Word.wrapSub Word.modulus b)) = Word.add a b := by
  by_cases h : a < Word.wrapSub Word.modulus b
  · simp (config := { decide := true }) only [Word.add, h, ↓reduceIte]
    exact word_wrapSub_epsilon_ptx _
  · simp (config := { decide := true }) only [Word.add, h, ↓reduceIte]

def ptxSubPTX (a b : Nat) : Nat :=
  if a < b then (Word.wrapSub a b + Word.modulus) % Word.base else Word.wrapSub a b

theorem ptxSubPTX_eq (a b : Nat) : ptxSubPTX a b = Word.sub a b :=
  word_sub_ptx_fold a b

def ptxAddPTX (a b : Nat) : Nat :=
  if a < Word.wrapSub Word.modulus b then
    (Word.wrapSub a (Word.wrapSub Word.modulus b) + Word.modulus) % Word.base
  else
    Word.wrapSub a (Word.wrapSub Word.modulus b)

theorem ptxAddPTX_eq (a b : Nat) : ptxAddPTX a b = Word.add a b :=
  word_add_ptx_fold a b

def ptxMulPTX (a b : Nat) : Nat :=
  let high := (a * b) / Word.base
  let low := (a * b) % Word.base
  let highWord := high / 2 ^ 32
  let lowWord := high &&& 0xffffffff
  let t := Word.wrapSub low highWord
  let borrow := if low < highWord then Word.modulus else 0
  let m := Word.wrapSub ((high * 2 ^ 32) % Word.base) lowWord
  let r := (t + m) % Word.base
  let r := (r + borrow) % Word.base
  let adjusted := (r + if r < m then Word.epsilon else 0) % Word.base
  if adjusted > Word.modulus - 1 then (adjusted + Word.epsilon) % Word.base else adjusted

theorem ptxMulPTX_eq (a b : Nat) : ptxMulPTX a b = Word.mul a b := by
  rw [ptxMulPTX, ptxReduce128_fold, ptxReduce128_eq, word_mul_fold]

def ptxMulTwoPTX (a : Nat) : Nat :=
  ptxReduce128 ((2 * a) / Word.base) ((a * 2 ^ 1) % Word.base)

theorem ptxMulTwoPTX_eq (a : Nat) : ptxMulTwoPTX a = Word.mul 2 a := by
  rw [ptxMulTwoPTX, ptxReduce128_eq, word_mul_two_fold]

set_option maxRecDepth 1000000 in
theorem Ptx.get64_subField (state : State) (base predicate left right : Nat)
    (hl : left ≠ base) (hr : right ≠ base) :
    get64 (subField state base predicate left right) (base + 2) =
      ptxSubPTX (get64 state left) (get64 state right) := by
  simp (config := { decide := true }) only [subField, sub64, add64, selp64, setPredicate,
    getPredicate, set64, get64, eval64, hl, hr, Nat.ne_add_one, decide_eq_true_eq, ↓reduceIte]
  rfl

theorem Ptx.get64_subField_ne (state : State) (base predicate left right r : Nat)
    (h0 : r ≠ base) (h1 : r ≠ base + 1) (h2 : r ≠ base + 2) :
    get64 (subField state base predicate left right) r = get64 state r := by
  simp [subField, sub64, add64, selp64, setPredicate, getPredicate, set64, get64, eval64, h0, h1, h2]

set_option maxRecDepth 1000000 in
theorem Ptx.get64_addField (state : State) (base predicate left right : Nat)
    (hl : left ≠ base) (hl1 : left ≠ base + 1) :
    get64 (addField state base predicate left right) (base + 3) =
      ptxAddPTX (get64 state left) (get64 state right) := by
  simp (config := { decide := true }) only [addField, sub64, add64, selp64, setPredicate,
    getPredicate, set64, get64, eval64, hl, hl1, Nat.ne_add_one, decide_eq_true_eq, ↓reduceIte]
  rfl

theorem Ptx.get64_addField_ne (state : State) (base predicate left right r : Nat)
    (h0 : r ≠ base) (h1 : r ≠ base + 1) (h2 : r ≠ base + 2) (h3 : r ≠ base + 3) :
    get64 (addField state base predicate left right) r = get64 state r := by
  simp [addField, sub64, add64, selp64, setPredicate, getPredicate, set64, get64, eval64, h0, h1, h2, h3]

set_option maxRecDepth 1000000 in
theorem Ptx.get64_mulField (state : State) (base predicate left right : Nat)
    (hl : left ≠ base) (hr : right ≠ base) :
    get64 (mulField state base predicate left right) (base + 13) =
      ptxMulPTX (get64 state left) (get64 state right) := by
  have ne_add (a k : Nat) (hk : 0 < k) : a ≠ a + k := by omega
  simp (config := { decide := true, maxSteps := 1000000 }) (discharger := omega) only [mulField, mulHi64, mulLo64, shr64Imm,
    and64ImmRight, sub64, add64, selp64, shl64Imm, get64_set64, get64_setPredicate,
    getPredicate_set64, getPredicate_setPredicate, eval64, hl, hr, Nat.add_left_inj,
    ne_add, decide_eq_true_eq, ↓reduceIte]
  simp only [ptxMulPTX]
  ac_rfl

theorem Ptx.get64_mulField_ne (state : State) (base predicate left right r : Nat)
    (h0 : r ≠ base) (h1 : r ≠ base + 1) (h2 : r ≠ base + 2) (h3 : r ≠ base + 3)
    (h4 : r ≠ base + 4) (h5 : r ≠ base + 5) (h6 : r ≠ base + 6) (h7 : r ≠ base + 7)
    (h8 : r ≠ base + 8) (h9 : r ≠ base + 9) (h10 : r ≠ base + 10)
    (h11 : r ≠ base + 11) (h12 : r ≠ base + 12) (h13 : r ≠ base + 13) :
    get64 (mulField state base predicate left right) r = get64 state r := by
  simp [mulField, mulHi64, mulLo64, shr64Imm, and64ImmRight, sub64, add64,
    selp64, shl64Imm, get64_set64, get64_setPredicate, eval64,
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13]

set_option maxRecDepth 1000000 in
theorem Ptx.get64_doubleField (state : State) (base predicate source : Nat)
    (hs : source ≠ base) (h84 : get64 state 84 = 2) :
    get64 (doubleField state base predicate source) (base + 13) =
      ptxMulTwoPTX (get64 state source) := by
  have ne_add (a k : Nat) (hk : 0 < k) : a ≠ a + k := by omega
  simp (config := { decide := true, maxSteps := 1000000 }) (discharger := omega) only [doubleField, mulHi64, shl64Imm, shr64Imm,
    and64ImmRight, sub64, add64, selp64, shl64Imm, get64_set64, get64_setPredicate,
    getPredicate_set64, getPredicate_setPredicate, eval64, hs, h84, Nat.add_left_inj,
    ne_add, decide_eq_true_eq, ↓reduceIte]
  simp only [ptxMulTwoPTX, ptxReduce128]
  ac_rfl

theorem Ptx.get64_doubleField_ne (state : State) (base predicate source r : Nat)
    (h0 : r ≠ base) (h1 : r ≠ base + 1) (h2 : r ≠ base + 2) (h3 : r ≠ base + 3)
    (h4 : r ≠ base + 4) (h5 : r ≠ base + 5) (h6 : r ≠ base + 6) (h7 : r ≠ base + 7)
    (h8 : r ≠ base + 8) (h9 : r ≠ base + 9) (h10 : r ≠ base + 10)
    (h11 : r ≠ base + 11) (h12 : r ≠ base + 12) (h13 : r ≠ base + 13) :
    get64 (doubleField state base predicate source) r = get64 state r := by
  simp [doubleField, mulHi64, shl64Imm, shr64Imm, and64ImmRight, sub64, add64,
    selp64, get64_set64, get64_setPredicate, eval64,
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13]

def ptxArithmeticPTX (launch : Launch) (thread : Ptx.Thread) : FieldExt :=
  let zero0 := Ptx.inputWord launch ((ptxZeroAddressPTX launch thread + 0 - launch.inputBase) / 8)
  let zero1 := Ptx.inputWord launch ((ptxZeroAddressPTX launch thread + 8 - launch.inputBase) / 8)
  let zero2 := Ptx.inputWord launch ((ptxZeroAddressPTX launch thread + 16 - launch.inputBase) / 8)
  let one0 := Ptx.inputWord launch ((ptxOneAddressPTX launch thread + 0 - launch.inputBase) / 8)
  let one1 := Ptx.inputWord launch ((ptxOneAddressPTX launch thread + 8 - launch.inputBase) / 8)
  let one2 := Ptx.inputWord launch ((ptxOneAddressPTX launch thread + 16 - launch.inputBase) / 8)
  let d0 := ptxSubPTX one0 zero0
  let d1 := ptxSubPTX one1 zero1
  let d2 := ptxSubPTX one2 zero2
  let x00 := ptxMulPTX d0 (Field.toWord launch.challenge.c0)
  let x01 := ptxMulPTX d0 (Field.toWord launch.challenge.c1)
  let x02 := ptxMulPTX d0 (Field.toWord launch.challenge.c2)
  let x10 := ptxMulPTX d1 (Field.toWord launch.challenge.c0)
  let x11 := ptxMulPTX d1 (Field.toWord launch.challenge.c1)
  let x12 := ptxMulPTX d1 (Field.toWord launch.challenge.c2)
  let x20 := ptxMulPTX d2 (Field.toWord launch.challenge.c0)
  let x21 := ptxMulPTX d2 (Field.toWord launch.challenge.c1)
  let x22 := ptxMulPTX d2 (Field.toWord launch.challenge.c2)
  let x2_12_21 := ptxMulTwoPTX (ptxAddPTX x12 x21)
  let x2_22 := ptxMulTwoPTX x22
  ⟨Field.ofWord (ptxAddPTX zero0 (ptxAddPTX x00 x2_12_21)),
   Field.ofWord (ptxAddPTX zero1 (ptxAddPTX (ptxAddPTX x01 x10) x2_22)),
   Field.ofWord (ptxAddPTX zero2 (ptxAddPTX x02 (ptxAddPTX x11 x20)))⟩

theorem ptxArithmeticPTX_eq_ssa (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    ptxArithmeticPTX launch thread = ptxArithmeticSSA launch thread := by
  simp only [ptxArithmeticPTX, ptxArithmeticSSA, ptxSubPTX_eq, ptxAddPTX_eq, ptxMulPTX_eq,
    ptxMulTwoPTX_eq]
  rw [ptxZeroAddressPTX_eq launch valid thread h32,
    ptxOneAddressPTX_eq launch valid thread h32]
  all_goals assumption

theorem ptxArithmeticSSA_correct (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) :
    ptxArithmeticSSA launch thread =
      FieldExt.foldFused
        (launch.input[Launch.zeroIndex launch thread.index]'
          (Launch.zeroIndex_lt_input launch valid hi))
        (launch.input[Launch.oneIndex launch thread.index]'
          (Launch.oneIndex_lt_input launch valid hi))
        launch.challenge := by
  unfold ptxArithmeticSSA
  simp only []
  rw [ptxZeroWordIndex launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]
  simp only [Ptx.inputWord_c0 launch _ (Launch.zeroIndex_lt_input launch valid hi),
    Ptx.inputWord_c1 launch _ (Launch.zeroIndex_lt_input launch valid hi),
    Ptx.inputWord_c2 launch _ (Launch.zeroIndex_lt_input launch valid hi),
    Ptx.inputWord_c0 launch _ (Launch.oneIndex_lt_input launch valid hi),
    Ptx.inputWord_c1 launch _ (Launch.oneIndex_lt_input launch valid hi),
    Ptx.inputWord_c2 launch _ (Launch.oneIndex_lt_input launch valid hi)]
  simp only [Field.toWord_word_sub, Field.toWord_word_mul, Field.toWord_word_mul_two,
    Field.toWord_word_add, Field.ofWord_toWord, FieldExt.foldFused]

def ptxRunThreadSSA (launch : Launch) (thread : Ptx.Thread) : Option FieldExt :=
  if launch.lowHalf = 0 then
    none
  else if launch.outputCount ≤ thread.index then
    none
  else
    some (ptxArithmeticSSA launch thread)

private theorem get64_subField_25 (s : Ptx.State) :
    Ptx.get64 (Ptx.subField s 25 4 23 24) 27 = ptxSubPTX (Ptx.get64 s 23) (Ptx.get64 s 24) :=
  Ptx.get64_subField s 25 4 23 24 (by decide) (by decide)
private theorem get64_subField_30 (s : Ptx.State) :
    Ptx.get64 (Ptx.subField s 30 5 28 29) 32 = ptxSubPTX (Ptx.get64 s 28) (Ptx.get64 s 29) :=
  Ptx.get64_subField s 30 5 28 29 (by decide) (by decide)
private theorem get64_subField_35 (s : Ptx.State) :
    Ptx.get64 (Ptx.subField s 35 6 33 34) 37 = ptxSubPTX (Ptx.get64 s 33) (Ptx.get64 s 34) :=
  Ptx.get64_subField s 35 6 33 34 (by decide) (by decide)

private theorem get64_mulField_38 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 38 7 27 3) 51 = ptxMulPTX (Ptx.get64 s 27) (Ptx.get64 s 3) :=
  Ptx.get64_mulField s 38 7 27 3 (by decide) (by decide)
private theorem get64_mulField_52 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 52 10 32 1) 65 = ptxMulPTX (Ptx.get64 s 32) (Ptx.get64 s 1) :=
  Ptx.get64_mulField s 52 10 32 1 (by decide) (by decide)
private theorem get64_mulField_66 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 66 13 37 2) 79 = ptxMulPTX (Ptx.get64 s 37) (Ptx.get64 s 2) :=
  Ptx.get64_mulField s 66 13 37 2 (by decide) (by decide)
private theorem get64_mulField_103 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 103 21 27 2) 116 = ptxMulPTX (Ptx.get64 s 27) (Ptx.get64 s 2) :=
  Ptx.get64_mulField s 103 21 27 2 (by decide) (by decide)
private theorem get64_mulField_117 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 117 24 32 3) 130 = ptxMulPTX (Ptx.get64 s 32) (Ptx.get64 s 3) :=
  Ptx.get64_mulField s 117 24 32 3 (by decide) (by decide)
private theorem get64_mulField_135 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 135 28 37 1) 148 = ptxMulPTX (Ptx.get64 s 37) (Ptx.get64 s 1) :=
  Ptx.get64_mulField s 135 28 37 1 (by decide) (by decide)
private theorem get64_mulField_167 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 167 35 27 1) 180 = ptxMulPTX (Ptx.get64 s 27) (Ptx.get64 s 1) :=
  Ptx.get64_mulField s 167 35 27 1 (by decide) (by decide)
private theorem get64_mulField_181 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 181 38 32 2) 194 = ptxMulPTX (Ptx.get64 s 32) (Ptx.get64 s 2) :=
  Ptx.get64_mulField s 181 38 32 2 (by decide) (by decide)
private theorem get64_mulField_195 (s : Ptx.State) :
    Ptx.get64 (Ptx.mulField s 195 41 37 3) 208 = ptxMulPTX (Ptx.get64 s 37) (Ptx.get64 s 3) :=
  Ptx.get64_mulField s 195 41 37 3 (by decide) (by decide)

private theorem get64_doubleField_85 (s : Ptx.State) (h84 : Ptx.get64 s 84 = 2) :
    Ptx.get64 (Ptx.doubleField s 85 17 83) 98 = ptxMulTwoPTX (Ptx.get64 s 83) :=
  Ptx.get64_doubleField s 85 17 83 (by decide) h84
private theorem get64_doubleField_85_set (s : Ptx.State) :
    Ptx.get64 (Ptx.doubleField (Ptx.set64 s 84 2) 85 17 83) 98 =
      ptxMulTwoPTX (Ptx.get64 s 83) := by
  rw [Ptx.get64_doubleField _ _ _ _ (by decide) (by simp only [Ptx.get64_set64, ↓reduceIte])]
  simp only [Ptx.get64_set64, if_neg (by decide : (83 : Nat) ≠ 84)]
private theorem get64_84_doubleField_85_set (s : Ptx.State) :
    Ptx.get64 (Ptx.doubleField (Ptx.set64 s 84 2) 85 17 83) 84 = 2 := by
  simp (config := { decide := true }) only [Ptx.get64_doubleField_ne, Ptx.get64_set64, ↓reduceIte, ite_true]
private theorem get64_doubleField_149 (s : Ptx.State) (h84 : Ptx.get64 s 84 = 2) :
    Ptx.get64 (Ptx.doubleField s 149 29 148) 162 = ptxMulTwoPTX (Ptx.get64 s 148) :=
  Ptx.get64_doubleField s 149 29 148 (by decide) h84
private theorem get64_doubleField_149_chain (s : Ptx.State) :
    Ptx.get64 (Ptx.doubleField
      (Ptx.mulField
        (Ptx.addField
          (Ptx.mulField
            (Ptx.mulField
              (Ptx.addField (Ptx.doubleField (Ptx.set64 s 84 2) 85 17 83) 99 20 51 98)
              103 21 27 2)
            117 24 32 3)
          131 27 116 130)
        135 28 37 1)
      149 29 148) 162 =
      ptxMulTwoPTX (Ptx.get64
        (Ptx.mulField
          (Ptx.addField
            (Ptx.mulField
              (Ptx.mulField
                (Ptx.addField (Ptx.doubleField (Ptx.set64 s 84 2) 85 17 83) 99 20 51 98)
                103 21 27 2)
              117 24 32 3)
            131 27 116 130)
          135 28 37 1) 148) := by
  apply Ptx.get64_doubleField _ _ _ _ (by decide)
  simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, get64_84_doubleField_85_set]

private theorem get64_addField_80 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 80 16 65 79) 83 = ptxAddPTX (Ptx.get64 s 65) (Ptx.get64 s 79) :=
  Ptx.get64_addField s 80 16 65 79 (by decide) (by decide)
private theorem get64_addField_99 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 99 20 51 98) 102 = ptxAddPTX (Ptx.get64 s 51) (Ptx.get64 s 98) :=
  Ptx.get64_addField s 99 20 51 98 (by decide) (by decide)
private theorem get64_addField_131 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 131 27 116 130) 134 = ptxAddPTX (Ptx.get64 s 116) (Ptx.get64 s 130) :=
  Ptx.get64_addField s 131 27 116 130 (by decide) (by decide)
private theorem get64_addField_163 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 163 34 134 162) 166 = ptxAddPTX (Ptx.get64 s 134) (Ptx.get64 s 162) :=
  Ptx.get64_addField s 163 34 134 162 (by decide) (by decide)
private theorem get64_addField_209 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 209 44 194 208) 212 = ptxAddPTX (Ptx.get64 s 194) (Ptx.get64 s 208) :=
  Ptx.get64_addField s 209 44 194 208 (by decide) (by decide)
private theorem get64_addField_213 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 213 45 180 212) 216 = ptxAddPTX (Ptx.get64 s 180) (Ptx.get64 s 212) :=
  Ptx.get64_addField s 213 45 180 212 (by decide) (by decide)
private theorem get64_addField_217 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 217 46 24 102) 220 = ptxAddPTX (Ptx.get64 s 24) (Ptx.get64 s 102) :=
  Ptx.get64_addField s 217 46 24 102 (by decide) (by decide)
private theorem get64_addField_221 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 221 47 29 166) 224 = ptxAddPTX (Ptx.get64 s 29) (Ptx.get64 s 166) :=
  Ptx.get64_addField s 221 47 29 166 (by decide) (by decide)
private theorem get64_addField_225 (s : Ptx.State) :
    Ptx.get64 (Ptx.addField s 225 48 34 216) 228 = ptxAddPTX (Ptx.get64 s 34) (Ptx.get64 s 216) :=
  Ptx.get64_addField s 225 48 34 216 (by decide) (by decide)

private theorem ptxIndexProduct_lt_base (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    launch.lowHalf * (thread.index / launch.lowHalf) < Word.base := by
  have hdiv : launch.lowHalf * (thread.index / launch.lowHalf) ≤ thread.index := by
    simpa [Nat.mul_comm] using Nat.div_mul_le_self thread.index launch.lowHalf
  change launch.lowHalf * (thread.index / launch.lowHalf) < 0x10000000000000000
  omega

private theorem ptxDoubleIndex_lt_base (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    launch.lowHalf * (thread.index / launch.lowHalf) * 2 ^ 1 < Word.base := by
  have hdiv : launch.lowHalf * (thread.index / launch.lowHalf) ≤ thread.index := by
    simpa [Nat.mul_comm] using Nat.div_mul_le_self thread.index launch.lowHalf
  change launch.lowHalf * (thread.index / launch.lowHalf) * 2 ^ 1 < 0x10000000000000000
  omega

private theorem ptxInputZero0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (z0 : Nat)
    (hz0 : Ptx.inputWord launch (3 * Launch.zeroIndex launch thread.index) = z0) :
    Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8) = z0 := by
  simp only [Nat.add_zero, ptxZeroWordIndex0 launch valid thread hi h32]
  exact hz0
private theorem ptxInputZero1 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (z1 : Nat)
    (hz1 : Ptx.inputWord launch (3 * Launch.zeroIndex launch thread.index + 1) = z1) :
    Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8) = z1 := by
  rw [ptxZeroWordIndex8 launch valid thread hi h32]
  exact hz1
private theorem ptxInputZero2 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (z2 : Nat)
    (hz2 : Ptx.inputWord launch (3 * Launch.zeroIndex launch thread.index + 2) = z2) :
    Ptx.inputWord launch ((ptxZeroAddress launch thread + 16 - launch.inputBase) / 8) = z2 := by
  rw [ptxZeroWordIndex16 launch valid thread hi h32]
  exact hz2
private theorem ptxInputOne0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (o0 : Nat)
    (ho0 : Ptx.inputWord launch (3 * Launch.oneIndex launch thread.index) = o0) :
    Ptx.inputWord launch ((ptxOneAddress launch thread + 0 - launch.inputBase) / 8) = o0 := by
  simp only [Nat.add_zero, ptxOneWordIndex0 launch valid thread hi h32]
  exact ho0
private theorem ptxInputOne1 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (o1 : Nat)
    (ho1 : Ptx.inputWord launch (3 * Launch.oneIndex launch thread.index + 1) = o1) :
    Ptx.inputWord launch ((ptxOneAddress launch thread + 8 - launch.inputBase) / 8) = o1 := by
  rw [ptxOneWordIndex8 launch valid thread hi h32]
  exact ho1
private theorem ptxInputOne2 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (o2 : Nat)
    (ho2 : Ptx.inputWord launch (3 * Launch.oneIndex launch thread.index + 2) = o2) :
    Ptx.inputWord launch ((ptxOneAddress launch thread + 16 - launch.inputBase) / 8) = o2 := by
  rw [ptxOneWordIndex16 launch valid thread hi h32]
  exact ho2

private theorem ptxArithmeticAddress_values (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (h11 : Ptx.get64 s 11 = launch.inputBase)
    (h14 : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    let s0 := ptxArithmeticAddress s
    (Ptx.get64 s0 21 = ptxZeroAddress launch thread ∧
     Ptx.get64 s0 22 = ptxOneAddress launch thread) := by
  dsimp only [ptxArithmeticAddress, Ptx.mulLo64, Ptx.shl64Imm, Ptx.add64, Ptx.cvta,
    Ptx.madLo64ImmRight, Ptx.eval64]
  constructor
  · simp (config := { decide := true }) only [Ptx.get64_set64, h4, h11, h14, h231, h232,
      ptxZeroAddress_raw_fold, ↓reduceIte]
  · simp (config := { decide := true }) only [Ptx.get64_set64, h4, h11, h14, h231, h232,
      ptxZeroAddress_raw_fold, ptxOneAddress_fold, ↓reduceIte]

private theorem ptxArithmeticLoad0_values (launch : Launch) (s : Ptx.State) :
    let a := ptxArithmeticAddress s
    let s0 := ptxArithmeticLoad0 launch a
    (Ptx.get64 s0 24 = Ptx.inputWord launch ((Ptx.get64 a 21 + 0 - launch.inputBase) / 8) ∧
     Ptx.get64 s0 27 = ptxSubPTX
       (Ptx.inputWord launch ((Ptx.get64 a 22 + 0 - launch.inputBase) / 8))
       (Ptx.inputWord launch ((Ptx.get64 a 21 + 0 - launch.inputBase) / 8))) := by
  dsimp only [ptxArithmeticLoad0, Ptx.ldGlobalOffset, Ptx.load]
  constructor
  · simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne,
      Nat.add_zero, ↓reduceIte]
  · simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]
    rw [get64_subField_25]
    simp (config := { decide := true }) only [Ptx.get64_set64, Nat.add_zero, ↓reduceIte]

private theorem ptxArithmeticLoad0_preserved (launch : Launch) (s : Ptx.State) :
    let a := ptxArithmeticAddress s
    let s0 := ptxArithmeticLoad0 launch a
    (Ptx.get64 s0 1 = Ptx.get64 a 1 ∧
     Ptx.get64 s0 2 = Ptx.get64 a 2 ∧
     Ptx.get64 s0 3 = Ptx.get64 a 3 ∧
     Ptx.get64 s0 4 = Ptx.get64 a 4 ∧
     Ptx.get64 s0 12 = Ptx.get64 a 12 ∧
     Ptx.get64 s0 21 = Ptx.get64 a 21 ∧
     Ptx.get64 s0 22 = Ptx.get64 a 22) := by
  dsimp only [ptxArithmeticLoad0, Ptx.ldGlobalOffset, Ptx.load]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad1_zero (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad1 launch a) 29 =
      Ptx.inputWord launch ((Ptx.get64 a 21 + 8 - launch.inputBase) / 8) := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad1_diff (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad1 launch a) 32 = ptxSubPTX
      (Ptx.inputWord launch ((Ptx.get64 a 22 + 8 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((Ptx.get64 a 21 + 8 - launch.inputBase) / 8)) := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]
  rw [get64_subField_30]
  simp (config := { decide := true }) only [Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticLoad2_zero (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 34 =
      Ptx.inputWord launch ((Ptx.get64 a 21 + 16 - launch.inputBase) / 8) := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_diff (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 37 = ptxSubPTX
      (Ptx.inputWord launch ((Ptx.get64 a 22 + 16 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((Ptx.get64 a 21 + 16 - launch.inputBase) / 8)) := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]
  rw [get64_subField_35]
  simp (config := { decide := true }) only [Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticLoad1_preserved (launch : Launch) (a : Ptx.State) :
    (Ptx.get64 (ptxArithmeticLoad1 launch a) 1 = Ptx.get64 a 1 ∧
     Ptx.get64 (ptxArithmeticLoad1 launch a) 2 = Ptx.get64 a 2 ∧
     Ptx.get64 (ptxArithmeticLoad1 launch a) 3 = Ptx.get64 a 3 ∧
     Ptx.get64 (ptxArithmeticLoad1 launch a) 4 = Ptx.get64 a 4 ∧
     Ptx.get64 (ptxArithmeticLoad1 launch a) 12 = Ptx.get64 a 12) := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad1_addresses (launch : Launch) (a : Ptx.State) :
    (Ptx.get64 (ptxArithmeticLoad1 launch a) 21 = Ptx.get64 a 21 ∧
     Ptx.get64 (ptxArithmeticLoad1 launch a) 22 = Ptx.get64 a 22) := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  constructor <;> simp (config := { decide := true }) only [
    Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_preserved (launch : Launch) (a : Ptx.State) :
    (Ptx.get64 (ptxArithmeticLoad2 launch a) 1 = Ptx.get64 a 1 ∧
     Ptx.get64 (ptxArithmeticLoad2 launch a) 2 = Ptx.get64 a 2 ∧
     Ptx.get64 (ptxArithmeticLoad2 launch a) 3 = Ptx.get64 a 3 ∧
     Ptx.get64 (ptxArithmeticLoad2 launch a) 4 = Ptx.get64 a 4 ∧
     Ptx.get64 (ptxArithmeticLoad2 launch a) 12 = Ptx.get64 a 12) := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_addresses (launch : Launch) (a : Ptx.State) :
    (Ptx.get64 (ptxArithmeticLoad2 launch a) 21 = Ptx.get64 a 21 ∧
     Ptx.get64 (ptxArithmeticLoad2 launch a) 22 = Ptx.get64 a 22) := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  constructor <;> simp (config := { decide := true }) only [
    Ptx.get64_set64, Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad1_preserve27 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad1 launch a) 27 = Ptx.get64 a 27 := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_preserve27 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 27 = Ptx.get64 a 27 := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_preserve32 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 32 = Ptx.get64 a 32 := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad1_preserve24 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad1 launch a) 24 = Ptx.get64 a 24 := by
  dsimp only [ptxArithmeticLoad1, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_preserve24 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 24 = Ptx.get64 a 24 := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

private theorem ptxArithmeticLoad2_preserve29 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticLoad2 launch a) 29 = Ptx.get64 a 29 := by
  dsimp only [ptxArithmeticLoad2, Ptx.ldGlobalOffset, Ptx.load]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_subField_ne, ↓reduceIte]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_diff27 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 27 = ptxSubPTX
      (Ptx.inputWord launch ((ptxOneAddress launch thread + 0 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8)) := by
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨_, h27⟩ := ptxArithmeticLoad0_values launch s
  have h27' : Ptx.get64 (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)) 27 = ptxSubPTX
      (Ptx.inputWord launch ((ptxOneAddress launch thread + 0 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8)) := by
    simpa only [ha21, ha22] using h27
  exact (ptxArithmeticLoad2_preserve27 launch
      (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))).trans
    ((ptxArithmeticLoad1_preserve27 launch
      (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))).trans h27')

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_diff32 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 32 = ptxSubPTX
      (Ptx.inputWord launch ((ptxOneAddress launch thread + 8 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8)) := by
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨_, _, _, _, _, h021, h022⟩ := ptxArithmeticLoad0_preserved launch s
  have h32 := ptxArithmeticLoad1_diff launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  have h32' : Ptx.get64 (ptxArithmeticLoad1 launch
      (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))) 32 = ptxSubPTX
      (Ptx.inputWord launch ((ptxOneAddress launch thread + 8 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8)) := by
    simpa only [h021, h022, ha21, ha22] using h32
  exact (ptxArithmeticLoad2_preserve32 launch
    (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))).trans h32'

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_diff37 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 37 = ptxSubPTX
      (Ptx.inputWord launch ((ptxOneAddress launch thread + 16 - launch.inputBase) / 8))
      (Ptx.inputWord launch ((ptxZeroAddress launch thread + 16 - launch.inputBase) / 8)) := by
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨_, _, _, _, _, h021, h022⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h121, h122⟩ :=
    ptxArithmeticLoad1_addresses launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  have h37 := ptxArithmeticLoad2_diff launch
    (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  dsimp only [ptxArithmeticInputs]
  simpa only [h121, h122, h021, h022, ha21, ha22] using h37

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_zero24 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 24 =
      Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8) := by
  obtain ⟨ha21, _⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨h24, _⟩ := ptxArithmeticLoad0_values launch s
  have h24' : Ptx.get64 (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)) 24 =
      Ptx.inputWord launch ((ptxZeroAddress launch thread + 0 - launch.inputBase) / 8) := by
    simpa only [ha21] using h24
  exact (ptxArithmeticLoad2_preserve24 launch
      (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))).trans
    ((ptxArithmeticLoad1_preserve24 launch
      (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))).trans h24')

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_zero29 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 29 =
      Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8) := by
  obtain ⟨ha21, _⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨_, _, _, _, _, h021, _⟩ := ptxArithmeticLoad0_preserved launch s
  have h29 := ptxArithmeticLoad1_zero launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  have h29' : Ptx.get64 (ptxArithmeticLoad1 launch
      (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))) 29 =
      Ptx.inputWord launch ((ptxZeroAddress launch thread + 8 - launch.inputBase) / 8) := by
    simpa only [h021, ha21] using h29
  exact (ptxArithmeticLoad2_preserve29 launch
    (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))).trans h29'

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticInputs_zero34 (launch : Launch) (s : Ptx.State) (thread : Ptx.Thread)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticInputs launch s) 34 =
      Ptx.inputWord launch ((ptxZeroAddress launch thread + 16 - launch.inputBase) / 8) := by
  obtain ⟨ha21, _⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨_, _, _, _, _, h021, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h121, _⟩ :=
    ptxArithmeticLoad1_addresses launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  have h34 := ptxArithmeticLoad2_zero launch
    (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  dsimp only [ptxArithmeticInputs]
  simpa only [h121, h021, ha21] using h34

private theorem ptxArithmeticBase_products (launch : Launch) (a : Ptx.State)
    (h1 : Ptx.get64 a 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 a 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 a 3 = Field.toWord launch.challenge.c0) :
    (Ptx.get64 (ptxArithmeticProducts a) 51 =
        ptxMulPTX (Ptx.get64 a 27) (Field.toWord launch.challenge.c0) ∧
     Ptx.get64 (ptxArithmeticProducts a) 65 =
        ptxMulPTX (Ptx.get64 a 32) (Field.toWord launch.challenge.c2) ∧
     Ptx.get64 (ptxArithmeticProducts a) 79 =
        ptxMulPTX (Ptx.get64 a 37) (Field.toWord launch.challenge.c1)) := by
  dsimp only [ptxArithmeticProducts]
  refine ⟨?_, ?_, ?_⟩
  · simp (config := { decide := true }) only [Ptx.get64_set64,
      Ptx.get64_subField_ne, Ptx.get64_mulField_ne, ↓reduceIte]
    rw [get64_mulField_38]
    simp (config := { decide := true }) only [Ptx.get64_set64,
      Ptx.get64_subField_ne, Ptx.get64_mulField_ne, h3, ↓reduceIte]
  · simp (config := { decide := true }) only [Ptx.get64_set64,
      Ptx.get64_subField_ne, Ptx.get64_mulField_ne, ↓reduceIte]
    rw [get64_mulField_52]
    simp (config := { decide := true }) only [Ptx.get64_set64,
      Ptx.get64_subField_ne, Ptx.get64_mulField_ne, h1, ↓reduceIte]
  · try simp (config := { decide := true }) only [Ptx.get64_set64,
      Ptx.get64_subField_ne, Ptx.get64_mulField_ne, ↓reduceIte]
    rw [get64_mulField_66]
    simp (config := { decide := true }) only [Ptx.get64_mulField_ne, h2, ↓reduceIte]

private theorem ptxArithmeticAddress_preserved (launch : Launch) (s : Ptx.State) :
    let s0 := ptxArithmeticAddress s
    (Ptx.get64 s0 1 = Ptx.get64 s 1 ∧
     Ptx.get64 s0 2 = Ptx.get64 s 2 ∧
     Ptx.get64 s0 3 = Ptx.get64 s 3 ∧
     Ptx.get64 s0 4 = Ptx.get64 s 4 ∧
     Ptx.get64 s0 12 = Ptx.get64 s 12) := by
  dsimp only [ptxArithmeticAddress, Ptx.mulLo64, Ptx.shl64Imm, Ptx.add64, Ptx.cvta,
    Ptx.madLo64ImmRight, Ptx.eval64]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticBase_input_regs (launch : Launch) (s : Ptx.State) :
    let input := ptxArithmeticInputs launch s
    let base := ptxArithmeticBase launch s
    (Ptx.get64 base 24 = Ptx.get64 input 24 ∧
     Ptx.get64 base 27 = Ptx.get64 input 27 ∧
     Ptx.get64 base 29 = Ptx.get64 input 29 ∧
     Ptx.get64 base 32 = Ptx.get64 input 32 ∧
     Ptx.get64 base 34 = Ptx.get64 input 34 ∧
     Ptx.get64 base 37 = Ptx.get64 input 37 ∧
     Ptx.get64 base 1 = Ptx.get64 input 1 ∧
     Ptx.get64 base 2 = Ptx.get64 input 2 ∧
     Ptx.get64 base 3 = Ptx.get64 input 3 ∧
     Ptx.get64 base 4 = Ptx.get64 input 4 ∧
     Ptx.get64 base 12 = Ptx.get64 input 12) := by
  dsimp only [ptxArithmeticBase, ptxArithmeticProducts]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    try simp (config := { decide := true }) only [Ptx.get64_mulField_ne]

private theorem ptxArithmeticC0_double (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticC0 launch a) 98 =
      ptxMulTwoPTX (ptxAddPTX (Ptx.get64 a 65) (Ptx.get64 a 79)) := by
  dsimp only [ptxArithmeticC0, Ptx.mov64]
  simp (config := { decide := true }) only [Ptx.get64_addField_ne]
  rw [get64_doubleField_85_set, get64_addField_80]

private def ptxArithmeticC0Pre (a : Ptx.State) : Ptx.State :=
  Ptx.doubleField (Ptx.mov64 (Ptx.addField a 80 16 65 79) 84 2) 85 17 83

private theorem ptxArithmeticC0_84 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticC0 launch a) 84 = 2 := by
  dsimp only [ptxArithmeticC0, Ptx.mov64]
  simp (config := { decide := true }) only [Ptx.get64_set64,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne, ↓reduceIte]

private theorem ptxArithmeticC0_get64_lt (launch : Launch) (a : Ptx.State) (r : Nat)
    (hr : r < 80) :
    Ptx.get64 (ptxArithmeticC0 launch a) r = Ptx.get64 a r := by
  dsimp only [ptxArithmeticC0, Ptx.mov64]
  simp (config := { decide := true }) (discharger := omega) only [
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne, Ptx.get64_set64, ↓reduceIte]
  rw [if_neg (show r ≠ 84 by omega)]

private theorem ptxArithmeticC0_sum (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticC0 launch a) 102 = ptxAddPTX
      (Ptx.get64 a 51)
      (ptxMulTwoPTX (ptxAddPTX (Ptx.get64 a 65) (Ptx.get64 a 79))) := by
  have h51 : Ptx.get64 (ptxArithmeticC0Pre a) 51 = Ptx.get64 a 51 := by
    have h := Ptx.get64_addField_ne (ptxArithmeticC0Pre a) 99 20 51 98 51
      (by decide) (by decide) (by decide) (by decide)
    exact h.symm.trans (ptxArithmeticC0_get64_lt launch a 51 (by decide))
  have h98 : Ptx.get64 (ptxArithmeticC0Pre a) 98 =
      ptxMulTwoPTX (ptxAddPTX (Ptx.get64 a 65) (Ptx.get64 a 79)) := by
    have h := Ptx.get64_addField_ne (ptxArithmeticC0Pre a) 99 20 51 98 98
      (by decide) (by decide) (by decide) (by decide)
    exact h.symm.trans (ptxArithmeticC0_double launch a)
  change Ptx.get64 (Ptx.addField (ptxArithmeticC0Pre a) 99 20 51 98) 102 = _
  rw [get64_addField_99, h51, h98]

private theorem ptxArithmeticC1_get64_lt (launch : Launch) (a : Ptx.State) (r : Nat)
    (hr : r < 100) :
    Ptx.get64 (ptxArithmeticC1 launch a) r = Ptx.get64 a r := by
  dsimp only [ptxArithmeticC1, Ptx.mov64]
  simp (config := { decide := true }) (discharger := omega) only [
    Ptx.get64_mulField_ne, Ptx.get64_addField_ne, Ptx.get64_doubleField_ne,
    Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticC1_preserve102 (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticC1 launch a) 102 = Ptx.get64 a 102 := by
  dsimp only [ptxArithmeticC1, Ptx.mov64]
  rfl

private theorem ptxArithmeticC1_sum (launch : Launch) (a : Ptx.State)
    (h84 : Ptx.get64 a 84 = 2) :
    Ptx.get64 (ptxArithmeticC1 launch a) 166 = ptxAddPTX
      (ptxAddPTX
        (ptxMulPTX (Ptx.get64 a 27) (Ptx.get64 a 2))
        (ptxMulPTX (Ptx.get64 a 32) (Ptx.get64 a 3)))
      (ptxMulTwoPTX (ptxMulPTX (Ptx.get64 a 37) (Ptx.get64 a 1))) := by
  dsimp only [ptxArithmeticC1, Ptx.mov64]
  try simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne]
  rw [get64_addField_163]
  simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne]
  rw [get64_addField_131, get64_doubleField_149]
  simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne]
  rw [get64_mulField_103, get64_mulField_117, get64_mulField_135]
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_mulField_ne, Ptx.get64_addField_ne,
      Ptx.get64_doubleField_ne, Ptx.get64_set64, h84, ↓reduceIte]

private theorem ptxArithmeticC2_sum (launch : Launch) (a : Ptx.State) :
    Ptx.get64 (ptxArithmeticC2 launch a) 216 = ptxAddPTX
      (ptxMulPTX (Ptx.get64 a 27) (Ptx.get64 a 1))
      (ptxAddPTX
        (ptxMulPTX (Ptx.get64 a 32) (Ptx.get64 a 2))
        (ptxMulPTX (Ptx.get64 a 37) (Ptx.get64 a 3))) := by
  dsimp only [ptxArithmeticC2]
  try simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne]
  rw [get64_addField_213, get64_addField_209]
  simp (config := { decide := true }) only [Ptx.get64_mulField_ne,
    Ptx.get64_addField_ne, Ptx.get64_doubleField_ne]
  rw [get64_mulField_167, get64_mulField_181, get64_mulField_195]
  simp (config := { decide := true }) only [Ptx.get64_mulField_ne, Ptx.get64_addField_ne,
    Ptx.get64_doubleField_ne, Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticC2_preserved (launch : Launch) (a : Ptx.State) :
    (Ptx.get64 (ptxArithmeticC2 launch a) 102 = Ptx.get64 a 102 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 166 = Ptx.get64 a 166 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 24 = Ptx.get64 a 24 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 29 = Ptx.get64 a 29 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 34 = Ptx.get64 a 34 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 4 = Ptx.get64 a 4 ∧
     Ptx.get64 (ptxArithmeticC2 launch a) 12 = Ptx.get64 a 12) := by
  dsimp only [ptxArithmeticC2]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    simp (config := { decide := true }) only [Ptx.get64_mulField_ne, Ptx.get64_addField_ne,
      Ptx.get64_doubleField_ne, Ptx.get64_set64, ↓reduceIte]

private theorem ptxArithmeticOutputBase_230 (launch : Launch) (a : Ptx.State)
    (base predicate left right : Nat) (hbase : 12 < base) (thread : Ptx.Thread)
    (h4 : Ptx.get64 a 4 = thread.index) (h12 : Ptx.get64 a 12 = launch.outputBase) :
    Ptx.get64 (ptxArithmeticOutputBase launch a base predicate left right) 230 =
      (thread.index * 24 + launch.outputBase) % Word.base := by
  change (Ptx.get64 (Ptx.addField a base predicate left right) 4 * 24 +
    Ptx.get64 (Ptx.addField a base predicate left right) 12) % Word.base = _
  rw [Ptx.get64_addField_ne _ _ _ _ _ _ (by omega) (by omega) (by omega) (by omega),
    Ptx.get64_addField_ne _ _ _ _ _ _ (by omega) (by omega) (by omega) (by omega)]
  rw [h4, h12]

private theorem ptxArithmeticOutputBase_value (launch : Launch) (a : Ptx.State)
    (base predicate left right : Nat) (hbase : base + 3 < 229)
    (hl : left ≠ base) (hl1 : left ≠ base + 1) :
    Ptx.get64 (ptxArithmeticOutputBase launch a base predicate left right) (base + 3) =
      ptxAddPTX (Ptx.get64 a left) (Ptx.get64 a right) := by
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticOutputBase, Ptx.cvta, Ptx.madLo64ImmRight, Ptx.get64_set64, ↓reduceIte]
  rw [if_neg (show base + 3 ≠ 230 by omega), if_neg (show base + 3 ≠ 229 by omega)]
  exact Ptx.get64_addField a base predicate left right hl hl1

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticOutput_c0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) (a : Ptx.State)
    (h4 : Ptx.get64 a 4 = thread.index) (h12 : Ptx.get64 a 12 = launch.outputBase) :
    (ptxArithmeticOutput launch a).output (3 * thread.index) =
      ptxAddPTX (Ptx.get64 a 24) (Ptx.get64 a 102) := by
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  dsimp only [ptxArithmeticOutput, Ptx.stGlobalOffset]
  rw [Ptx.output_store, Ptx.output_store, Ptx.output_store]
  simp (config := { failIfUnchanged := false }) only [
    ptxArithmeticOutputBase_230 launch a 217 46 24 102 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 221 47 29 166 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 225 48 34 216 (by decide) thread h4 h12,
    ptxOutputWordIndex16 launch valid thread hi,
    ptxOutputWordIndex8 launch valid thread hi,
    ptxOutputWordIndex0 launch valid thread hi, Nat.add_zero,
    hn2, hn1, hn12, ↓reduceIte]
  exact ptxArithmeticOutputBase_value launch a 217 46 24 102 (by decide) (by decide) (by decide)

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticOutput_c1 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) (a : Ptx.State)
    (h4 : Ptx.get64 a 4 = thread.index) (h12 : Ptx.get64 a 12 = launch.outputBase) :
    (ptxArithmeticOutput launch a).output (3 * thread.index + 1) =
      ptxAddPTX (Ptx.get64 a 29) (Ptx.get64 a 166) := by
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  dsimp only [ptxArithmeticOutput, Ptx.stGlobalOffset]
  rw [Ptx.output_store, Ptx.output_store]
  simp (config := { failIfUnchanged := false }) only [
    ptxArithmeticOutputBase_230 launch a 217 46 24 102 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 221 47 29 166 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 225 48 34 216 (by decide) thread h4 h12,
    ptxOutputWordIndex16 launch valid thread hi,
    ptxOutputWordIndex8 launch valid thread hi,
    ptxOutputWordIndex0 launch valid thread hi, Nat.add_zero,
    hn2, hn1, hn12, ↓reduceIte]
  exact ptxArithmeticOutputBase_value launch a 221 47 29 166 (by decide) (by decide) (by decide)

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticOutput_c2 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount) (a : Ptx.State)
    (h4 : Ptx.get64 a 4 = thread.index) (h12 : Ptx.get64 a 12 = launch.outputBase) :
    (ptxArithmeticOutput launch a).output (3 * thread.index + 2) =
      ptxAddPTX (Ptx.get64 a 34) (Ptx.get64 a 216) := by
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  dsimp only [ptxArithmeticOutput, Ptx.stGlobalOffset]
  rw [Ptx.output_store, Ptx.output_store]
  simp (config := { failIfUnchanged := false }) only [
    ptxArithmeticOutputBase_230 launch a 217 46 24 102 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 221 47 29 166 (by decide) thread h4 h12,
    ptxArithmeticOutputBase_230 launch a 225 48 34 216 (by decide) thread h4 h12,
    ptxOutputWordIndex16 launch valid thread hi,
    ptxOutputWordIndex8 launch valid thread hi,
    ptxOutputWordIndex0 launch valid thread hi, Nat.add_zero,
    hn2, hn1, hn12, ↓reduceIte]
  exact ptxArithmeticOutputBase_value launch a 225 48 34 216 (by decide) (by decide) (by decide)

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticGround_c0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (h11 : Ptx.get64 s 11 = launch.inputBase)
    (h14 : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticC0 launch (ptxArithmeticBase launch s)) 102 =
      ptxAddPTX
        (ptxMulPTX
          (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index))
            (Ptx.inputWord launch (3 * launch.zeroIndex thread.index)))
          launch.challenge.c0.toWord)
        (ptxMulTwoPTX
          (ptxAddPTX
            (ptxMulPTX
              (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 1))
                (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 1)))
              launch.challenge.c2.toWord)
            (ptxMulPTX
              (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 2))
                (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 2)))
              launch.challenge.c1.toWord))) := by
  have hd27 := ptxArithmeticInputs_diff27 launch s thread h4 h11 h14 h231 h232
  have hd32 := ptxArithmeticInputs_diff32 launch s thread h4 h11 h14 h231 h232
  have hd37 := ptxArithmeticInputs_diff37 launch s thread h4 h11 h14 h231 h232
  obtain ⟨ha1, ha2, ha3, _, _⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, _, _, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, _, _⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, _, _⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  obtain ⟨hb51, hb65, hb79⟩ := ptxArithmeticBase_products launch (ptxArithmeticInputs launch s)
    (hb1.trans h1) (hb2.trans h2) (hb3.trans h3)
  rw [ptxArithmeticC0_sum]
  dsimp only [ptxArithmeticBase]
  simp only [hb51, hb65, hb79, hd27, hd32, hd37, Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticGround_c1 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (h11 : Ptx.get64 s 11 = launch.inputBase)
    (h14 : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))) 166 =
      ptxAddPTX
        (ptxAddPTX
          (ptxMulPTX
            (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index))
              (Ptx.inputWord launch (3 * launch.zeroIndex thread.index)))
            launch.challenge.c1.toWord)
          (ptxMulPTX
            (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 1))
              (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 1)))
            launch.challenge.c0.toWord))
        (ptxMulTwoPTX
          (ptxMulPTX
            (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 2))
              (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 2)))
            launch.challenge.c2.toWord)) := by
  have hd27 := ptxArithmeticInputs_diff27 launch s thread h4 h11 h14 h231 h232
  have hd32 := ptxArithmeticInputs_diff32 launch s thread h4 h11 h14 h231 h232
  have hd37 := ptxArithmeticInputs_diff37 launch s thread h4 h11 h14 h231 h232
  obtain ⟨ha1, ha2, ha3, _, _⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, _, _, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, _, _⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, _, _⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  obtain ⟨_, hi27, _, hi32, _, hi37, hi1, hi2, hi3, _, _⟩ :=
    ptxArithmeticBase_input_regs launch s
  rw [ptxArithmeticC1_sum launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))
    (ptxArithmeticC0_84 launch (ptxArithmeticBase launch s))]
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticC0_get64_lt, hi27, hi32, hi37, hi1, hi2, hi3,
    hb1, hb2, hb3, h1, h2, h3, hd27, hd32, hd37, Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticGround_c2 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (h11 : Ptx.get64 s 11 = launch.inputBase)
    (h14 : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.get64 (ptxArithmeticC2 launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 216 =
      ptxAddPTX
        (ptxMulPTX
          (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index))
            (Ptx.inputWord launch (3 * launch.zeroIndex thread.index)))
          launch.challenge.c2.toWord)
        (ptxAddPTX
          (ptxMulPTX
            (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 1))
              (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 1)))
            launch.challenge.c1.toWord)
          (ptxMulPTX
            (ptxSubPTX (Ptx.inputWord launch (3 * launch.oneIndex thread.index + 2))
              (Ptx.inputWord launch (3 * launch.zeroIndex thread.index + 2)))
            launch.challenge.c0.toWord)) := by
  have hd27 := ptxArithmeticInputs_diff27 launch s thread h4 h11 h14 h231 h232
  have hd32 := ptxArithmeticInputs_diff32 launch s thread h4 h11 h14 h231 h232
  have hd37 := ptxArithmeticInputs_diff37 launch s thread h4 h11 h14 h231 h232
  obtain ⟨ha1, ha2, ha3, _, _⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, _, _, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, _, _⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, _, _⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  obtain ⟨_, hi27, _, hi32, _, hi37, hi1, hi2, hi3, _, _⟩ :=
    ptxArithmeticBase_input_regs launch s
  rw [ptxArithmeticC2_sum]
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt,
    hi27, hi32, hi37, hi1, hi2, hi3, hb1, hb2, hb3, h1, h2, h3,
    hd27, hd32, hd37, Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticPTX_c0 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hout : Ptx.get64 s 12 = launch.outputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Field.ofWord ((ptxArithmeticOutput launch (ptxArithmeticC2 launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))))).output (3 * thread.index)) = (ptxArithmeticPTX launch thread).c0 := by
  have hz24 := ptxArithmeticInputs_zero24 launch s thread h4 hinput hlow h231 h232
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  obtain ⟨hc2_102, hc2_166, hc2_24, hc2_29, hc2_34, hc2_4, hc2_12⟩ :=
    ptxArithmeticC2_preserved launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))
  obtain hc1_102 := ptxArithmeticC1_preserve102 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨ha1, ha2, ha3, ha4, ha12⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, h04, h012, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, h14, h112⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, h24, h212⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  have hb4 : Ptx.get64 (ptxArithmeticInputs launch s) 4 = Ptx.get64 s 4 :=
    h24.trans (h14.trans (h04.trans ha4))
  have hb12 : Ptx.get64 (ptxArithmeticInputs launch s) 12 = Ptx.get64 s 12 :=
    h212.trans (h112.trans (h012.trans ha12))
  obtain ⟨hb51, hb65, hb79⟩ := ptxArithmeticBase_products launch (ptxArithmeticInputs launch s)
    (hb1.trans h1) (hb2.trans h2) (hb3.trans h3)
  obtain ⟨hi24, hi27, hi29, hi32, hi34, hi37, hi1, hi2, hi3, hi4, hi12⟩ :=
    ptxArithmeticBase_input_regs launch s
  have ha4 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 4 =
      thread.index := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_4, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi4, hb4, h4]
  have ha12 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 12 =
      launch.outputBase := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_12, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi12, hb12, hout]
  simp only [ptxArithmeticOutput_c0 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c1 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c2 launch valid thread hi _ ha4 ha12]
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt,
    hc2_24, hi24, hz24, hc2_102, hc1_102,
    ptxArithmeticGround_c0 launch valid thread hi h32 s h1 h2 h3 h4 hinput hlow h231 h232]
  rw [ptxArithmeticPTX_eq_ssa launch valid thread h32]
  simp only [ptxArithmeticSSA, ptxSubPTX_eq, ptxAddPTX_eq, ptxMulPTX_eq, ptxMulTwoPTX_eq,
    Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticPTX_c1 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hout : Ptx.get64 s 12 = launch.outputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Field.ofWord ((ptxArithmeticOutput launch (ptxArithmeticC2 launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))))).output (3 * thread.index + 1)) = (ptxArithmeticPTX launch thread).c1 := by
  have hz29 := ptxArithmeticInputs_zero29 launch s thread h4 hinput hlow h231 h232
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  obtain ⟨hc2_102, hc2_166, hc2_24, hc2_29, hc2_34, hc2_4, hc2_12⟩ :=
    ptxArithmeticC2_preserved launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))
  obtain hc1_102 := ptxArithmeticC1_preserve102 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨ha1, ha2, ha3, ha4, ha12⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, h04, h012, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, h14, h112⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, h24, h212⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  have hb4 : Ptx.get64 (ptxArithmeticInputs launch s) 4 = Ptx.get64 s 4 :=
    h24.trans (h14.trans (h04.trans ha4))
  have hb12 : Ptx.get64 (ptxArithmeticInputs launch s) 12 = Ptx.get64 s 12 :=
    h212.trans (h112.trans (h012.trans ha12))
  obtain ⟨hb51, hb65, hb79⟩ := ptxArithmeticBase_products launch (ptxArithmeticInputs launch s)
    (hb1.trans h1) (hb2.trans h2) (hb3.trans h3)
  obtain ⟨hi24, hi27, hi29, hi32, hi34, hi37, hi1, hi2, hi3, hi4, hi12⟩ :=
    ptxArithmeticBase_input_regs launch s
  have ha4 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 4 =
      thread.index := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_4, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi4, hb4, h4]
  have ha12 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 12 =
      launch.outputBase := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_12, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi12, hb12, hout]
  simp only [ptxArithmeticOutput_c0 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c1 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c2 launch valid thread hi _ ha4 ha12]
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt,
    hc2_29, hi29, hz29, hc2_166,
    ptxArithmeticGround_c1 launch valid thread hi h32 s h1 h2 h3 h4 hinput hlow h231 h232]
  rw [ptxArithmeticPTX_eq_ssa launch valid thread h32]
  simp only [ptxArithmeticSSA, ptxSubPTX_eq, ptxAddPTX_eq, ptxMulPTX_eq, ptxMulTwoPTX_eq,
    Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

set_option maxRecDepth 1000000 in
private theorem ptxArithmeticPTX_c2 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hout : Ptx.get64 s 12 = launch.outputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Field.ofWord ((ptxArithmeticOutput launch (ptxArithmeticC2 launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))))).output (3 * thread.index + 2)) = (ptxArithmeticPTX launch thread).c2 := by
  have hz34 := ptxArithmeticInputs_zero34 launch s thread h4 hinput hlow h231 h232
  have hn2 : 3 * thread.index ≠ 3 * thread.index + 2 := by omega
  have hn1 : 3 * thread.index ≠ 3 * thread.index + 1 := by omega
  have hn12 : 3 * thread.index + 1 ≠ 3 * thread.index + 2 := by omega
  obtain ⟨hc2_102, hc2_166, hc2_24, hc2_29, hc2_34, hc2_4, hc2_12⟩ :=
    ptxArithmeticC2_preserved launch (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))
  obtain hc1_102 := ptxArithmeticC1_preserve102 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s))
  obtain ⟨ha21, ha22⟩ := ptxArithmeticAddress_values launch s thread h4 hinput hlow h231 h232
  obtain ⟨ha1, ha2, ha3, ha4, ha12⟩ := ptxArithmeticAddress_preserved launch s
  obtain ⟨h01, h02, h03, h04, h012, _, _⟩ := ptxArithmeticLoad0_preserved launch s
  obtain ⟨h11, h12, h13, h14, h112⟩ :=
    ptxArithmeticLoad1_preserved launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s))
  obtain ⟨h21, h22, h23, h24, h212⟩ :=
    ptxArithmeticLoad2_preserved launch (ptxArithmeticLoad1 launch (ptxArithmeticLoad0 launch (ptxArithmeticAddress s)))
  have hb1 : Ptx.get64 (ptxArithmeticInputs launch s) 1 = Ptx.get64 s 1 :=
    h21.trans (h11.trans (h01.trans ha1))
  have hb2 : Ptx.get64 (ptxArithmeticInputs launch s) 2 = Ptx.get64 s 2 :=
    h22.trans (h12.trans (h02.trans ha2))
  have hb3 : Ptx.get64 (ptxArithmeticInputs launch s) 3 = Ptx.get64 s 3 :=
    h23.trans (h13.trans (h03.trans ha3))
  have hb4 : Ptx.get64 (ptxArithmeticInputs launch s) 4 = Ptx.get64 s 4 :=
    h24.trans (h14.trans (h04.trans ha4))
  have hb12 : Ptx.get64 (ptxArithmeticInputs launch s) 12 = Ptx.get64 s 12 :=
    h212.trans (h112.trans (h012.trans ha12))
  obtain ⟨hb51, hb65, hb79⟩ := ptxArithmeticBase_products launch (ptxArithmeticInputs launch s)
    (hb1.trans h1) (hb2.trans h2) (hb3.trans h3)
  obtain ⟨hi24, hi27, hi29, hi32, hi34, hi37, hi1, hi2, hi3, hi4, hi12⟩ :=
    ptxArithmeticBase_input_regs launch s
  have ha4 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 4 =
      thread.index := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_4, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi4, hb4, h4]
  have ha12 : Ptx.get64 (ptxArithmeticC2 launch
      (ptxArithmeticC1 launch (ptxArithmeticC0 launch (ptxArithmeticBase launch s)))) 12 =
      launch.outputBase := by
    simp (config := { decide := true }) (discharger := omega) only [
      hc2_12, ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt, hi12, hb12, hout]
  simp only [ptxArithmeticOutput_c0 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c1 launch valid thread hi _ ha4 ha12,
    ptxArithmeticOutput_c2 launch valid thread hi _ ha4 ha12]
  simp (config := { decide := true }) (discharger := omega) only [
    ptxArithmeticC1_get64_lt, ptxArithmeticC0_get64_lt,
    hc2_34, hi34, hz34,
    ptxArithmeticGround_c2 launch valid thread hi h32 s h1 h2 h3 h4 hinput hlow h231 h232]
  rw [ptxArithmeticPTX_eq_ssa launch valid thread h32]
  simp only [ptxArithmeticSSA, ptxSubPTX_eq, ptxAddPTX_eq, ptxMulPTX_eq, ptxMulTwoPTX_eq,
    Nat.add_zero,
    ptxZeroWordIndex0 launch valid thread hi h32,
    ptxOneWordIndex0 launch valid thread hi h32,
    ptxZeroWordIndex8 launch valid thread hi h32,
    ptxZeroWordIndex16 launch valid thread hi h32,
    ptxOneWordIndex8 launch valid thread hi h32,
    ptxOneWordIndex16 launch valid thread hi h32]

private theorem ptxArithmeticPTX_eq_of_state (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000) (s : Ptx.State)
    (h1 : Ptx.get64 s 1 = Field.toWord launch.challenge.c2)
    (h2 : Ptx.get64 s 2 = Field.toWord launch.challenge.c1)
    (h3 : Ptx.get64 s 3 = Field.toWord launch.challenge.c0)
    (h4 : Ptx.get64 s 4 = thread.index)
    (hinput : Ptx.get64 s 11 = launch.inputBase)
    (hout : Ptx.get64 s 12 = launch.outputBase)
    (hlow : Ptx.get64 s 14 = launch.lowHalf)
    (h231 : Ptx.get64 s 231 = thread.index / launch.lowHalf)
    (h232 : Ptx.get64 s 232 = thread.index % launch.lowHalf) :
    Ptx.outputExt (ptxArithmetic launch s) thread.index = ptxArithmeticPTX launch thread := by
  apply FieldExt.ext
  · simpa only [Ptx.outputExt, ptxArithmetic] using
      ptxArithmeticPTX_c0 launch valid thread hi h32 s h1 h2 h3 h4 hinput hout hlow h231 h232
  · simpa only [Ptx.outputExt, ptxArithmetic] using
      ptxArithmeticPTX_c1 launch valid thread hi h32 s h1 h2 h3 h4 hinput hout hlow h231 h232
  · simpa only [Ptx.outputExt, ptxArithmetic] using
      ptxArithmeticPTX_c2 launch valid thread hi h32 s h1 h2 h3 h4 hinput hout hlow h231 h232
set_option maxRecDepth 1000000 in
set_option maxHeartbeats 1000000 in
private theorem ptxArithmeticPTX_eq_of_32 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000)
    (haddr : launch.lowHalf &&& 18446744069414584320 = 0) :
    Ptx.outputExt (ptxArithmetic launch (ptxAddress (ptxPrepare launch thread))) thread.index =
      ptxArithmeticPTX launch thread := by
  have hrem :
      (thread.index + 0x100000000 -
        (thread.index / launch.lowHalf * launch.lowHalf) % 0x100000000) % 0x100000000 =
        thread.index % launch.lowHalf := by
    simpa only [Nat.mod_eq_of_lt
      (and_high_zero_lt_32 (Launch.lowHalf_lt_base launch valid) haddr)] using
      ptxAddress32_remainder launch valid thread h32 haddr
  generalize hs : ptxAddress (ptxPrepare launch thread) = s
  have p1 : Ptx.get64 (ptxPrepare launch thread) 1 = Field.toWord launch.challenge.c2 := rfl
  have p2 : Ptx.get64 (ptxPrepare launch thread) 2 = Field.toWord launch.challenge.c1 := rfl
  have p3 : Ptx.get64 (ptxPrepare launch thread) 3 = Field.toWord launch.challenge.c0 := rfl
  have p4 : Ptx.get64 (ptxPrepare launch thread) 4 = thread.index :=
    Ptx.Thread.index_mod thread h32
  have p11 : Ptx.get64 (ptxPrepare launch thread) 11 = launch.inputBase := rfl
  have p12 : Ptx.get64 (ptxPrepare launch thread) 12 = launch.outputBase := rfl
  have p14 : Ptx.get64 (ptxPrepare launch thread) 14 = launch.lowHalf := rfl
  have h15 : Ptx.get64 (Ptx.and64 (ptxPrepare launch thread) 15
      (Ptx.Value.register 14) (Ptx.Value.immediate 18446744069414584320)) 15 =
      launch.lowHalf &&& 18446744069414584320 := by
    simp (config := { decide := true }) only [Ptx.and64, Ptx.get64_set64, Ptx.eval64, p14,
      ↓reduceIte]
  obtain ⟨h1, h2, h3, h4, h11, h12, h14, h231, h232⟩ :
      Ptx.get64 s 1 = Field.toWord launch.challenge.c2 ∧
      Ptx.get64 s 2 = Field.toWord launch.challenge.c1 ∧
      Ptx.get64 s 3 = Field.toWord launch.challenge.c0 ∧
      Ptx.get64 s 4 = thread.index ∧
      Ptx.get64 s 11 = launch.inputBase ∧
      Ptx.get64 s 12 = launch.outputBase ∧
      Ptx.get64 s 14 = launch.lowHalf ∧
      Ptx.get64 s 231 = thread.index / launch.lowHalf ∧
      Ptx.get64 s 232 = thread.index % launch.lowHalf := by
    rw [← hs]
    simp only [ptxAddress, h15, haddr, ↓reduceIte]
    simp (config := { decide := true }) only [
      Ptx.cvt64, Ptx.cvt32, Ptx.div32, Ptx.mulLo32, Ptx.sub32, Ptx.and64,
      Ptx.eval64, Ptx.eval32, Ptx.get64_set64, Ptx.get64_set32, Ptx.get32_set32,
      Ptx.get32_set64,
      p1, p2, p3, p4, p11, p12, p14, hrem, Nat.mod_eq_of_lt h32,
      Nat.mod_eq_of_lt (and_high_zero_lt_32 (Launch.lowHalf_lt_base launch valid) haddr),
      ↓reduceIte]
  exact ptxArithmeticPTX_eq_of_state launch valid thread hi h32 s
    h1 h2 h3 h4 h11 h12 h14 h231 h232

set_option maxHeartbeats 1000000 in
private theorem ptxArithmeticPTX_eq_of_64 (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (hi : thread.index < launch.outputCount)
    (h32 : thread.index < 0x100000000)
    (haddr : ¬launch.lowHalf &&& 18446744069414584320 = 0) :
    Ptx.outputExt (ptxArithmetic launch (ptxAddress (ptxPrepare launch thread))) thread.index =
      ptxArithmeticPTX launch thread := by
  generalize hs : ptxAddress (ptxPrepare launch thread) = s
  have haddrBool : (launch.lowHalf &&& 18446744069414584320 != 0) = true := by
    simp [haddr]
  have p1 : Ptx.get64 (ptxPrepare launch thread) 1 = Field.toWord launch.challenge.c2 := rfl
  have p2 : Ptx.get64 (ptxPrepare launch thread) 2 = Field.toWord launch.challenge.c1 := rfl
  have p3 : Ptx.get64 (ptxPrepare launch thread) 3 = Field.toWord launch.challenge.c0 := rfl
  have p4 : Ptx.get64 (ptxPrepare launch thread) 4 = thread.index :=
    Ptx.Thread.index_mod thread h32
  have p11 : Ptx.get64 (ptxPrepare launch thread) 11 = launch.inputBase := rfl
  have p12 : Ptx.get64 (ptxPrepare launch thread) 12 = launch.outputBase := rfl
  have p14 : Ptx.get64 (ptxPrepare launch thread) 14 = launch.lowHalf := rfl
  have h15 : Ptx.get64 (Ptx.and64 (ptxPrepare launch thread) 15
      (Ptx.Value.register 14) (Ptx.Value.immediate 18446744069414584320)) 15 =
      launch.lowHalf &&& 18446744069414584320 := by
    simp (config := { decide := true }) only [Ptx.and64, Ptx.get64_set64, Ptx.eval64, p14,
      ↓reduceIte]
  obtain ⟨h1, h2, h3, h4, h11, h12, h14, h231, h232⟩ :
      Ptx.get64 s 1 = Field.toWord launch.challenge.c2 ∧
      Ptx.get64 s 2 = Field.toWord launch.challenge.c1 ∧
      Ptx.get64 s 3 = Field.toWord launch.challenge.c0 ∧
      Ptx.get64 s 4 = thread.index ∧
      Ptx.get64 s 11 = launch.inputBase ∧
      Ptx.get64 s 12 = launch.outputBase ∧
      Ptx.get64 s 14 = launch.lowHalf ∧
      Ptx.get64 s 231 = thread.index / launch.lowHalf ∧
      Ptx.get64 s 232 = thread.index % launch.lowHalf := by
    rw [← hs]
    simp only [ptxAddress, h15, haddrBool, ↓reduceIte]
    simp (config := { decide := true }) only [
      Ptx.cvt64, Ptx.div64, Ptx.mulLo64, Ptx.sub64, Ptx.and64,
      Ptx.eval64, Ptx.eval32, Ptx.get64_set64, Ptx.get64_set32,
      p1, p2, p3, p4, p11, p12, p14, ptxAddress64_remainder launch thread h32,
      ↓reduceIte]
  exact ptxArithmeticPTX_eq_of_state launch valid thread hi h32 s
    h1 h2 h3 h4 h11 h12 h14 h231 h232

theorem ptxRunThread_eq_ssa (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    ptxRunThread launch thread = ptxRunThreadSSA launch thread := by
  by_cases hlow : launch.lowHalf = 0
  · simp [ptxRunThread, ptxRunThreadSSA, hlow]
  by_cases hinactive : launch.outputCount ≤ thread.index
  · simp [ptxRunThread, ptxRunThreadSSA, hlow, hinactive]
  have hi : thread.index < launch.outputCount := Nat.lt_of_not_ge hinactive
  simp only [ptxRunThread, ptxRunThreadSSA, hlow, hinactive, ↓reduceIte]
  apply congrArg some
  rw [← ptxArithmeticPTX_eq_ssa launch valid thread h32]
  by_cases haddr : launch.lowHalf &&& 18446744069414584320 = 0
  · exact ptxArithmeticPTX_eq_of_32 launch valid thread hi h32 haddr
  · exact ptxArithmeticPTX_eq_of_64 launch valid thread hi h32 haddr

theorem ptxRunThread_correct (launch : Launch) (valid : Launch.Valid launch)
    (thread : Ptx.Thread) (h32 : thread.index < 0x100000000) :
    ptxRunThread launch thread = Launch.runThread launch thread.index := by
  by_cases hlow : launch.lowHalf = 0
  · simp [ptxRunThread, Launch.runThread, hlow]
  by_cases hinactive : launch.outputCount ≤ thread.index
  · simp [ptxRunThread, Launch.runThread, hlow, hinactive]
  have hi : thread.index < launch.outputCount := Nat.lt_of_not_ge hinactive
  rw [ptxRunThread_eq_ssa launch valid thread h32]
  simp only [ptxRunThreadSSA, hlow, hinactive, ↓reduceIte]
  rw [ptxArithmeticSSA_correct launch valid thread hi h32,
    Launch.runThread_correct launch valid hi, FieldExt.foldFused_correct]

/-- A concrete valid launch used to establish non-vacuity. -/
def exampleLaunch : Launch where
  input := #[FieldExt.ofInt ⟨1, 2, 3⟩, FieldExt.ofInt ⟨4, 5, 6⟩]
  outputCount := 1
  lowHalf := 1
  challenge := FieldExt.ofInt ⟨7, 11, 13⟩
  inputBase := 0
  outputBase := 0

theorem exampleLaunch_valid : exampleLaunch.Valid := by
  constructor <;> decide

theorem runThread_nonvacuous : Launch.runThread exampleLaunch 0 ≠ some FieldExt.zero := by
  decide

def exampleThread : Ptx.Thread where
  blockIndex := 0
  blockDim := 1
  threadIndex := 0

theorem ptxRunThread_nonvacuous : ptxRunThread exampleLaunch exampleThread ≠ some FieldExt.zero := by
  decide

#print axioms ptxRunThread_nonvacuous
#print axioms ptxRunThread_eq_ssa
#print axioms ptxRunThread_correct

#print axioms Word.add_correct
#print axioms Word.sub_correct
#print axioms Word.reduce128_correct
#print axioms Word.mul_correct
#print axioms Field.word_add_correct
#print axioms Field.word_sub_correct
#print axioms Field.word_mul_correct
#print axioms FieldExt.foldFused_correct
#print axioms FieldExt.foldFused_nonvacuous
#print axioms Launch.oneIndex_lt_twice_output
#print axioms Launch.runThread_correct
#print axioms Launch.runThread_inactive
#print axioms runThread_nonvacuous

end Kernel

end Weightproof
