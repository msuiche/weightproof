set_option maxRecDepth 100000
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
  · simp only [canonical, h, ↓reduceIte]
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

def outputExt (state : State) (index : Nat) : FieldExt :=
  ⟨Field.ofWord (state.output (3 * index)),
   Field.ofWord (state.output (3 * index + 1)),
   Field.ofWord (state.output (3 * index + 2))⟩

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

def ptxArithmetic (launch : Launch) (state : Ptx.State) : Ptx.State :=
  let s := state
  let s := Ptx.mulLo64 s 17 14 231
  let s := Ptx.shl64Imm s 18 17 1
  let s := Ptx.add64 s 19 232 18
  let s := Ptx.cvta s 20 11
  let s := Ptx.madLo64ImmRight s 21 19 24 20
  let s := Ptx.madLo64ImmRight s 22 14 24 21
  let s := Ptx.ldGlobalOffset s launch 23 22 0
  let s := Ptx.ldGlobalOffset s launch 24 21 0
  let s := Ptx.sub64 s 25 23 24
  let s := Ptx.setpLt s 4 23 24
  let s := Ptx.add64 s 26 25 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 27 26 25 4
  let s := Ptx.ldGlobalOffset s launch 28 22 8
  let s := Ptx.ldGlobalOffset s launch 29 21 8
  let s := Ptx.sub64 s 30 28 29
  let s := Ptx.setpLt s 5 28 29
  let s := Ptx.add64 s 31 30 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 32 31 30 5
  let s := Ptx.ldGlobalOffset s launch 33 22 16
  let s := Ptx.ldGlobalOffset s launch 34 21 16
  let s := Ptx.sub64 s 35 33 34
  let s := Ptx.setpLt s 6 33 34
  let s := Ptx.add64 s 36 35 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 37 36 35 6
  let s := Ptx.mulHi64 s 38 27 3
  let s := Ptx.mulLo64 s 39 27 3
  let s := Ptx.shr64Imm s 40 38 32
  let s := Ptx.and64ImmRight s 41 38 4294967295
  let s := Ptx.sub64 s 42 39 40
  let s := Ptx.setpLt s 7 39 40
  let s := Ptx.selp64 s 43 (Ptx.Value.immediate 18446744069414584321) 0 7
  let s := Ptx.shl64Imm s 44 38 32
  let s := Ptx.sub64 s 45 44 41
  let s := Ptx.add64 s 46 42 45
  let s := Ptx.add64 s 47 46 43
  let s := Ptx.setpLt s 8 47 45
  let s := Ptx.selp64 s 48 4294967295 0 8
  let s := Ptx.add64 s 49 48 47
  let s := Ptx.setpGt s 9 49 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 50 49 4294967295
  let s := Ptx.selp64 s 51 50 49 9
  let s := Ptx.mulHi64 s 52 32 1
  let s := Ptx.mulLo64 s 53 32 1
  let s := Ptx.shr64Imm s 54 52 32
  let s := Ptx.and64ImmRight s 55 52 4294967295
  let s := Ptx.sub64 s 56 53 54
  let s := Ptx.setpLt s 10 53 54
  let s := Ptx.selp64 s 57 (Ptx.Value.immediate 18446744069414584321) 0 10
  let s := Ptx.shl64Imm s 58 52 32
  let s := Ptx.sub64 s 59 58 55
  let s := Ptx.add64 s 60 56 59
  let s := Ptx.add64 s 61 60 57
  let s := Ptx.setpLt s 11 61 59
  let s := Ptx.selp64 s 62 4294967295 0 11
  let s := Ptx.add64 s 63 62 61
  let s := Ptx.setpGt s 12 63 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 64 63 4294967295
  let s := Ptx.selp64 s 65 64 63 12
  let s := Ptx.mulHi64 s 66 37 2
  let s := Ptx.mulLo64 s 67 37 2
  let s := Ptx.shr64Imm s 68 66 32
  let s := Ptx.and64ImmRight s 69 66 4294967295
  let s := Ptx.sub64 s 70 67 68
  let s := Ptx.setpLt s 13 67 68
  let s := Ptx.selp64 s 71 (Ptx.Value.immediate 18446744069414584321) 0 13
  let s := Ptx.shl64Imm s 72 66 32
  let s := Ptx.sub64 s 73 72 69
  let s := Ptx.add64 s 74 70 73
  let s := Ptx.add64 s 75 74 71
  let s := Ptx.setpLt s 14 75 73
  let s := Ptx.selp64 s 76 4294967295 0 14
  let s := Ptx.add64 s 77 76 75
  let s := Ptx.setpGt s 15 77 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 78 77 4294967295
  let s := Ptx.selp64 s 79 78 77 15
  let s := Ptx.sub64 s 80 (Ptx.Value.immediate 18446744069414584321) 79
  let s := Ptx.sub64 s 81 65 80
  let s := Ptx.setpLt s 16 65 80
  let s := Ptx.add64 s 82 81 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 83 82 81 16
  let s := Ptx.mov64 s 84 2
  let s := Ptx.mulHi64 s 85 84 83
  let s := Ptx.shl64Imm s 86 83 1
  let s := Ptx.shr64Imm s 87 85 32
  let s := Ptx.and64ImmRight s 88 85 4294967295
  let s := Ptx.sub64 s 89 86 87
  let s := Ptx.setpLt s 17 86 87
  let s := Ptx.selp64 s 90 (Ptx.Value.immediate 18446744069414584321) 0 17
  let s := Ptx.shl64Imm s 91 85 32
  let s := Ptx.sub64 s 92 91 88
  let s := Ptx.add64 s 93 89 92
  let s := Ptx.add64 s 94 93 90
  let s := Ptx.setpLt s 18 94 92
  let s := Ptx.selp64 s 95 4294967295 0 18
  let s := Ptx.add64 s 96 95 94
  let s := Ptx.setpGt s 19 96 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 97 96 4294967295
  let s := Ptx.selp64 s 98 97 96 19
  let s := Ptx.sub64 s 99 (Ptx.Value.immediate 18446744069414584321) 98
  let s := Ptx.sub64 s 100 51 99
  let s := Ptx.setpLt s 20 51 99
  let s := Ptx.add64 s 101 100 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 102 101 100 20
  let s := Ptx.mulHi64 s 103 27 2
  let s := Ptx.mulLo64 s 104 27 2
  let s := Ptx.shr64Imm s 105 103 32
  let s := Ptx.and64ImmRight s 106 103 4294967295
  let s := Ptx.sub64 s 107 104 105
  let s := Ptx.setpLt s 21 104 105
  let s := Ptx.selp64 s 108 (Ptx.Value.immediate 18446744069414584321) 0 21
  let s := Ptx.shl64Imm s 109 103 32
  let s := Ptx.sub64 s 110 109 106
  let s := Ptx.add64 s 111 107 110
  let s := Ptx.add64 s 112 111 108
  let s := Ptx.setpLt s 22 112 110
  let s := Ptx.selp64 s 113 4294967295 0 22
  let s := Ptx.add64 s 114 113 112
  let s := Ptx.setpGt s 23 114 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 115 114 4294967295
  let s := Ptx.selp64 s 116 115 114 23
  let s := Ptx.mulHi64 s 117 32 3
  let s := Ptx.mulLo64 s 118 32 3
  let s := Ptx.shr64Imm s 119 117 32
  let s := Ptx.and64ImmRight s 120 117 4294967295
  let s := Ptx.sub64 s 121 118 119
  let s := Ptx.setpLt s 24 118 119
  let s := Ptx.selp64 s 122 (Ptx.Value.immediate 18446744069414584321) 0 24
  let s := Ptx.shl64Imm s 123 117 32
  let s := Ptx.sub64 s 124 123 120
  let s := Ptx.add64 s 125 121 124
  let s := Ptx.add64 s 126 125 122
  let s := Ptx.setpLt s 25 126 124
  let s := Ptx.selp64 s 127 4294967295 0 25
  let s := Ptx.add64 s 128 127 126
  let s := Ptx.setpGt s 26 128 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 129 128 4294967295
  let s := Ptx.selp64 s 130 129 128 26
  let s := Ptx.sub64 s 131 (Ptx.Value.immediate 18446744069414584321) 130
  let s := Ptx.sub64 s 132 116 131
  let s := Ptx.setpLt s 27 116 131
  let s := Ptx.add64 s 133 132 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 134 133 132 27
  let s := Ptx.mulHi64 s 135 37 1
  let s := Ptx.mulLo64 s 136 37 1
  let s := Ptx.shr64Imm s 137 135 32
  let s := Ptx.and64ImmRight s 138 135 4294967295
  let s := Ptx.sub64 s 139 136 137
  let s := Ptx.setpLt s 28 136 137
  let s := Ptx.selp64 s 140 (Ptx.Value.immediate 18446744069414584321) 0 28
  let s := Ptx.shl64Imm s 141 135 32
  let s := Ptx.sub64 s 142 141 138
  let s := Ptx.add64 s 143 139 142
  let s := Ptx.add64 s 144 143 140
  let s := Ptx.setpLt s 29 144 142
  let s := Ptx.selp64 s 145 4294967295 0 29
  let s := Ptx.add64 s 146 145 144
  let s := Ptx.setpGt s 30 146 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 147 146 4294967295
  let s := Ptx.selp64 s 148 147 146 30
  let s := Ptx.mulHi64 s 149 84 148
  let s := Ptx.shl64Imm s 150 148 1
  let s := Ptx.shr64Imm s 151 149 32
  let s := Ptx.and64ImmRight s 152 149 4294967295
  let s := Ptx.sub64 s 153 150 151
  let s := Ptx.setpLt s 31 150 151
  let s := Ptx.selp64 s 154 (Ptx.Value.immediate 18446744069414584321) 0 31
  let s := Ptx.shl64Imm s 155 149 32
  let s := Ptx.sub64 s 156 155 152
  let s := Ptx.add64 s 157 153 156
  let s := Ptx.add64 s 158 157 154
  let s := Ptx.setpLt s 32 158 156
  let s := Ptx.selp64 s 159 4294967295 0 32
  let s := Ptx.add64 s 160 159 158
  let s := Ptx.setpGt s 33 160 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 161 160 4294967295
  let s := Ptx.selp64 s 162 161 160 33
  let s := Ptx.sub64 s 163 (Ptx.Value.immediate 18446744069414584321) 162
  let s := Ptx.sub64 s 164 134 163
  let s := Ptx.setpLt s 34 134 163
  let s := Ptx.add64 s 165 164 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 166 165 164 34
  let s := Ptx.mulHi64 s 167 27 1
  let s := Ptx.mulLo64 s 168 27 1
  let s := Ptx.shr64Imm s 169 167 32
  let s := Ptx.and64ImmRight s 170 167 4294967295
  let s := Ptx.sub64 s 171 168 169
  let s := Ptx.setpLt s 35 168 169
  let s := Ptx.selp64 s 172 (Ptx.Value.immediate 18446744069414584321) 0 35
  let s := Ptx.shl64Imm s 173 167 32
  let s := Ptx.sub64 s 174 173 170
  let s := Ptx.add64 s 175 171 174
  let s := Ptx.add64 s 176 175 172
  let s := Ptx.setpLt s 36 176 174
  let s := Ptx.selp64 s 177 4294967295 0 36
  let s := Ptx.add64 s 178 177 176
  let s := Ptx.setpGt s 37 178 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 179 178 4294967295
  let s := Ptx.selp64 s 180 179 178 37
  let s := Ptx.mulHi64 s 181 32 2
  let s := Ptx.mulLo64 s 182 32 2
  let s := Ptx.shr64Imm s 183 181 32
  let s := Ptx.and64ImmRight s 184 181 4294967295
  let s := Ptx.sub64 s 185 182 183
  let s := Ptx.setpLt s 38 182 183
  let s := Ptx.selp64 s 186 (Ptx.Value.immediate 18446744069414584321) 0 38
  let s := Ptx.shl64Imm s 187 181 32
  let s := Ptx.sub64 s 188 187 184
  let s := Ptx.add64 s 189 185 188
  let s := Ptx.add64 s 190 189 186
  let s := Ptx.setpLt s 39 190 188
  let s := Ptx.selp64 s 191 4294967295 0 39
  let s := Ptx.add64 s 192 191 190
  let s := Ptx.setpGt s 40 192 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 193 192 4294967295
  let s := Ptx.selp64 s 194 193 192 40
  let s := Ptx.mulHi64 s 195 37 3
  let s := Ptx.mulLo64 s 196 37 3
  let s := Ptx.shr64Imm s 197 195 32
  let s := Ptx.and64ImmRight s 198 195 4294967295
  let s := Ptx.sub64 s 199 196 197
  let s := Ptx.setpLt s 41 196 197
  let s := Ptx.selp64 s 200 (Ptx.Value.immediate 18446744069414584321) 0 41
  let s := Ptx.shl64Imm s 201 195 32
  let s := Ptx.sub64 s 202 201 198
  let s := Ptx.add64 s 203 199 202
  let s := Ptx.add64 s 204 203 200
  let s := Ptx.setpLt s 42 204 202
  let s := Ptx.selp64 s 205 4294967295 0 42
  let s := Ptx.add64 s 206 205 204
  let s := Ptx.setpGt s 43 206 (Ptx.Value.immediate 18446744069414584320)
  let s := Ptx.add64 s 207 206 4294967295
  let s := Ptx.selp64 s 208 207 206 43
  let s := Ptx.sub64 s 209 (Ptx.Value.immediate 18446744069414584321) 208
  let s := Ptx.sub64 s 210 194 209
  let s := Ptx.setpLt s 44 194 209
  let s := Ptx.add64 s 211 210 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 212 211 210 44
  let s := Ptx.sub64 s 213 (Ptx.Value.immediate 18446744069414584321) 212
  let s := Ptx.sub64 s 214 180 213
  let s := Ptx.setpLt s 45 180 213
  let s := Ptx.add64 s 215 214 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 216 215 214 45
  let s := Ptx.sub64 s 217 (Ptx.Value.immediate 18446744069414584321) 102
  let s := Ptx.sub64 s 218 24 217
  let s := Ptx.setpLt s 46 24 217
  let s := Ptx.add64 s 219 218 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 220 219 218 46
  let s := Ptx.sub64 s 221 (Ptx.Value.immediate 18446744069414584321) 166
  let s := Ptx.sub64 s 222 29 221
  let s := Ptx.setpLt s 47 29 221
  let s := Ptx.add64 s 223 222 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 224 223 222 47
  let s := Ptx.sub64 s 225 (Ptx.Value.immediate 18446744069414584321) 216
  let s := Ptx.sub64 s 226 34 225
  let s := Ptx.setpLt s 48 34 225
  let s := Ptx.add64 s 227 226 (Ptx.Value.immediate 18446744069414584321)
  let s := Ptx.selp64 s 228 227 226 48
  let s := Ptx.cvta s 229 12
  let s := Ptx.madLo64ImmRight s 230 4 24 229
  let s := Ptx.stGlobalOffset s launch 230 0 220
  let s := Ptx.stGlobalOffset s launch 230 8 224
  let s := Ptx.stGlobalOffset s launch 230 16 228
  s

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

def ptxArithmeticSSA (launch : Launch) (thread : Ptx.Thread) : FieldExt :=
  let rd11 := launch.inputBase
  let rd1 := Field.toWord launch.challenge.c2
  let rd2 := Field.toWord launch.challenge.c1
  let rd3 := Field.toWord launch.challenge.c0
  let rd12 := launch.outputBase
  let rd13 := launch.outputCount
  let rd14 := launch.lowHalf
  let r1 := thread.blockIndex
  let r2 := thread.blockDim
  let r3 := thread.threadIndex
  let r4 := (r1 * r2 + r3) % 0x100000000
  let rd4 := r4
  let rd15 := rd14 &&& 18446744069414584320
  let (rd231, rd232) :=
    if rd15 != 0 then
      (rd4 / rd14, Word.wrapSub rd4 ((rd4 / rd14) * rd14 % Word.base))
    else
      (rd4 / rd14, rd4 % rd14)
  let rd17 := (rd14 * rd231) % Word.base
  let rd18 := (rd17 * 2 ^ 1) % Word.base
  let rd19 := (rd232 + rd18) % Word.base
  let rd20 := rd11
  let rd21 := ptxZeroAddress launch thread
  let rd22 := ptxOneAddress launch thread
  let rd23 := Ptx.inputWord launch ((rd22 + 0 - launch.inputBase) / 8)
  let rd24 := Ptx.inputWord launch ((rd21 + 0 - launch.inputBase) / 8)
  let rd25 := Word.wrapSub rd23 rd24
  let p4 := rd23 < rd24
  let rd26 := (rd25 + 18446744069414584321) % Word.base
  let rd27 := if p4 then rd26 else rd25
  let rd28 := Ptx.inputWord launch ((rd22 + 8 - launch.inputBase) / 8)
  let rd29 := Ptx.inputWord launch ((rd21 + 8 - launch.inputBase) / 8)
  let rd30 := Word.wrapSub rd28 rd29
  let p5 := rd28 < rd29
  let rd31 := (rd30 + 18446744069414584321) % Word.base
  let rd32 := if p5 then rd31 else rd30
  let rd33 := Ptx.inputWord launch ((rd22 + 16 - launch.inputBase) / 8)
  let rd34 := Ptx.inputWord launch ((rd21 + 16 - launch.inputBase) / 8)
  let rd35 := Word.wrapSub rd33 rd34
  let p6 := rd33 < rd34
  let rd36 := (rd35 + 18446744069414584321) % Word.base
  let rd37 := if p6 then rd36 else rd35
  let rd38 := (rd27 * rd3) / Word.base
  let rd39 := (rd27 * rd3) % Word.base
  let rd40 := rd38 / 2 ^ 32
  let rd41 := rd38 &&& 4294967295
  let rd42 := Word.wrapSub rd39 rd40
  let p7 := rd39 < rd40
  let rd43 := if p7 then 18446744069414584321 else 0
  let rd44 := (rd38 * 2 ^ 32) % Word.base
  let rd45 := Word.wrapSub rd44 rd41
  let rd46 := (rd42 + rd45) % Word.base
  let rd47 := (rd46 + rd43) % Word.base
  let p8 := rd47 < rd45
  let rd48 := if p8 then 4294967295 else 0
  let rd49 := (rd48 + rd47) % Word.base
  let p9 := rd49 > 18446744069414584320
  let rd50 := (rd49 + 4294967295) % Word.base
  let rd51 := if p9 then rd50 else rd49
  let rd52 := (rd32 * rd1) / Word.base
  let rd53 := (rd32 * rd1) % Word.base
  let rd54 := rd52 / 2 ^ 32
  let rd55 := rd52 &&& 4294967295
  let rd56 := Word.wrapSub rd53 rd54
  let p10 := rd53 < rd54
  let rd57 := if p10 then 18446744069414584321 else 0
  let rd58 := (rd52 * 2 ^ 32) % Word.base
  let rd59 := Word.wrapSub rd58 rd55
  let rd60 := (rd56 + rd59) % Word.base
  let rd61 := (rd60 + rd57) % Word.base
  let p11 := rd61 < rd59
  let rd62 := if p11 then 4294967295 else 0
  let rd63 := (rd62 + rd61) % Word.base
  let p12 := rd63 > 18446744069414584320
  let rd64 := (rd63 + 4294967295) % Word.base
  let rd65 := if p12 then rd64 else rd63
  let rd66 := (rd37 * rd2) / Word.base
  let rd67 := (rd37 * rd2) % Word.base
  let rd68 := rd66 / 2 ^ 32
  let rd69 := rd66 &&& 4294967295
  let rd70 := Word.wrapSub rd67 rd68
  let p13 := rd67 < rd68
  let rd71 := if p13 then 18446744069414584321 else 0
  let rd72 := (rd66 * 2 ^ 32) % Word.base
  let rd73 := Word.wrapSub rd72 rd69
  let rd74 := (rd70 + rd73) % Word.base
  let rd75 := (rd74 + rd71) % Word.base
  let p14 := rd75 < rd73
  let rd76 := if p14 then 4294967295 else 0
  let rd77 := (rd76 + rd75) % Word.base
  let p15 := rd77 > 18446744069414584320
  let rd78 := (rd77 + 4294967295) % Word.base
  let rd79 := if p15 then rd78 else rd77
  let rd80 := Word.wrapSub 18446744069414584321 rd79
  let rd81 := Word.wrapSub rd65 rd80
  let p16 := rd65 < rd80
  let rd82 := (rd81 + 18446744069414584321) % Word.base
  let rd83 := if p16 then rd82 else rd81
  let rd84 := 2
  let rd85 := (rd84 * rd83) / Word.base
  let rd86 := (rd83 * 2 ^ 1) % Word.base
  let rd87 := rd85 / 2 ^ 32
  let rd88 := rd85 &&& 4294967295
  let rd89 := Word.wrapSub rd86 rd87
  let p17 := rd86 < rd87
  let rd90 := if p17 then 18446744069414584321 else 0
  let rd91 := (rd85 * 2 ^ 32) % Word.base
  let rd92 := Word.wrapSub rd91 rd88
  let rd93 := (rd89 + rd92) % Word.base
  let rd94 := (rd93 + rd90) % Word.base
  let p18 := rd94 < rd92
  let rd95 := if p18 then 4294967295 else 0
  let rd96 := (rd95 + rd94) % Word.base
  let p19 := rd96 > 18446744069414584320
  let rd97 := (rd96 + 4294967295) % Word.base
  let rd98 := if p19 then rd97 else rd96
  let rd99 := Word.wrapSub 18446744069414584321 rd98
  let rd100 := Word.wrapSub rd51 rd99
  let p20 := rd51 < rd99
  let rd101 := (rd100 + 18446744069414584321) % Word.base
  let rd102 := if p20 then rd101 else rd100
  let rd103 := (rd27 * rd2) / Word.base
  let rd104 := (rd27 * rd2) % Word.base
  let rd105 := rd103 / 2 ^ 32
  let rd106 := rd103 &&& 4294967295
  let rd107 := Word.wrapSub rd104 rd105
  let p21 := rd104 < rd105
  let rd108 := if p21 then 18446744069414584321 else 0
  let rd109 := (rd103 * 2 ^ 32) % Word.base
  let rd110 := Word.wrapSub rd109 rd106
  let rd111 := (rd107 + rd110) % Word.base
  let rd112 := (rd111 + rd108) % Word.base
  let p22 := rd112 < rd110
  let rd113 := if p22 then 4294967295 else 0
  let rd114 := (rd113 + rd112) % Word.base
  let p23 := rd114 > 18446744069414584320
  let rd115 := (rd114 + 4294967295) % Word.base
  let rd116 := if p23 then rd115 else rd114
  let rd117 := (rd32 * rd3) / Word.base
  let rd118 := (rd32 * rd3) % Word.base
  let rd119 := rd117 / 2 ^ 32
  let rd120 := rd117 &&& 4294967295
  let rd121 := Word.wrapSub rd118 rd119
  let p24 := rd118 < rd119
  let rd122 := if p24 then 18446744069414584321 else 0
  let rd123 := (rd117 * 2 ^ 32) % Word.base
  let rd124 := Word.wrapSub rd123 rd120
  let rd125 := (rd121 + rd124) % Word.base
  let rd126 := (rd125 + rd122) % Word.base
  let p25 := rd126 < rd124
  let rd127 := if p25 then 4294967295 else 0
  let rd128 := (rd127 + rd126) % Word.base
  let p26 := rd128 > 18446744069414584320
  let rd129 := (rd128 + 4294967295) % Word.base
  let rd130 := if p26 then rd129 else rd128
  let rd131 := Word.wrapSub 18446744069414584321 rd130
  let rd132 := Word.wrapSub rd116 rd131
  let p27 := rd116 < rd131
  let rd133 := (rd132 + 18446744069414584321) % Word.base
  let rd134 := if p27 then rd133 else rd132
  let rd135 := (rd37 * rd1) / Word.base
  let rd136 := (rd37 * rd1) % Word.base
  let rd137 := rd135 / 2 ^ 32
  let rd138 := rd135 &&& 4294967295
  let rd139 := Word.wrapSub rd136 rd137
  let p28 := rd136 < rd137
  let rd140 := if p28 then 18446744069414584321 else 0
  let rd141 := (rd135 * 2 ^ 32) % Word.base
  let rd142 := Word.wrapSub rd141 rd138
  let rd143 := (rd139 + rd142) % Word.base
  let rd144 := (rd143 + rd140) % Word.base
  let p29 := rd144 < rd142
  let rd145 := if p29 then 4294967295 else 0
  let rd146 := (rd145 + rd144) % Word.base
  let p30 := rd146 > 18446744069414584320
  let rd147 := (rd146 + 4294967295) % Word.base
  let rd148 := if p30 then rd147 else rd146
  let rd149 := (rd84 * rd148) / Word.base
  let rd150 := (rd148 * 2 ^ 1) % Word.base
  let rd151 := rd149 / 2 ^ 32
  let rd152 := rd149 &&& 4294967295
  let rd153 := Word.wrapSub rd150 rd151
  let p31 := rd150 < rd151
  let rd154 := if p31 then 18446744069414584321 else 0
  let rd155 := (rd149 * 2 ^ 32) % Word.base
  let rd156 := Word.wrapSub rd155 rd152
  let rd157 := (rd153 + rd156) % Word.base
  let rd158 := (rd157 + rd154) % Word.base
  let p32 := rd158 < rd156
  let rd159 := if p32 then 4294967295 else 0
  let rd160 := (rd159 + rd158) % Word.base
  let p33 := rd160 > 18446744069414584320
  let rd161 := (rd160 + 4294967295) % Word.base
  let rd162 := if p33 then rd161 else rd160
  let rd163 := Word.wrapSub 18446744069414584321 rd162
  let rd164 := Word.wrapSub rd134 rd163
  let p34 := rd134 < rd163
  let rd165 := (rd164 + 18446744069414584321) % Word.base
  let rd166 := if p34 then rd165 else rd164
  let rd167 := (rd27 * rd1) / Word.base
  let rd168 := (rd27 * rd1) % Word.base
  let rd169 := rd167 / 2 ^ 32
  let rd170 := rd167 &&& 4294967295
  let rd171 := Word.wrapSub rd168 rd169
  let p35 := rd168 < rd169
  let rd172 := if p35 then 18446744069414584321 else 0
  let rd173 := (rd167 * 2 ^ 32) % Word.base
  let rd174 := Word.wrapSub rd173 rd170
  let rd175 := (rd171 + rd174) % Word.base
  let rd176 := (rd175 + rd172) % Word.base
  let p36 := rd176 < rd174
  let rd177 := if p36 then 4294967295 else 0
  let rd178 := (rd177 + rd176) % Word.base
  let p37 := rd178 > 18446744069414584320
  let rd179 := (rd178 + 4294967295) % Word.base
  let rd180 := if p37 then rd179 else rd178
  let rd181 := (rd32 * rd2) / Word.base
  let rd182 := (rd32 * rd2) % Word.base
  let rd183 := rd181 / 2 ^ 32
  let rd184 := rd181 &&& 4294967295
  let rd185 := Word.wrapSub rd182 rd183
  let p38 := rd182 < rd183
  let rd186 := if p38 then 18446744069414584321 else 0
  let rd187 := (rd181 * 2 ^ 32) % Word.base
  let rd188 := Word.wrapSub rd187 rd184
  let rd189 := (rd185 + rd188) % Word.base
  let rd190 := (rd189 + rd186) % Word.base
  let p39 := rd190 < rd188
  let rd191 := if p39 then 4294967295 else 0
  let rd192 := (rd191 + rd190) % Word.base
  let p40 := rd192 > 18446744069414584320
  let rd193 := (rd192 + 4294967295) % Word.base
  let rd194 := if p40 then rd193 else rd192
  let rd195 := (rd37 * rd3) / Word.base
  let rd196 := (rd37 * rd3) % Word.base
  let rd197 := rd195 / 2 ^ 32
  let rd198 := rd195 &&& 4294967295
  let rd199 := Word.wrapSub rd196 rd197
  let p41 := rd196 < rd197
  let rd200 := if p41 then 18446744069414584321 else 0
  let rd201 := (rd195 * 2 ^ 32) % Word.base
  let rd202 := Word.wrapSub rd201 rd198
  let rd203 := (rd199 + rd202) % Word.base
  let rd204 := (rd203 + rd200) % Word.base
  let p42 := rd204 < rd202
  let rd205 := if p42 then 4294967295 else 0
  let rd206 := (rd205 + rd204) % Word.base
  let p43 := rd206 > 18446744069414584320
  let rd207 := (rd206 + 4294967295) % Word.base
  let rd208 := if p43 then rd207 else rd206
  let rd209 := Word.wrapSub 18446744069414584321 rd208
  let rd210 := Word.wrapSub rd194 rd209
  let p44 := rd194 < rd209
  let rd211 := (rd210 + 18446744069414584321) % Word.base
  let rd212 := if p44 then rd211 else rd210
  let rd213 := Word.wrapSub 18446744069414584321 rd212
  let rd214 := Word.wrapSub rd180 rd213
  let p45 := rd180 < rd213
  let rd215 := (rd214 + 18446744069414584321) % Word.base
  let rd216 := if p45 then rd215 else rd214
  let rd217 := Word.wrapSub 18446744069414584321 rd102
  let rd218 := Word.wrapSub rd24 rd217
  let p46 := rd24 < rd217
  let rd219 := (rd218 + 18446744069414584321) % Word.base
  let rd220 := if p46 then rd219 else rd218
  let rd221 := Word.wrapSub 18446744069414584321 rd166
  let rd222 := Word.wrapSub rd29 rd221
  let p47 := rd29 < rd221
  let rd223 := (rd222 + 18446744069414584321) % Word.base
  let rd224 := if p47 then rd223 else rd222
  let rd225 := Word.wrapSub 18446744069414584321 rd216
  let rd226 := Word.wrapSub rd34 rd225
  let p48 := rd34 < rd225
  let rd227 := (rd226 + 18446744069414584321) % Word.base
  let rd228 := if p48 then rd227 else rd226
  let rd229 := rd12
  let rd230 := (rd4 * 24 + rd229) % Word.base
  ⟨Field.ofWord rd220, Field.ofWord rd224, Field.ofWord rd228⟩

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

theorem word_highWord_pow_fold (hi : Nat) :
    hi / 2 ^ 32 = Word.highWord hi := by
  rfl

theorem word_lowWord_mask_fold (hi : Nat) :
    hi &&& 0xffffffff = Word.lowWord hi := by
  simpa [Word.lowWord] using (Nat.and_two_pow_sub_one_eq_mod hi 32)

theorem word_mul_fold (a b : Nat) :
    Word.reduce128 ((a * b) / Word.base) ((a * b) % Word.base) = Word.mul a b := rfl

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

theorem word_sub_ptx_fold (a b : Nat) :
    (if a < b then (Word.wrapSub a b + Word.modulus) % Word.base
      else Word.wrapSub a b) = Word.sub a b := by
  by_cases h : a < b
  · simp only [Word.sub, Word.wrapSub, Word.modulus, Word.base, Word.epsilon,
      h, ↓reduceIte]
    omega
  · simp only [Word.sub, Word.wrapSub, Word.modulus, Word.base, Word.epsilon,
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
  · simp only [Word.add, h, ↓reduceIte]
    exact word_wrapSub_epsilon_ptx _
  · simp only [Word.add, h, ↓reduceIte]

def ptxRunThreadSSA (launch : Launch) (thread : Ptx.Thread) : Option FieldExt :=
  if launch.lowHalf = 0 then
    none
  else if launch.outputCount ≤ thread.index then
    none
  else
    some (ptxArithmeticSSA launch thread)

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
