set_option maxRecDepth 10000

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
  let notB := modulus - b
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
  by_cases h : a < modulus - b
  · have hd : wrapSub a (modulus - b) = base - (modulus - b - a) :=
      wrapSub_of_lt haBase hnotBase h
    have hdBase : wrapSub a (modulus - b) < base := wrapSub_lt _ _
    have he : epsilon ≤ wrapSub a (modulus - b) := by
      simp [base, modulus, epsilon] at *
      omega
    simp only [add, h, ↓reduceIte]
    rw [wrapSub_of_le hdBase he, hd]
    rw [Nat.mod_eq_of_lt]
    · simp [base, modulus, epsilon] at *
      omega
    · simp [modulus] at *
      omega
  · have hle : modulus - b ≤ a := Nat.le_of_not_gt h
    have hd : wrapSub a (modulus - b) = a - (modulus - b) :=
      wrapSub_of_le haBase hle
    simp only [add, h, ↓reduceIte, hd]
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

/-- A concrete valid launch used to establish non-vacuity. -/
def exampleLaunch : Launch where
  input := #[FieldExt.ofInt ⟨1, 2, 3⟩, FieldExt.ofInt ⟨4, 5, 6⟩]
  outputCount := 1
  lowHalf := 1
  challenge := FieldExt.ofInt ⟨7, 11, 13⟩

theorem exampleLaunch_valid : exampleLaunch.Valid := by
  constructor <;> decide

theorem runThread_nonvacuous : Launch.runThread exampleLaunch 0 ≠ some FieldExt.zero := by
  decide

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
