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

#print axioms FieldExt.foldFused_correct
#print axioms FieldExt.foldFused_nonvacuous
#print axioms Launch.oneIndex_lt_twice_output
#print axioms Launch.runThread_correct
#print axioms Launch.runThread_inactive
#print axioms runThread_nonvacuous

end Kernel

end Weightproof
