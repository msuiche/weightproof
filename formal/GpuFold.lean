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
end Weightproof
