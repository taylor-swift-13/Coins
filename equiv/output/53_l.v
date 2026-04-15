Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (Human-written spec)                          *)
(* ================================================================= *)

(* Pre: no special constraints for `add` *)
Definition problem_53_pre (x y : Z) : Prop := True.

Definition problem_53_spec (x : Z) (y : Z) (output : Z) : Prop :=
  output = x + y.

(* ================================================================= *)
(* Second Specification (LLM-generated spec)                         *)
(* ================================================================= *)

Definition add_spec (x y result : Z) : Prop :=
  result = x + y.

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

Theorem spec1_implies_spec2 : forall (x y output : Z),
  problem_53_pre x y ->
  problem_53_spec x y output ->
  add_spec x y output.
Proof.
  intros x y output Hpre Hspec.
  unfold problem_53_spec in Hspec.
  unfold add_spec.
  assumption.
Qed.