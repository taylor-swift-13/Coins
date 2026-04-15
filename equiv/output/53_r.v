Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (LLM-generated spec)                          *)
(* ================================================================= *)

Definition add_spec (x y result : Z) : Prop :=
  result = x + y.

(* ================================================================= *)
(* Second Specification (Human-written spec)                         *)
(* ================================================================= *)

(* Pre: no special constraints for `add` *)
Definition problem_53_pre (x y : Z) : Prop := True.

Definition problem_53_spec (x : Z) (y : Z) (output : Z) : Prop :=
  output = x + y.

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

Theorem spec1_implies_spec2 : forall (x y result : Z),
  problem_53_pre x y ->
  add_spec x y result ->
  problem_53_spec x y result.
Proof.
  intros x y result Hpre Hspec.
  unfold add_spec in Hspec.
  unfold problem_53_spec.
  assumption.
Qed.