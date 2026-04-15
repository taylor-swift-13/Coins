(* 
   This file proves that the human-written specification (problem_52_spec) 
   implies the LLM-generated specification (below_threshold_spec).
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (Human-written spec)                          *)
(* ================================================================= *)

(* Pre: no special constraints for `below_threshold` *)
Definition problem_52_pre (l : list Z) : Prop := True.

Definition problem_52_spec (l : list Z) (t : Z) (output : bool) : Prop :=
  (forall x, In x l -> x < t) <-> (output = true).

(* ================================================================= *)
(* Second Specification (LLM-generated spec)                         *)
(* ================================================================= *)

Definition below_threshold_spec (l : list Z) (t : Z) (res : bool) : Prop :=
  res = true <-> (forall x, In x l -> x < t).

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

Theorem human_spec_implies_llm_spec : forall (l : list Z) (t : Z) (output : bool),
  problem_52_spec l t output -> below_threshold_spec l t output.
Proof.
  intros l t output H.
  (* Unfold definitions to expose the logical structure *)
  unfold problem_52_spec in H.
  unfold below_threshold_spec.
  
  (* The human spec states: (All x < t) <-> (output = true) *)
  (* The LLM spec states:   (output = true) <-> (All x < t) *)
  (* Equality and bi-implication are symmetric. *)
  symmetry.
  apply H.
Qed.