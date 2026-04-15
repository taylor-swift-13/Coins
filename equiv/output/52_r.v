(* 
   This file proves that the LLM-generated specification (below_threshold_spec) 
   implies the human-written specification (problem_52_spec).
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (LLM-generated spec)                          *)
(* ================================================================= *)

Definition below_threshold_spec (l : list Z) (t : Z) (res : bool) : Prop :=
  res = true <-> (forall x, In x l -> x < t).

(* ================================================================= *)
(* Second Specification (Human-written spec)                         *)
(* ================================================================= *)

(* Pre: no special constraints for `below_threshold` *)
Definition problem_52_pre (l : list Z) : Prop := True.

Definition problem_52_spec (l : list Z) (t : Z) (output : bool) : Prop :=
  (forall x, In x l -> x < t) <-> (output = true).

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

Theorem llm_spec_implies_human_spec : forall (l : list Z) (t : Z) (output : bool),
  below_threshold_spec l t output -> problem_52_spec l t output.
Proof.
  intros l t output H.
  (* Unfold definitions to expose the logical structure *)
  unfold below_threshold_spec in H.
  unfold problem_52_spec.
  
  (* The LLM spec states: (output = true) <-> (All x < t) *)
  (* The Human spec states: (All x < t) <-> (output = true) *)
  (* We can use symmetry of bi-implication or rewriting *)
  symmetry.
  apply H.
Qed.