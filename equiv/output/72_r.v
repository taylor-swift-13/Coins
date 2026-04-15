(* 
   This file proves that the LLM-generated specification (will_it_fly_spec) 
   implies the human-written specification (problem_72_spec).
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

(* ========================================================================= *)
(* First Specification (LLM-generated) *)
(* ========================================================================= *)

Definition sum_list (l : list Z) : Z :=
  fold_right Z.add 0 l.

Definition will_it_fly_spec (q : list Z) (w : Z) (res : bool) : Prop :=
  res = true <-> (q = rev q /\ sum_list q <= w).

(* ========================================================================= *)
(* Second Specification (Human-written) *)
(* ========================================================================= *)

Definition problem_72_pre (q : list Z) (w : Z) : Prop := True.

(* will_it_fly 函数的程序规约*)
Definition  problem_72_spec (q : list Z) (w : Z) (output : bool) : Prop :=
  (output = true <-> (q = rev q) /\ (fold_left (fun acc x => acc + x) q 0 <= w)).

(* ========================================================================= *)
(* Proof of Implication *)
(* ========================================================================= *)

(* 
   To prove equivalence, we need to show that the summation defined via 
   fold_left (in spec 2) is equal to the summation defined via fold_right (in spec 1).
*)

Lemma fold_left_add_assoc : forall (l : list Z) (acc : Z),
  fold_left (fun acc x => acc + x) l acc = acc + fold_right Z.add 0 l.
Proof.
  intros l.
  induction l as [| x l' IH].
  - (* Base case *)
    intros acc. simpl. 
    rewrite Z.add_0_r. 
    reflexivity.
  - (* Inductive step *)
    intros acc. simpl.
    rewrite IH.
    (* We have (acc + x) + fold_right ... *)
    (* We want acc + (x + fold_right ...) *)
    rewrite Z.add_assoc.
    reflexivity.
Qed.

Lemma fold_left_eq_sum_list : forall (l : list Z),
  fold_left (fun acc x => acc + x) l 0 = sum_list l.
Proof.
  intros l.
  unfold sum_list.
  rewrite fold_left_add_assoc.
  rewrite Z.add_0_l.
  reflexivity.
Qed.

Theorem spec1_implies_spec2 : forall (q : list Z) (w : Z) (output : bool),
  will_it_fly_spec q w output -> problem_72_spec q w output.
Proof.
  intros q w output H.
  unfold will_it_fly_spec in H.
  unfold problem_72_spec.
  
  (* Replace the fold_left summation in the goal with sum_list to match hypothesis *)
  rewrite fold_left_eq_sum_list.
  
  (* Now the goal is identical to the hypothesis H *)
  assumption.
Qed.