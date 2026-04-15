(* 
   This file proves that the human-written specification (problem_72_spec) 
   implies the LLM-generated specification (will_it_fly_spec).
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

(* ========================================================================= *)
(* First Specification (Human-written) *)
(* ========================================================================= *)

Definition problem_72_pre (q : list Z) (w : Z) : Prop := True.

(* will_it_fly 函数的程序规约 *)
Definition problem_72_spec (q : list Z) (w : Z) (output : bool) : Prop :=
  (output = true <-> (q = rev q) /\ (fold_left (fun acc x => acc + x) q 0 <= w)).

(* ========================================================================= *)
(* Second Specification (LLM-generated) *)
(* ========================================================================= *)

Definition sum_list (l : list Z) : Z :=
  fold_right Z.add 0 l.

Definition will_it_fly_spec (q : list Z) (w : Z) (res : bool) : Prop :=
  res = true <-> (q = rev q /\ sum_list q <= w).

(* ========================================================================= *)
(* Proof of Implication *)
(* ========================================================================= *)

(* 
   To prove that spec1 implies spec2, we must show that the definition of sum 
   used in spec1 (fold_left with accumulator) is equivalent to the definition 
   of sum used in spec2 (fold_right).
*)

(* Lemma: fold_left with addition is equivalent to fold_right with addition, accounting for the accumulator. *)
Lemma fold_left_add_assoc : forall (l : list Z) (acc : Z),
  fold_left (fun acc x => acc + x) l acc = acc + fold_right Z.add 0 l.
Proof.
  intros l.
  induction l as [| a l' IH].
  - (* Base case: list is empty *)
    intros acc.
    simpl.
    rewrite Z.add_0_r. (* acc + 0 = acc *)
    reflexivity.
  - (* Inductive step: list is a :: l' *)
    intros acc.
    simpl.
    (* Apply IH with (acc + a) as the new accumulator *)
    rewrite IH. 
    (* We now have: (acc + a) + fold_right Z.add 0 l' *)
    (* We want: acc + (a + fold_right Z.add 0 l') *)
    rewrite Z.add_assoc.
    reflexivity.
Qed.

(* Lemma: Specializing the previous lemma for accumulator 0 matches sum_list. *)
Lemma fold_left_eq_sum_list : forall (l : list Z),
  fold_left (fun acc x => acc + x) l 0 = sum_list l.
Proof.
  intros l.
  unfold sum_list.
  rewrite fold_left_add_assoc.
  rewrite Z.add_0_l. (* 0 + x = x *)
  reflexivity.
Qed.

(* Main Theorem: Spec 1 implies Spec 2 *)
Theorem spec1_implies_spec2 : forall (q : list Z) (w : Z) (output : bool),
  problem_72_spec q w output -> will_it_fly_spec q w output.
Proof.
  intros q w output H_spec1.
  unfold problem_72_spec in H_spec1.
  unfold will_it_fly_spec.
  
  (* The specifications are identical except for the summation implementation.
     We rewrite the summation in the goal to match the hypothesis. *)
  rewrite <- fold_left_eq_sum_list.
  
  (* Now the goal matches the hypothesis exactly. *)
  assumption.
Qed.