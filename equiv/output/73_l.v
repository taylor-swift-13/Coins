(** 
   Proof that Spec 1 (Human-written) implies Spec 2 (LLM-generated).
   
   Spec 1: problem_73_spec (using smallest_change_impl)
   Spec 2: smallest_change_spec
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.micromega.Lia.
Import ListNotations.
Open Scope Z_scope.

(* ========================================================================= *)
(*                           First Specification                             *)
(* ========================================================================= *)

(*
  count_diff l1 l2 acc := Calculate the number of different elements between l1 and l2
*)
Fixpoint count_diff (l1 l2: list Z) (acc: Z): Z :=
  match l1, l2 with
  | [], _ => acc
  | _, [] => acc
  | h1 :: t1, h2 :: t2 =>
    if Z.eqb h1 h2 then
      count_diff t1 t2 acc
    else
      count_diff t1 t2 (Z.succ acc)
  end.

Definition smallest_change_impl (arr: list Z): Z :=
  let len := length arr in
  let half_len := (len / 2)%nat in
  let first_half := firstn half_len arr in
  (* Use skipn to get the second half of the list *)
  let second_half := skipn (len - half_len) arr in
  count_diff first_half (rev second_half) 0.

Definition problem_73_pre (arr : list Z) : Prop := True.

Definition problem_73_spec (arr: list Z) (n: Z): Prop :=
  n = smallest_change_impl arr.

(* ========================================================================= *)
(*                           Second Specification                            *)
(* ========================================================================= *)

Fixpoint count_diffs (l1 l2 : list Z) : Z :=
  match l1, l2 with
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => (if Z.eq_dec x y then 0 else 1) + count_diffs xs ys
  end.

Definition smallest_change_spec (arr : list Z) (cnt : Z) : Prop :=
  let half_len := Nat.div (length arr) 2 in
  let prefix := firstn half_len arr in
  let suffix_reversed := firstn half_len (rev arr) in
  cnt = count_diffs prefix suffix_reversed.

(* ========================================================================= *)
(*                                 Proof                                     *)
(* ========================================================================= *)

(** Lemma: Equivalence between count_diff (accumulator based) and count_diffs (structural) *)
Lemma count_diff_equivalence : forall l1 l2 acc,
  count_diff l1 l2 acc = acc + count_diffs l1 l2.
Proof.
  induction l1 as [|x xs IH]; destruct l2 as [|y ys]; simpl; intros; try lia.
  destruct (Z.eqb_spec x y); destruct (Z.eq_dec x y); try congruence.
  - (* x = y *)
    rewrite IH. 
    lia.
  - (* x <> y *)
    rewrite IH.
    unfold Z.succ.
    lia.
Qed.

(** Main Theorem: Spec 1 implies Spec 2 *)
Theorem spec1_implies_spec2 : forall arr n,
  problem_73_spec arr n -> smallest_change_spec arr n.
Proof.
  intros arr n Hspec1.
  
  (* Unfold definitions from Spec 1 *)
  unfold problem_73_spec, smallest_change_impl in Hspec1.
  
  (* Unfold definitions from Spec 2 *)
  unfold smallest_change_spec.
  
  (* Substitute n from Spec 1 into goal *)
  subst n.
  
  (* Simplify terms to match *)
  (* Note: Spec 1 uses (len / 2)%nat, Spec 2 uses Nat.div (length arr) 2. These are identical. *)
  
  (* Apply the equivalence of the counting functions *)
  rewrite count_diff_equivalence.
  rewrite Z.add_0_l.
  
  (* Now we need to prove that the arguments to the counting functions are equivalent.
     Spec 1: firstn half arr, rev (skipn (len - half) arr)
     Spec 2: firstn half arr, firstn half (rev arr)
     
     The first arguments are trivially identical.
     We need to prove: rev (skipn (length arr - half) arr) = firstn half (rev arr)
  *)
  
  f_equal.
  
  (* Use the standard library theorem firstn_rev: 
     forall (A : Type) (k : nat) (l : list A), firstn k (rev l) = rev (skipn (length l - k) l) 
  *)
  symmetry.
  apply firstn_rev.
Qed.