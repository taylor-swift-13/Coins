Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Bool.Bool.
Require Import Coq.Arith.Arith.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(*                    First Specification (LLM-generated)            *)
(* ================================================================= *)

Fixpoint add_sum (lst : list Z) (is_odd_idx : bool) : Z :=
  match lst with
  | [] => 0
  | x :: xs => 
      (if is_odd_idx && Z.even x then x else 0) + add_sum xs (negb is_odd_idx)
  end.

Definition add_spec (lst : list Z) (res : Z) : Prop :=
  res = add_sum lst false.

(* ================================================================= *)
(*                    Second Specification (Human-written)           *)
(* ================================================================= *)

(* def add(lst):
"""Given a non-empty list of integers lst. add the even elements that are at odd indices..


Examples:
add([4, 2, 6, 7]) ==> 2
""" *)

Fixpoint sum_even_at_odd_indices (l : list Z) (n : nat) : Z :=
  match l with
  | nil => 0%Z
  | h :: t =>
      if andb (Nat.odd n) (Z.even h)
      then (h + sum_even_at_odd_indices t (S n))%Z
      else sum_even_at_odd_indices t (S n)
  end.

Definition add_impl (lst : list Z) : Z := sum_even_at_odd_indices lst 0.

Definition problem_85_pre (lst : list Z) : Prop := lst <> []%list.

Definition problem_85_spec (lst : list Z) (output : Z) : Prop :=
  output = add_impl lst.

(* ================================================================= *)
(*                    Proof of Implication                           *)
(* ================================================================= *)

(* Helper lemma: The successor of a natural number flips its oddness parity. *)
Lemma odd_succ_negb : forall n, Nat.odd (S n) = negb (Nat.odd n).
Proof.
  induction n.
  - simpl. reflexivity.
  - simpl. rewrite IHn. symmetry. apply Bool.negb_involutive.
Qed.

(* Main Equivalence Lemma:
   Proves that sum_even_at_odd_indices corresponds exactly to add_sum
   when the boolean flag matches the oddness of the index n.
*)
Lemma sum_equivalence : forall (l : list Z) (n : nat),
  sum_even_at_odd_indices l n = add_sum l (Nat.odd n).
Proof.
  induction l as [|h t IH].
  - (* Base case: empty list *)
    intros n. simpl. reflexivity.
  - (* Inductive step: h :: t *)
    intros n.
    simpl.
    (* Apply the Inductive Hypothesis for the tail with (S n) *)
    rewrite (IH (S n)).
    (* Use the helper lemma to match the boolean argument *)
    rewrite odd_succ_negb.
    
    (* Analyze the conditional logic *)
    destruct (Nat.odd n) eqn:Hodd; destruct (Z.even h) eqn:Heven; simpl.
    + (* odd=true, even=true -> add h *)
      reflexivity.
    + (* odd=true, even=false -> add 0 *)
      reflexivity.
    + (* odd=false, even=true -> add 0 *)
      reflexivity.
    + (* odd=false, even=false -> add 0 *)
      reflexivity.
Qed.

(* Final Theorem: Spec 1 implies Spec 2 *)
Theorem spec1_implies_spec2 : forall (lst : list Z) (output : Z),
  add_spec lst output ->
  problem_85_pre lst ->
  problem_85_spec lst output.
Proof.
  intros lst output Hspec Hpre.
  (* Unfold definitions to expose the underlying functions *)
  unfold add_spec in Hspec.
  unfold problem_85_spec, add_impl.
  
  (* Substitute the output from Spec 1 *)
  rewrite Hspec.
  
  (* Apply the equivalence lemma with n = 0 *)
  rewrite sum_equivalence.
  
  (* Simplify Nat.odd 0 to false *)
  simpl.
  
  reflexivity.
Qed.