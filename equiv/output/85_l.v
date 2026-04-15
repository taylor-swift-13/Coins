Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Bool.Bool.
Require Import Coq.Arith.Arith.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(*                    First Specification                            *)
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
(*                    Second Specification                           *)
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
(*                    Proof of Implication                           *)
(* ================================================================= *)

(* Helper lemma: The successor of a natural number flips its oddness parity. 
   This is equivalent to Nat.odd_succ but proved locally to ensure self-containment.
*)
Lemma local_odd_succ : forall n, Nat.odd (S n) = negb (Nat.odd n).
Proof.
  induction n.
  - simpl. reflexivity.
  - simpl. rewrite IHn. 
    (* Nat.odd (S (S n)) simplifies to Nat.odd n *)
    symmetry. 
    apply Bool.negb_involutive.
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
    
    (* Apply the Inductive Hypothesis for the tail.
       We instantiate IH with (S n) to match the recursive call structure.
    *)
    rewrite (IH (S n)).
    
    (* Align the boolean parameter for the recursive call.
       We know Nat.odd (S n) = negb (Nat.odd n).
    *)
    rewrite local_odd_succ.
    
    (* Now we check if the conditional logic matches.
       LHS: if (Nat.odd n && Z.even h) then h + rest else rest
       RHS: (if (Nat.odd n && Z.even h) then h else 0) + rest
    *)
    destruct (Nat.odd n && Z.even h)%bool.
    + (* Condition is true: LHS = h + rest, RHS = h + rest *)
      reflexivity.
    + (* Condition is false: LHS = rest, RHS = 0 + rest = rest *)
      simpl. reflexivity.
Qed.

(* Final Theorem: Spec 1 implies Spec 2 *)
Theorem spec1_implies_spec2 : forall (lst : list Z) (output : Z),
  problem_85_pre lst ->
  problem_85_spec lst output ->
  add_spec lst output.
Proof.
  intros lst output Hpre Hspec.
  (* Unfold definitions to expose the underlying functions *)
  unfold problem_85_spec, add_impl in Hspec.
  unfold add_spec.
  
  (* Substitute the output from Spec 1 *)
  rewrite Hspec.
  
  (* Apply the equivalence lemma with n = 0 *)
  rewrite sum_equivalence.
  
  (* Simplify Nat.odd 0 to false *)
  simpl.
  
  reflexivity.
Qed.