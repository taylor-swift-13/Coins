(**
 * This file proves that the first specification (problem_74_spec) 
 * implies the second specification (total_match_spec).
 *)

(* ========================================================================= *)
(*                           Required Imports                                *)
(* ========================================================================= *)

Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.

Import ListNotations.
Open Scope string_scope.
Open Scope nat_scope.

(* ========================================================================= *)
(*                           First Specification                             *)
(* ========================================================================= *)

(**
 * @spec problem_74_pre
 * @brief 程序前置条件：接受两个字符串列表作为输入。
 *)
Definition problem_74_pre (lst1 lst2 : list string) : Prop := True.

(**
 * @spec problem_74_spec
 * @brief 程序规约：选择两个字符串列表中总字符数较少的那个作为输出（若相等则选择第一个）。
 *)
Definition problem_74_spec (lst1 lst2 output : list string) : Prop :=
  let sum := fun l => fold_left (fun acc s => acc + String.length s) l 0 in
  (sum lst1 <= sum lst2 /\ output = lst1) \/
  (sum lst1 > sum lst2 /\ output = lst2).

(* ========================================================================= *)
(*                           Second Specification                            *)
(* ========================================================================= *)

Definition total_chars (l : list string) : nat :=
  fold_right (fun s acc => String.length s + acc) 0 l.

Definition total_match_spec (lst1 lst2 res : list string) : Prop :=
  (total_chars lst1 <= total_chars lst2 /\ res = lst1) \/
  (total_chars lst2 < total_chars lst1 /\ res = lst2).

(* ========================================================================= *)
(*                           Proof of Implication                            *)
(* ========================================================================= *)

(**
 * Lemma: Generalized equivalence between the fold_left summation used in Spec 1
 * and the fold_right summation (total_chars) used in Spec 2.
 * 
 * fold_left (fun acc s => acc + len s) l n = n + total_chars l
 *)
Lemma fold_left_length_equiv : forall (l : list string) (n : nat),
  fold_left (fun acc s => acc + String.length s) l n = n + total_chars l.
Proof.
  induction l as [| s l' IH].
  - (* Base case: empty list *)
    simpl. intros n. rewrite Nat.add_0_r. reflexivity.
  - (* Inductive step *)
    simpl. intros n.
    rewrite IH.
    unfold total_chars. simpl.
    (* We need to show: (n + len s) + fold_right ... = n + (len s + fold_right ...) *)
    (* This relies on associativity of addition *)
    rewrite Nat.add_assoc.
    reflexivity.
Qed.

(**
 * Lemma: Specific equivalence for starting accumulator 0.
 * This connects the local 'sum' definition in Spec 1 to 'total_chars' in Spec 2.
 *)
Lemma sum_spec1_eq_total_chars : forall (l : list string),
  fold_left (fun acc s => acc + String.length s) l 0 = total_chars l.
Proof.
  intros l.
  rewrite fold_left_length_equiv.
  simpl. reflexivity.
Qed.

(**
 * Theorem: Spec 1 implies Spec 2.
 * If the input satisfies problem_74_spec, it satisfies total_match_spec.
 *)
Theorem spec1_implies_spec2 : forall (lst1 lst2 output : list string),
  problem_74_pre lst1 lst2 ->
  problem_74_spec lst1 lst2 output ->
  total_match_spec lst1 lst2 output.
Proof.
  intros lst1 lst2 output Hpre Hspec.
  
  (* Unfold the definitions to see the underlying logic *)
  unfold problem_74_spec in Hspec.
  unfold total_match_spec.
  
  (* Replace the fold_left summation in Hspec with total_chars using our lemma *)
  rewrite sum_spec1_eq_total_chars in Hspec.
  rewrite sum_spec1_eq_total_chars in Hspec.
  
  (* Now Hspec matches the structure of total_match_spec, modulo logic definitions *)
  destruct Hspec as [[Hle Heq] | [Hgt Heq]].
  
  - (* Case 1: total_chars lst1 <= total_chars lst2 *)
    left.
    split; assumption.
    
  - (* Case 2: total_chars lst1 > total_chars lst2 *)
    right.
    split.
    + (* Prove that (a > b) implies (b < a) *)
      (* In Coq's Arith, 'a > b' is defined as 'b < a'. We can just unfold 'gt'. *)
      unfold gt in Hgt.
      assumption.
    + assumption.
Qed.