(* 
   Coq Proof File: Spec 1 (Human) implies Spec 2 (LLM)
   
   This file contains:
   1. Definitions from the first specification (problem_67).
   2. Definitions from the second specification (fruit_distribution_spec).
   3. A theorem and proof showing that if the first specification holds (under valid physical constraints),
      then the second specification also holds.
*)

(* ========================================================================== *)
(*                           IMPORTS                                          *)
(* ========================================================================== *)

Require Import ZArith.
Require Import Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Lia.

(* ========================================================================== *)
(*                           FIRST SPECIFICATION                              *)
(*            (Human-written: Nat-based, String parsing, Saturation sub)      *)
(* ========================================================================== *)

Definition char_to_digit (c : ascii) : nat :=
  nat_of_ascii c - nat_of_ascii "0"%char.

(* 辅助函数：将字符串转换为自然数 *)
Fixpoint string_to_nat_aux (s : string) (acc : nat) : nat :=
  match s with
  | EmptyString => acc
  | String c s' => string_to_nat_aux s' (acc * 10 + char_to_digit c)
  end.

(* 主函数：将字符串转换为自然数 *)
Definition string_to_nat (s : string) : nat :=
  string_to_nat_aux s 0.

(*
  辅助规约: parse_fruit_string
  这个规约描述了从输入字符串 s 中解析出苹果和橘子数量的逻辑。
*)
Definition parse_fruit_string (s : string) (apples oranges : nat) : Prop :=
  exists s_apples s_oranges,
    apples = string_to_nat s_apples /\
    oranges = string_to_nat s_oranges /\
    s = (s_apples ++ " apples and " ++ s_oranges ++ " oranges")%string.
       

Definition problem_67_pre (s : string) (n : nat) : Prop := True.

Definition problem_67_spec (s : string) (n : nat) (ret : nat) : Prop :=
  exists apples oranges,
    parse_fruit_string s apples oranges /\
    ret = n - (apples + oranges).

(* ========================================================================== *)
(*                           SECOND SPECIFICATION                             *)
(*            (LLM-generated: Z-based, explicit non-negativity)               *)
(* ========================================================================== *)

Open Scope Z_scope.

Definition fruit_distribution_spec (apples : Z) (oranges : Z) (n : Z) (result : Z) : Prop :=
  apples >= 0 /\
  oranges >= 0 /\
  n >= 0 /\
  n - apples - oranges >= 0 /\
  result = n - apples - oranges.

(* ========================================================================== *)
(*                           IMPLICATION PROOF                                *)
(* ========================================================================== *)

(* 
   Theorem: spec1_implies_spec2
   
   We prove that for any result 'ret' satisfying the first specification (problem_67_spec),
   there exist apple and orange counts (found by parsing) such that:
   IF the scenario is physically valid (total fruits <= n),
   THEN the second specification (fruit_distribution_spec) holds for the converted Z values.
   
   Note: This conditional implication is necessary because Spec 1 uses saturating subtraction on nats
   (returns 0 if apples+oranges > n), whereas Spec 2 uses Z subtraction and explicitly requires 
   the result to be non-negative.
*)

Theorem spec1_implies_spec2 : 
  forall (s : string) (n : nat) (ret : nat),
  problem_67_spec s n ret ->
  exists (apples oranges : nat),
    parse_fruit_string s apples oranges /\
    ((apples + oranges <= n)%nat -> 
     fruit_distribution_spec (Z.of_nat apples) (Z.of_nat oranges) (Z.of_nat n) (Z.of_nat ret)).
Proof.
  intros s n ret Hspec.
  
  (* 1. Deconstruct the hypothesis from Spec 1 to get the witnesses *)
  unfold problem_67_spec in Hspec.
  destruct Hspec as [apples [oranges [Hparse Hret]]].
  
  (* 2. Provide these witnesses to satisfy the existence claim *)
  exists apples, oranges.
  split.
  - (* The parsing logic is identical, so this is trivial *)
    exact Hparse.
  - (* Now prove the implication: Valid Input -> Spec 2 Holds *)
    intros Hvalid.
    unfold fruit_distribution_spec.
    
    (* We use the 'lia' tactic (Linear Integer Arithmetic) which is powerful enough
       to handle Z and nat arithmetic, inequalities, and conversions automatically. *)
    split; [|split; [|split; [|split]]].
    
    + (* Goal: Z.of_nat apples >= 0 *)
      lia.
      
    + (* Goal: Z.of_nat oranges >= 0 *)
      lia.
      
    + (* Goal: Z.of_nat n >= 0 *)
      lia.
      
    + (* Goal: Z.of_nat n - Z.of_nat apples - Z.of_nat oranges >= 0 *)
      (* This follows from Hvalid: apples + oranges <= n *)
      lia.
      
    + (* Goal: Z.of_nat ret = Z.of_nat n - Z.of_nat apples - Z.of_nat oranges *)
      (* Substitute ret from Spec 1: ret = n - (apples + oranges) *)
      rewrite Hret.
      
      (* We need to show: Z.of_nat (n - (apples + oranges)) = ... *)
      (* Since apples + oranges <= n (Hvalid), natural subtraction maps directly to Z subtraction *)
      rewrite Nat2Z.inj_sub.
      * (* Goal: Z.of_nat n - Z.of_nat (apples + oranges) = ... *)
        rewrite Nat2Z.inj_add.
        lia. (* Z arithmetic equality *)
      * (* Proof of condition for inj_sub: apples + oranges <= n *)
        exact Hvalid.
Qed.