(* 
   Coq Proof File: Spec 1 (LLM) implies Spec 2 (Human)
   
   This file contains:
   1. Definitions from the first specification (fruit_distribution_spec).
   2. Definitions from the second specification (problem_67).
   3. A theorem and proof showing that if the logic in the first specification holds 
      (for values parsed from the string), then the second specification holds.
*)

(* ========================================================================== *)
(*                           IMPORTS                                          *)
(* ========================================================================== *)

Require Import ZArith.
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Lia.

(* ========================================================================== *)
(*                           FIRST SPECIFICATION                              *)
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
(*                           SECOND SPECIFICATION                             *)
(*            (Human-written: Nat-based, String parsing, Saturation sub)      *)
(* ========================================================================== *)

(* Switch back to nat scope for the human specification which uses nat mostly *)
Open Scope nat_scope.

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
(*                           IMPLICATION PROOF                                *)
(* ========================================================================== *)

(* 
   Theorem: spec1_implies_spec2
   
   We prove that for any input string 's', total count 'n', and result 'ret',
   if there exist apple and orange counts parsed from 's' such that the 
   abstract logic (Spec 1) holds for them (when converted to Z), 
   then the concrete specification (Spec 2) also holds.
*)

Theorem spec1_implies_spec2 : 
  forall (s : string) (n : nat) (ret : nat),
  (exists apples oranges : nat, 
     parse_fruit_string s apples oranges /\ 
     fruit_distribution_spec (Z.of_nat apples) (Z.of_nat oranges) (Z.of_nat n) (Z.of_nat ret)) ->
  problem_67_spec s n ret.
Proof.
  intros s n ret H.
  
  (* 1. Deconstruct the hypothesis *)
  destruct H as [apples [oranges [Hparse Hspec1]]].
  
  (* 2. Unfold the destination spec *)
  unfold problem_67_spec.
  
  (* 3. Provide the witnesses found in the hypothesis *)
  exists apples, oranges.
  split.
  - (* The parsing is identical *)
    exact Hparse.
  - (* We need to prove: ret = n - (apples + oranges) *)
    (* Based on Hspec1: result = n - apples - oranges (in Z) *)
    unfold fruit_distribution_spec in Hspec1.
    destruct Hspec1 as [_ [_ [_ [Hvalid Hres]]]].
    
    (* Hres: Z.of_nat ret = Z.of_nat n - Z.of_nat apples - Z.of_nat oranges *)
    (* Hvalid: Z.of_nat n - Z.of_nat apples - Z.of_nat oranges >= 0 *)
    
    (* Since we are working with nats, we can inject into Z to use the hypothesis *)
    apply Nat2Z.inj.
    rewrite Hres.
    
    (* Goal: Z.of_nat n - Z.of_nat apples - Z.of_nat oranges = Z.of_nat (n - (apples + oranges)) *)
    
    (* We want to use Nat2Z.inj_sub: Z.of_nat (a - b) = Z.of_nat a - Z.of_nat b IF b <= a *)
    rewrite Nat2Z.inj_sub.
    + (* Subgoal 1: Equality of Z expressions *)
      (* Z.of_nat n - Z.of_nat apples - Z.of_nat oranges = Z.of_nat n - Z.of_nat (apples + oranges) *)
      rewrite Nat2Z.inj_add.
      lia. (* Linear Integer Arithmetic handles the algebra *)
      
    + (* Subgoal 2: Condition for subtraction (apples + oranges <= n)%nat *)
      (* This follows from Hvalid: Z.of_nat n - Z.of_nat apples - Z.of_nat oranges >= 0 *)
      (* lia can infer nat inequalities from Z facts if Z.of_nat is involved *)
      lia.
Qed.