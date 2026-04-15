(* 
   COMPLETE, STANDALONE Coq file proving that problem_128_spec implies prod_signs_spec.
   Includes all necessary definitions and a self-contained proof.
*)

(* ================================================================= *)
(* Required Imports                                                  *)
(* ================================================================= *)
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (Human-written)                               *)
(* ================================================================= *)

(* 输入可为空或任意整数列表 *)
Definition problem_128_pre (arr : list Z) : Prop := True.

(*
  程序规约 (Specification) 定义了输入 `arr` (一个整数列表)
  与输出 `output` (一个可选的整数) 之间的关系。
*)
Definition problem_128_spec (arr : list Z) (output : option Z) : Prop :=
  (* 对输入列表进行模式匹配 *)
  match arr with
  (* 情况1: 如果列表为空 *)
  | [] =>
    (* 规约要求输出必须是 None *)
    output = None

  (* 情况2: 如果列表非空 *)
  | _ :: _ =>
    (* 使用 let ... in ... 结构来定义中间计算 *)
    (* 计算列表中所有元素绝对值的和 *)
    let sum_mags := fold_right Z.add 0 (map Z.abs arr) in
    (* 计算列表中所有元素符号的乘积 *)
    let prod_sgs := fold_right Z.mul 1 (map Z.sgn arr) in
    (* 规约要求输出必须是 Some (和 * 积) *)
    output = Some (sum_mags * prod_sgs)
  end.

(* ================================================================= *)
(* Second Specification (LLM-generated)                              *)
(* ================================================================= *)

Definition prod_signs_spec (arr : list Z) (res : option Z) : Prop :=
  match arr with
  | [] => res = None
  | _ =>
      let sum_magnitudes := fold_right (fun x acc => Z.abs x + acc) 0 arr in
      let prod_signs := fold_right (fun x acc => Z.sgn x * acc) 1 arr in
      res = Some (sum_magnitudes * prod_signs)
  end.

(* ================================================================= *)
(* Helper Lemmas & Proof                                             *)
(* ================================================================= *)

(* 
   A helper lemma to relate fold_right with map to fold_right with a composition.
   This avoids dependency on library theorem names like fold_right_map which might vary.
*)
Lemma fold_right_map_equiv : forall (A B : Type) (f : A -> B) (op : B -> B -> B) (base : B) (l : list A),
  fold_right op base (map f l) = fold_right (fun x acc => op (f x) acc) base l.
Proof.
  intros A B f op base l.
  induction l as [|x xs IH].
  - (* Base case: empty list *)
    simpl. reflexivity.
  - (* Inductive step: x :: xs *)
    simpl.
    rewrite IH.
    reflexivity.
Qed.

(* Main theorem: Spec 1 implies Spec 2 *)
Theorem spec1_implies_spec2 : forall arr output,
  problem_128_spec arr output -> prod_signs_spec arr output.
Proof.
  intros arr output H.
  
  (* Unfold the definitions to see the structure *)
  unfold problem_128_spec in H.
  unfold prod_signs_spec.
  
  (* Perform case analysis on the input list 'arr' *)
  destruct arr as [| z l].
  
  - (* Case 1: arr is empty [] *)
    (* Both specs define output/res to be None. *)
    (* H : output = None *)
    assumption.
    
  - (* Case 2: arr is non-empty (z :: l) *)
    (* H : output = Some (sum_mags * prod_sgs) defined in spec1 style *)
    (* Goal: res = Some (sum_magnitudes * prod_signs) defined in spec2 style *)
    
    rewrite H. (* Substitute output with the value from H *)
    
    (* We need to prove that the values inside Some are equal *)
    f_equal. f_equal.
    
    (* Subgoal 1: Prove sums are equal *)
    (* LHS: fold_right Z.add 0 (map Z.abs (z :: l)) *)
    (* RHS: fold_right (fun x acc => Z.abs x + acc) 0 (z :: l) *)
    + rewrite fold_right_map_equiv.
      (* Now LHS becomes: fold_right (fun x acc => Z.add (Z.abs x) acc) 0 (z :: l) *)
      (* Since Z.add a b is a + b, the functions are identical. *)
      reflexivity.
      
    (* Subgoal 2: Prove products are equal *)
    (* LHS: fold_right Z.mul 1 (map Z.sgn (z :: l)) *)
    (* RHS: fold_right (fun x acc => Z.sgn x * acc) 1 (z :: l) *)
    + rewrite fold_right_map_equiv.
      (* Now LHS becomes: fold_right (fun x acc => Z.mul (Z.sgn x) acc) 1 (z :: l) *)
      (* Since Z.mul a b is a * b, the functions are identical. *)
      reflexivity.
Qed.