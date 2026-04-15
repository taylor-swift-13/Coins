(* 
   COMPLETE, STANDALONE Coq file proving that prod_signs_spec (LLM-generated) 
   implies problem_128_spec (Human-written).
*)

(* ================================================================= *)
(* Required Imports                                                  *)
(* ================================================================= *)
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (LLM-generated)                               *)
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
(* Second Specification (Human-written)                              *)
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
(* Helper Lemma                                                      *)
(* ================================================================= *)

(* 
   A helper lemma to relate fold_right with map to fold_right with a composition.
   This proves that: fold_right op base (map f l) is equivalent to 
   fold_right (fun x acc => op (f x) acc) base l.
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

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

Theorem spec1_implies_spec2 : forall arr output,
  prod_signs_spec arr output -> problem_128_spec arr output.
Proof.
  intros arr output H.
  
  (* Unfold definitions to expose the underlying logic *)
  unfold prod_signs_spec in H.
  unfold problem_128_spec.
  
  (* Perform case analysis on the input list *)
  destruct arr as [| z l].
  
  - (* Case: Empty list *)
    (* In spec1 (H), match arr with [] => res = None. So H is output = None. *)
    (* In spec2, match arr with [] => output = None. *)
    assumption.
    
  - (* Case: Non-empty list (z :: l) *)
    (* In spec1 (H), match arr with _ => ... *)
    (* In spec2, match arr with _ :: _ => ... *)
    
    (* Rewrite the goal using the assumption H *)
    rewrite H.
    
    (* We need to prove that the result calculated in spec1 is equal to the result in spec2. *)
    (* Specifically: Some (spec1_sum * spec1_prod) = Some (spec2_sum * spec2_prod) *)
    f_equal. f_equal.
    
    (* Subgoal 1: Prove equivalence of sum calculations *)
    (* Spec1 uses: fold_right (fun x acc => Z.abs x + acc) 0 (z :: l) *)
    (* Spec2 uses: fold_right Z.add 0 (map Z.abs (z :: l)) *)
    + rewrite fold_right_map_equiv.
      (* After rewriting Spec2 using the lemma, it becomes:
         fold_right (fun x acc => Z.add (Z.abs x) acc) 0 (z :: l) *)
      (* Since Z.add a b is definitionally equal to a + b in Z_scope, 
         the functions are identical. *)
      reflexivity.
      
    (* Subgoal 2: Prove equivalence of product calculations *)
    (* Spec1 uses: fold_right (fun x acc => Z.sgn x * acc) 1 (z :: l) *)
    (* Spec2 uses: fold_right Z.mul 1 (map Z.sgn (z :: l)) *)
    + rewrite fold_right_map_equiv.
      (* After rewriting Spec2 using the lemma, it becomes:
         fold_right (fun x acc => Z.mul (Z.sgn x) acc) 1 (z :: l) *)
      (* Since Z.mul a b is definitionally equal to a * b in Z_scope, 
         the functions are identical. *)
      reflexivity.
Qed.