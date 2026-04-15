Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (Human-written)                               *)
(* ================================================================= *)

(* 任意两个整数输入均合法 *)
Definition problem_97_pre (a b : Z) : Prop := True.

(*
  multiply_spec_abs 定义了程序 multiply 的一个更精确的规约。
  它是一个命题（Prop），描述了输入 a, b 和输出 r 之间的关系。
  - a: 第一个输入整数。
  - b: 第二个输入整数。
  - r: 函数的返回值。
  规约指出，返回值 r 必须等于 a 的绝对值的个位数与 b 的绝对值的个位数的乘积。
  使用 Z.abs 可以确保对于负数（例如 -15），我们取其个位数（5）而不是负的余数。
*)
Definition problem_97_spec (a b r : Z) : Prop :=
  r = (Z.abs a mod 10) * (Z.abs b mod 10).

(* ================================================================= *)
(* Second Specification (LLM-generated)                              *)
(* ================================================================= *)

Definition multiply_spec (a b result : Z) : Prop :=
  result = (Z.abs a mod 10) * (Z.abs b mod 10).

(* ================================================================= *)
(* Implication Proof                                                 *)
(* ================================================================= *)

(* 
   Theorem: If the preconditions and postconditions of the first specification hold,
   then the postcondition of the second specification holds.
*)
Theorem spec1_implies_spec2 : forall a b r : Z,
  problem_97_pre a b ->
  problem_97_spec a b r ->
  multiply_spec a b r.
Proof.
  intros a b r Hpre Hspec.
  (* Unfold the definitions to see their underlying structure *)
  unfold problem_97_spec in Hspec.
  unfold multiply_spec.
  (* Since the definitions are identical, the hypothesis Hspec directly proves the goal *)
  assumption.
Qed.