Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

(* ================================================================= *)
(* First Specification (LLM-generated)                               *)
(* ================================================================= *)

Definition multiply_spec (a b result : Z) : Prop :=
  result = (Z.abs a mod 10) * (Z.abs b mod 10).

(* ================================================================= *)
(* Second Specification (Human-written)                              *)
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
(* Implication Proof                                                 *)
(* ================================================================= *)

(* 
   Theorem: If the precondition of the second specification holds (which is trivial here),
   and the postcondition of the first specification holds,
   then the postcondition of the second specification must hold.
*)
Theorem spec1_implies_spec2 : forall a b r : Z,
  problem_97_pre a b ->
  multiply_spec a b r ->
  problem_97_spec a b r.
Proof.
  intros a b r Hpre Hspec.
  (* Unfold definitions to expose the underlying logic *)
  unfold multiply_spec in Hspec.
  unfold problem_97_spec.
  (* The definitions are identical, so the hypothesis directly implies the goal *)
  exact Hspec.
Qed.