(* 
   This file proves that the human-written specification (problem_157_spec) 
   implies the LLM-generated specification (right_angle_triangle_spec).
*)

Require Import Coq.ZArith.ZArith.
Require Import Coq.Bool.Bool.
Open Scope Z_scope.

(* ========================================================= *)
(* First Specification (Human-written spec)                  *)
(* ========================================================= *)

(* 任意整数边长输入 *)
Definition problem_157_pre (a b c : Z) : Prop :=   (a > 0 /\ b > 0 /\ c > 0).

(*
  right_angle_triangle_spec a b c res

  - a, b, c: 代表三角形三边长度的整数 (Z)。
  - res: 程序的布尔值输出 (bool)。

  该规约规定，如果输入 a, b, c 均为正数，
  那么当且仅当这三条边满足勾股定理的任意一种排列时，
  程序的返回结果 res 为 true。
*)
Definition problem_157_spec (a b c : Z) (res : bool) : Prop :=
  res = true <-> (a * a + b * b = c * c \/
                  a * a + c * c = b * b \/
                  b * b + c * c = a * a).

(* ========================================================= *)
(* Second Specification (LLM-generated spec)                 *)
(* ========================================================= *)

Definition right_angle_triangle_spec (a b c : Z) (res : bool) : Prop :=
  res = true <-> 
  (a ^ 2 + b ^ 2 = c ^ 2 \/ a ^ 2 + c ^ 2 = b ^ 2 \/ b ^ 2 + c ^ 2 = a ^ 2).

(* ========================================================= *)
(* Proof of Implication                                      *)
(* ========================================================= *)

Theorem spec1_implies_spec2 : forall (a b c : Z) (res : bool),
  problem_157_spec a b c res -> right_angle_triangle_spec a b c res.
Proof.
  intros a b c res H.
  
  (* Unfold definitions to reveal the logical structure *)
  unfold problem_157_spec in H.
  unfold right_angle_triangle_spec.
  
  (* 
     The human spec uses multiplication (x * x).
     The LLM spec uses exponentiation (x ^ 2).
     We use the library lemma Z.pow_2_r: forall x : Z, x ^ 2 = x * x.
  *)
  
  (* Rewrite all occurrences of x^2 to x*x in the goal *)
  rewrite !Z.pow_2_r.
  
  (* Now the goal matches the hypothesis exactly *)
  exact H.
Qed.