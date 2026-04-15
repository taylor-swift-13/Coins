(* 
   This file proves that the LLM-generated specification (right_angle_triangle_spec) 
   implies the human-written specification (problem_157_spec).
*)

Require Import Coq.ZArith.ZArith.
Require Import Coq.Bool.Bool.
Open Scope Z_scope.

(* ========================================================= *)
(* First Specification (LLM-generated spec)                  *)
(* ========================================================= *)

Definition right_angle_triangle_spec (a b c : Z) (res : bool) : Prop :=
  res = true <-> 
  (a ^ 2 + b ^ 2 = c ^ 2 \/ a ^ 2 + c ^ 2 = b ^ 2 \/ b ^ 2 + c ^ 2 = a ^ 2).

(* ========================================================= *)
(* Second Specification (Human-written spec)                 *)
(* ========================================================= *)

(* def right_angle_triangle(a, b, c):
'''
Given the lengths of the three sides of a triangle. Return True if the three
sides form a right-angled triangle, False otherwise.
A right-angled triangle is a triangle in which one angle is right angle or
90 degree.
Example:
right_angle_triangle(3, 4, 5) == True
right_angle_triangle(1, 2, 3) == False
''' *)

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
(* Proof of Implication                                      *)
(* ========================================================= *)

Theorem spec1_implies_spec2 : forall (a b c : Z) (res : bool),
  right_angle_triangle_spec a b c res -> problem_157_spec a b c res.
Proof.
  intros a b c res H.
  
  (* Unfold definitions to see the underlying propositions *)
  unfold right_angle_triangle_spec in H.
  unfold problem_157_spec.
  
  (* 
     The LLM spec uses exponentiation (x ^ 2), while the human spec uses 
     multiplication (x * x). We need to show they are equivalent.
     Z.pow_2_r is a standard library lemma: forall x, x ^ 2 = x * x.
  *)
  
  (* Rewrite all occurrences of x^2 to x*x in the hypothesis H *)
  rewrite !Z.pow_2_r in H.
  
  (* Now the hypothesis H exactly matches the goal *)
  exact H.
Qed.