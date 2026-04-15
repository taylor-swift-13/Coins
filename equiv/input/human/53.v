(* def add(x: int, y: int):
"""Add two numbers x and y
>>> add(2, 3)
5
>>> add(5, 7)
12
""" *)
Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

(* Pre: no special constraints for `add` *)
Definition problem_53_pre (x y : Z) : Prop := True.

Definition problem_53_spec (x : Z) (y : Z) (output : Z) : Prop :=
  output = x + y.