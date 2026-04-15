
Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

Fixpoint sum_odd_at_even_pos (lst : list Z) : Z :=
  match lst with
  | [] => 0
  | [x] => if Z.odd x then x else 0
  | x :: _ :: xs => (if Z.odd x then x else 0) + sum_odd_at_even_pos xs
  end.

Definition solution_spec (lst : list Z) (res : Z) : Prop :=
  res = sum_odd_at_even_pos lst.
