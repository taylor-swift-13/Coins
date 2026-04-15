
Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope Z_scope.

Fixpoint sum_list (l : list Z) : Z :=
  match l with
  | [] => 0
  | x :: xs => x + sum_list xs
  end.

Definition minSubArraySum_spec (nums : list Z) (ans : Z) : Prop :=
  (exists sub pre post, nums = pre ++ sub ++ post /\ sub <> [] /\ sum_list sub = ans) /\
  (forall sub pre post, nums = pre ++ sub ++ post -> sub <> [] -> ans <= sum_list sub).
