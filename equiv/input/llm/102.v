
Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

Definition choose_num_spec (x y result : Z) : Prop :=
  (result = -1 <-> (forall n : Z, x <= n <= y -> Z.Odd n)) /\
  (result <> -1 -> 
     Z.Even result /\ 
     x <= result <= y /\ 
     (forall n : Z, x <= n <= y -> Z.Even n -> n <= result)).
