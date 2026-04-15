
Require Import Coq.ZArith.ZArith.
Open Scope Z_scope.

Definition car_race_collision_spec (n : Z) (result : Z) : Prop :=
  result = n * n.
