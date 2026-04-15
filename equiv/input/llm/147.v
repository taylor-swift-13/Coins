
Require Import ZArith.
Require Import List.
Require Import Lia.

Open Scope Z_scope.

Definition a_value (i : Z) : Z := i * i - i + 1.

Definition is_valid_triple (n i j k : Z) : Prop :=
  1 <= i /\ i < j /\ j < k /\ k <= n /\
  (a_value i + a_value j + a_value k) mod 3 = 0.

Definition count_valid_triples (n : Z) : Z :=
  let one_cnt := 1 + (n - 2) / 3 * 2 + (n - 2) mod 3 in
  let zero_cnt := n - one_cnt in
  one_cnt * (one_cnt - 1) * (one_cnt - 2) / 6 + 
  zero_cnt * (zero_cnt - 1) * (zero_cnt - 2) / 6.

Definition get_max_triples_spec (n : Z) (result : Z) : Prop :=
  (n <= 2 -> result = 0) /\
  (n > 2 -> result = count_valid_triples n).
