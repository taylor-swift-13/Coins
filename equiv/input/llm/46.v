
Require Import ZArith.
Open Scope Z_scope.

Definition fib4_spec (n : Z) (result : Z) : Prop :=
  (n = 0 /\ result = 0) \/
  (n = 1 /\ result = 0) \/
  (n = 2 /\ result = 2) \/
  (n = 3 /\ result = 0) \/
  (n >= 4 /\
   exists a b c d i,
     a = 0 /\ b = 0 /\ c = 2 /\ d = 0 /\
     i = 4 /\
     (forall k, 4 <= k <= n ->
       exists a' b' c' d',
         a' = b /\
         b' = c /\
         c' = d /\
         d' = a + b + c + d /\
         (a, b, c, d) = (a', b', c', d')) /\
     d = result).
