
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.

Local Open Scope Z_scope.

Fixpoint value_base_acc (base : Z) (acc : Z) (ds : list Z) : Z :=
  match ds with
  | nil => acc
  | d :: rest => value_base_acc base (base * acc + d) rest
  end.

Definition value_base_Z (base : Z) (ds : list Z) : Z :=
  value_base_acc base 0 ds.

Fixpoint sum_list_Z (ds : list Z) : Z :=
  match ds with
  | nil => 0
  | d :: rest => d + sum_list_Z rest
  end.

Definition decimal_digits_of (N : Z) (ds : list Z) : Prop :=
  (forall d, In d ds -> 0 <= d <= 9) /\
  value_base_Z 10 ds = N /\
  ((N = 0 /\ ds = (0 :: nil)) \/
   (N <> 0 /\ exists d rest, ds = d :: rest /\ d <> 0)).

Definition bit_to_Z (b : bool) : Z := if b then 1 else 0.

Fixpoint string_of_bits (bs : list bool) : string :=
  match bs with
  | nil => EmptyString
  | b :: rest => String (if b then "1"%char else "0"%char) (string_of_bits rest)
  end.

Definition value_bits (bits : list bool) : Z :=
  value_base_acc 2 0 (map bit_to_Z bits).

Definition binary_string_of (s : Z) (out : string) : Prop :=
  exists bits,
    value_bits bits = s /\
    ((s = 0 /\ bits = (false :: nil)) \/
     (s <> 0 /\ exists rest, bits = true :: rest)) /\
    out = string_of_bits bits.

Definition solve_spec (N : Z) (out : string) : Prop :=
  0 <= N <= 10000 /\
  exists ds s,
    decimal_digits_of N ds /\
    s = sum_list_Z ds /\
    binary_string_of s out.
