
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope Z_scope.

Fixpoint digits_nat (n : nat) (fuel : nat) : list nat :=
  match fuel with
  | 0%nat => []
  | S fuel' =>
    match n with
    | 0%nat => []
    | _ => (n mod 10)%nat :: digits_nat (n / 10)%nat fuel'
    end
  end.

Definition z_to_digits (n : Z) : list Z :=
  let n_nat := Z.to_nat n in
  map Z.of_nat (rev (digits_nat n_nat (S n_nat))).

Definition is_palindrome (n : Z) : bool :=
  let d := z_to_digits n in
  if list_eq_dec Z.eq_dec d (rev d) then true else false.

Fixpoint seq_Z (start len : nat) : list Z :=
  match len with
  | 0%nat => []
  | S len' => Z.of_nat start :: seq_Z (S start) len'
  end.

Definition even_odd_palindrome_spec (n : Z) (res : Z * Z) : Prop :=
  let (even_cnt, odd_cnt) := res in
  let range := seq_Z 1 (Z.to_nat n) in
  let palindromes := filter is_palindrome range in
  even_cnt = Z.of_nat (length (filter Z.even palindromes)) /\
  odd_cnt = Z.of_nat (length (filter Z.odd palindromes)).
