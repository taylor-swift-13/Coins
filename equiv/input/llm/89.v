
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.

Open Scope string_scope.
Open Scope nat_scope.

Definition is_lowercase (c : ascii) : bool :=
  let n := nat_of_ascii c in
  (97 <=? n) && (n <=? 122).

Definition rotate_char (c : ascii) : ascii :=
  if is_lowercase c then
    let n := nat_of_ascii c in
    ascii_of_nat (97 + (n - 97 + 4) mod 26)
  else 
    c.

Fixpoint encrypt_model (s : string) : string :=
  match s with
  | EmptyString => EmptyString
  | String c s' => String (rotate_char c) (encrypt_model s')
  end.

Definition encrypt_spec (s : string) (result : string) : Prop :=
  result = encrypt_model s.
