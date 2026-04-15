
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.
Require Import Coq.Bool.Bool.
Import ListNotations.

Fixpoint string_to_list (s : string) : list ascii :=
  match s with
  | EmptyString => []
  | String c s' => c :: string_to_list s'
  end.

Fixpoint list_to_string (l : list ascii) : string :=
  match l with
  | [] => EmptyString
  | c :: l' => String c (list_to_string l')
  end.

Definition is_upper (c : ascii) : bool :=
  let n := nat_of_ascii c in
  (65 <=? n) && (n <=? 90).

Definition is_lower (c : ascii) : bool :=
  let n := nat_of_ascii c in
  (97 <=? n) && (n <=? 122).

Definition switch_case_char (c : ascii) : ascii :=
  let n := nat_of_ascii c in
  if is_upper c then ascii_of_nat (n + 32)
  else if is_lower c then ascii_of_nat (n - 32)
  else c.

Definition is_vowel (c : ascii) : bool :=
  let n := nat_of_ascii c in
  match n with
  | 65 | 69 | 73 | 79 | 85    (* A, E, I, O, U *)
  | 97 | 101 | 105 | 111 | 117 (* a, e, i, o, u *)
    => true
  | _ => false
  end.

Definition vowel_change_char (c : ascii) : ascii :=
  if is_vowel c then ascii_of_nat (nat_of_ascii c + 2)
  else c.

Definition encode_process (l : list ascii) : list ascii :=
  map vowel_change_char (map switch_case_char l).

Definition encode_spec (message : string) (result : string) : Prop :=
  result = list_to_string (encode_process (string_to_list message)).
