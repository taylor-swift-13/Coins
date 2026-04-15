
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.
Import ListNotations.

Definition is_upper_vowel (c : ascii) : bool :=
  match c with
  | "A"%char => true
  | "E"%char => true
  | "I"%char => true
  | "O"%char => true
  | "U"%char => true
  | _ => false
  end.

Fixpoint string_to_list (s : string) : list ascii :=
  match s with
  | EmptyString => []
  | String c rest => c :: string_to_list rest
  end.

Fixpoint count_upper_vowels_even_indices_aux (l : list ascii) (idx : nat) : nat :=
  match l with
  | [] => 0
  | c :: rest =>
    let count_rest := count_upper_vowels_even_indices_aux rest (S idx) in
    if Nat.even idx then
      if is_upper_vowel c then S count_rest else count_rest
    else
      count_rest
  end.

Definition count_upper_vowels_even_indices (l : list ascii) : nat :=
  count_upper_vowels_even_indices_aux l 0.

Definition count_upper_spec (s : string) (result : nat) : Prop :=
  result = count_upper_vowels_even_indices (string_to_list s).
