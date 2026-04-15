
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Import ListNotations.
Open Scope string_scope.
Open Scope char_scope.

Definition is_vowel (c : ascii) : bool :=
  existsb (fun v => Ascii.eqb c v) ["a"; "e"; "i"; "o"; "u"; "A"; "E"; "I"; "O"; "U"].

Definition remove_vowels_spec (text : string) (result : string) : Prop :=
  result = string_of_list_ascii (filter (fun c => negb (is_vowel c)) (list_ascii_of_string text)).
