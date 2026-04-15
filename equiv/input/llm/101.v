
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.
Import ListNotations.

Local Open Scope string_scope.

Definition is_separator (c : ascii) : bool :=
  orb (Ascii.eqb c " "%char) (Ascii.eqb c ","%char).

Fixpoint words_string_aux (s : string) (acc : string) : list string :=
  match s with
  | EmptyString => 
      if String.eqb acc "" then [] else [acc]
  | String c s' =>
      if is_separator c then
        if String.eqb acc "" then words_string_aux s' ""
        else acc :: words_string_aux s' ""
      else
        words_string_aux s' (acc ++ String c EmptyString)
  end.

Definition words_string_spec (s : string) (result : list string) : Prop :=
  result = words_string_aux s "".
