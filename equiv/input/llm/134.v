
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Init.Nat.
Require Import Coq.Bool.Bool.

Definition is_alpha (c : ascii) : bool :=
  let n := nat_of_ascii c in
  ((65 <=? n) && (n <=? 90)) || ((97 <=? n) && (n <=? 122)).

Definition check_if_last_char_is_a_letter_spec (txt : string) (res : bool) : Prop :=
  let len := String.length txt in
  if len =? 0 then
    res = false
  else if len =? 1 then
    match String.get 0 txt with
    | Some c => res = is_alpha c
    | None => False
    end
  else
    match String.get (len - 1) txt, String.get (len - 2) txt with
    | Some last_char, Some prev_char =>
        res = (is_alpha last_char && Ascii.eqb prev_char " "%char)
    | _, _ => False
    end.
