
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.

Open Scope Z_scope.

Fixpoint correct_bracketing_aux (l : list ascii) (cnt : Z) : bool :=
  match l with
  | nil => Z.eqb cnt 0
  | c :: rest =>
    let cnt' := if Ascii.eqb c "<" then cnt + 1
                else if Ascii.eqb c ">" then cnt - 1
                else cnt in
    if Z.ltb cnt' 0 then false
    else correct_bracketing_aux rest cnt'
  end.

Definition correct_bracketing_spec (brackets : string) (res : bool) : Prop :=
  res = correct_bracketing_aux (list_ascii_of_string brackets) 0.
