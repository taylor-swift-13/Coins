
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Open Scope string_scope.

Definition concatenate_spec (strings : list string) (result : string) : Prop :=
  result = fold_right append "" strings.
