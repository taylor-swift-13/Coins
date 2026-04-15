
Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Arith.Arith.

Open Scope nat_scope.
Open Scope string_scope.

Definition total_chars (l : list string) : nat :=
  fold_right (fun s acc => String.length s + acc) 0 l.

Definition total_match_spec (lst1 lst2 res : list string) : Prop :=
  (total_chars lst1 <= total_chars lst2 /\ res = lst1) \/
  (total_chars lst2 < total_chars lst1 /\ res = lst2).
