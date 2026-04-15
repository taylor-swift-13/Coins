
Require Import Coq.Strings.String.
Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.
Open Scope string_scope.
Open Scope Z_scope.

Definition m_table : list string := ["" ; "m"].
Definition c_table : list string := ["" ; "c" ; "cc" ; "ccc" ; "cd" ; "d" ; "dc" ; "dcc" ; "dccc" ; "cm"].
Definition x_table : list string := ["" ; "x" ; "xx" ; "xxx" ; "xl" ; "l" ; "lx" ; "lxx" ; "lxxx" ; "xc"].
Definition i_table : list string := ["" ; "i" ; "ii" ; "iii" ; "iv" ; "v" ; "vi" ; "vii" ; "viii" ; "ix"].

Definition int_to_mini_roman_spec (number : Z) (result : string) : Prop :=
  1 <= number <= 1000 ->
  let thousands_idx := Z.to_nat (number / 1000) in
  let hundreds_idx := Z.to_nat ((number mod 1000) / 100) in
  let tens_idx := Z.to_nat ((number mod 100) / 10) in
  let ones_idx := Z.to_nat (number mod 10) in
  let thousands := nth thousands_idx m_table "" in
  let hundreds := nth hundreds_idx c_table "" in
  let tens := nth tens_idx x_table "" in
  let ones := nth ones_idx i_table "" in
  result = thousands ++ hundreds ++ tens ++ ones.
