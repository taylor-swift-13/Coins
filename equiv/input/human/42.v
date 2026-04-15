(* Return list with elements incremented by 1.
>>> incr_list([1, 2, 3])
[2, 3, 4]
>>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
[6, 4, 6, 3, 4, 4, 10, 1, 124] *)

(* Spec(input : list int, output : list int) :=
​	length(input) = length(output) /\
​	∀i ∈ [0, length(input)), outpu[i] = input[i] + 1
*)

Require Import Coq.Lists.List.
Require Import Coq.ZArith.ZArith.
Import ListNotations.

(* Pre: no special constraints for `incr_list` *)
Definition problem_42_pre (input : list Z) : Prop := True.

Definition problem_42_spec(input output : list Z) : Prop :=
  length input = length output /\
  forall i : nat, i < length output -> nth i output 0%Z = ((nth i input 0%Z) + 1)%Z.