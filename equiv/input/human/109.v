(* We have an array 'arr' of N integers arr[1], arr[2], ..., arr[N].The
numbers in the array will be randomly ordered. Your task is to determine if
it is possible to get an array sorted in non-decreasing order by performing
the following operation on the given array:
You are allowed to perform right shift operation any number of times.

One right shift operation means shifting all elements of the array by one
position in the right direction. The last element of the array will be moved to
the starting position in the array i.e. 0th index.

If it is possible to obtain the sorted array by performing the above operation
then return True else return False.
If the given array is empty then return True.

Note: The given list is guaranteed to have unique elements.

For Example:

move_one_ball([3, 4, 5, 1, 2])==>True
Explanation: By performin 2 right shift operations, non-decreasing order can
be achieved for the given array.
move_one_ball([3, 5, 4, 1, 2])==>False
Explanation:It is not possible to get non-decreasing order for the given
array by performing any number of right shift operations. *)

(* 导入列表、整数和自然数所需的基础库 *)
Require Import List ZArith Arith.
Require Import Coq.Sorting.Sorted.
Open Scope Z_scope.
Import ListNotations.



Fixpoint is_sorted_bool (l : list Z) : bool :=
  match l with
  | [] => true
  | [_] => true
  | h1 :: t =>
      match t with
      | [] => true
      | h2 :: t' => if Z.leb h1 h2 then is_sorted_bool t else false
      end
  end.

Definition right_shift (l : list Z) : list Z :=
  match rev l with
  | [] => []
  | hd :: tl => hd :: rev tl
  end.


Fixpoint n_shifts (n : nat) (l : list Z) : list Z :=
  match n with
  | 0%nat => l
  | S n' => right_shift (n_shifts n' l)
  end.


Fixpoint check_all_shifts (arr : list Z) (n : nat) : bool :=
  match n with
  | O => is_sorted_bool arr
  | S n' => orb (is_sorted_bool (n_shifts n arr)) (check_all_shifts arr n')
  end.

Definition move_one_ball_impl (arr : list Z) : bool :=
  match arr with
  | [] => true
  | _ :: _ => check_all_shifts arr (length arr)
  end.

(* 输入数组元素互不相同（唯一性），可空列表允许直接返回 True *)
Definition problem_109_pre (arr : list Z) : Prop := NoDup arr.

Definition problem_109_spec (arr : list Z) (result : bool) : Prop :=
  result = move_one_ball_impl arr.