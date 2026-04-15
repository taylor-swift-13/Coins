(* Given a positive integer n, return a tuple that has the number of even and odd
integer palindromes that fall within the range(1, n), inclusive.

Example 1:

Input: 3
Output: (1, 2)
Explanation:
Integer palindrome are 1, 2, 3. one of them is even, and two of them are odd.

Example 2:

Input: 12
Output: (4, 6)
Explanation:
Integer palindrome are 1, 2, 3, 4, 5, 6, 7, 8, 9, 11. four of them are even, and 6 of them are odd.
 *)

Require Import Coq.Arith.Arith Coq.Lists.List Coq.Bool.Bool.
Import ListNotations.



Fixpoint to_digits_helper (n k : nat) {struct k} : list nat :=
  match k with
  | 0 => []
  | S k' =>
    match n with
    | 0 => []
    | _ => (n mod 10) :: to_digits_helper (n / 10) k'
    end
  end.
Definition to_digits_rev (n : nat) : list nat :=
  to_digits_helper n n.

Fixpoint list_nat_eqb (l1 l2 : list nat) : bool :=
  match l1, l2 with
  | [], [] => true
  | x :: t1, y :: t2 => andb (x =? y) (list_nat_eqb t1 t2)
  | _, _ => false
  end.

Definition is_palindrome_b (n : nat) : bool :=
  let d := to_digits_rev n in
  if (n =? 0) then false else list_nat_eqb d (rev d).

Definition is_even (n : nat) : bool :=
  (n mod 2 =? 0).

Definition is_even_pal (n : nat) : bool :=
  andb (is_palindrome_b n) (is_even n).

Definition is_odd_pal (n : nat) : bool :=
  andb (is_palindrome_b n) (negb (is_even n)).

Fixpoint count_in_range (P : nat -> bool) (k : nat) : nat :=
  match k with
  | 0 => 0
  | S k' =>
    (if P (S k') then 1 else 0) +
    count_in_range P k'
  end.

Definition count_palindromes_impl (n : nat) : nat * nat :=
  (count_in_range is_even_pal n, count_in_range is_odd_pal n).

(* n 为正整数 *)
Definition problem_107_pre (n : nat) : Prop := n > 0.

Definition problem_107_spec (n : nat) (output : nat * nat) : Prop :=
  output = count_palindromes_impl n.

