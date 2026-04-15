(* Given a string s and a natural number n, you have been tasked to implement
a function that returns a list of all words from string s that contain exactly
n consonants, in order these words appear in the string s.
If the string s is empty then the function should return an empty list.
Note: you may assume the input string contains only letters and spaces.
Examples:
select_words("Mary had a little lamb", 4) ==> ["little"]
select_words("Mary had a little lamb", 3) ==> ["Mary", "lamb"]
select_words("simple white space", 2) ==> []
select_words("Hello world", 4) ==> ["world"]
select_words("Uncle sam", 3) ==> ["Uncle"] *)

Require Import Coq.Strings.Ascii Coq.Strings.String Coq.Lists.List Coq.Arith.Arith Coq.Bool.Bool.
Import ListNotations.



Definition is_vowel (c : ascii) : bool :=
  match c with
  | "a"%char => true | "e"%char => true | "i"%char => true | "o"%char => true | "u"%char => true
  | "A"%char => true | "E"%char => true | "I"%char => true | "O"%char => true | "U"%char => true
  | _ => false
  end.

Fixpoint count_consonants (w : list ascii) : nat :=
  match w with
  | [] => 0
  | h :: t =>
    let n := nat_of_ascii h in
    let is_upper := (Nat.leb 65 n) && (Nat.leb n 90) in
    let is_lower := (Nat.leb 97 n) && (Nat.leb n 122) in
    let is_letter := is_upper || is_lower in
    (if is_letter && negb (is_vowel h) then 1 else 0) +
    count_consonants t
  end.

Definition split_words (s : list ascii) : list (list ascii) :=
  let space := " "%char in
  let fix aux (cur : list ascii) (rest : list ascii) : list (list ascii) :=
    match rest with
    | [] =>
      match cur with
      | [] => []
      | _ => [rev cur]
      end
    | h :: t =>
      if Ascii.eqb h space then
        match cur with
        | [] => aux [] t
        | _ => (rev cur) :: aux [] t
        end
      else
        aux (h :: cur) t
    end
  in aux [] s.

Definition select_words_impl (s : list ascii) (n : nat) : list (list ascii) :=
  filter (fun w => Nat.eqb (count_consonants w) n) (split_words s).

Definition select_words (s : string) (n : nat) : list string :=
  let l := list_ascii_of_string s in
  let res := select_words_impl l n in
  map string_of_list_ascii res.

(* 字符串只含字母与空格 *)
Definition problem_117_pre (s : string) : Prop :=
  let l := list_ascii_of_string s in
  Forall (fun c => c = " "%char \/ let n := nat_of_ascii c in (65 <= n /\ n <= 90) \/ (97 <= n /\ n <= 122)) l.

Definition problem_117_spec (s : string) (n : nat) (output : list string) : Prop :=
  output = select_words s n.