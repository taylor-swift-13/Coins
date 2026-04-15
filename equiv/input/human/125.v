(* Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
Examples
split_words("Hello world!") ➞ ["Hello", "world!"]
split_words("Hello,world!") ➞ ["Hello", "world!"]
split_words("abcdef") == 3 *)

(* 引入必要的 Coq 库 *)
Require Import Coq.Lists.List.
Require Import Coq.Strings.Ascii.
Require Import Coq.Strings.String.
Require Import Coq.Arith.PeanoNat.
Import ListNotations.


(*
  辅助函数定义
*)

(* 1. 检查一个 (list ascii) 是否包含一个特定的 ascii 字符。 *)
Definition contains (l : list ascii) (c : ascii) : bool :=
  existsb (fun x => Ascii.eqb x c) l.

(* 2. 将一个 (list ascii) 按给定的分隔符拆分为一个单词列表 (list (list ascii))。*)
Definition split (sep : ascii) (s : list ascii) : list (list ascii) :=
  let s' := s ++ [sep] in
  let f (acc : list (list ascii) * list ascii) (c : ascii) :=
    let (res, current_word) := acc in
    if Ascii.eqb c sep then
      match current_word with
      | [] => (res, []) (* 如果当前单词为空，则忽略它 *)
      | _ :: _ => (res ++ [rev current_word], []) (* 否则，将反转后的当前单词添加到结果中 *)
      end
    else
      (res, c :: current_word)
  in
  let (res, _) := fold_left f s' ([], []) in
  res.

(* 3. 计算列表中字母序为奇数的小写字母的数量。
      'a' 的序数为 0，'b' 为 1，以此类推。 *)
Definition count_odd_lowercase (l : list ascii) : nat :=
  (* 定义一个函数，根据规约将小写字母映射到自然数 0-25。
     任何非小写字母的字符都映射到 0 (一个偶数)，因此它们不会被计数。*)
  let lowercase_ord (c : ascii) : nat :=
    match c with
    | "a"%char => 0 | "b"%char => 1 | "c"%char => 2 | "d"%char => 3
    | "e"%char => 4 | "f"%char => 5 | "g"%char => 6 | "h"%char => 7
    | "i"%char => 8 | "j"%char => 9 | "k"%char => 10 | "l"%char => 11
    | "m"%char => 12 | "n"%char => 13 | "o"%char => 14 | "p"%char => 15
    | "q"%char => 16 | "r"%char => 17 | "s"%char => 18 | "t"%char => 19
    | "u"%char => 20 | "v"%char => 21 | "w"%char => 22 | "x"%char => 23
    | "y"%char => 24 | "z"%char => 25
    | _ => 0
    end
  in
  (* 检查一个自然数是否为奇数 *)
  let is_odd (n : nat) : bool :=
    negb (Nat.eqb (Nat.modulo n 2) 0)
  in
  (* 定义目标字符的判断条件。
     is_lower 的检查是多余的，因为任何非小写字母通过 lowercase_ord 都会得到 0，
     而 is_odd 0 是 false，所以它们不会被计入。*)
  let is_target_char (c : ascii) : bool :=
    is_odd (lowercase_ord c)
  in
  List.length (filter is_target_char l).

(* 输入可为任意字符列表 *)
Definition problem_125_pre (input : string) : Prop := True.
(*
  程序规约 (Program Specification)
*)

Definition  problem_125_spec (input : string) (output : sum (list string) nat) : Prop :=
  let l := list_ascii_of_string input in
  if contains l " "%char then
    let res := split " "%char l in
    output = inl (map string_of_list_ascii res)
  else if contains l ","%char then
    let res := split ","%char l in
    output = inl (map string_of_list_ascii res)
  else
    output = inr (count_odd_lowercase l).