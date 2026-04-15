(* def compare(game,guess):
"""I think we all remember that feeling when the result of some long-awaited
event is finally known. The feelings and thoughts you have at that moment are
definitely worth noting down and comparing.
Your task is to determine if a person correctly guessed the results of a number of matches.
You are given two arrays of scores and guesses of equal length, where each index shows a match.
Return an array of the same length denoting how far off each guess was. If they have guessed correctly,
the value is 0, and if not, the value is the absolute difference between the guess and the score.


example:

compare([1,2,3,4,5,1],[1,2,3,4,2,-2]) -> [0,0,0,0,3,3]
compare([0,5,0,0,0,4],[4,1,1,0,0,-2]) -> [4,4,1,0,0,6]
""" *)
(* 导入整数库 ZArith 和列表库 List *)
Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Import ListNotations.

(* 输入两个列表长度必须相等 *)
Definition problem_152_pre (game guess : list Z) : Prop := length game = length guess.

(*
  compare_spec 定义了 'compare' 函数的规约。

  参数:
  - game: 一个 Z（整数）列表，代表实际分数。
  - guess: 一个 Z 列表，代表猜测的分数。
  - result: 一个 Z 列表，代表函数的输出。

  规约内容:
  该规约包含三个部分的合取（AND）：
  1. 输入列表 'game' 和 'guess' 的长度必须相等。
  2. 输出列表 'result' 的长度必须与输入列表的长度相等。
  3. 对于任意一个在列表长度范围内的合法索引 'i'（即 0 <= i < length game），
     输出列表 'result' 在索引 'i' 处的值，必须等于 'game' 和 'guess'
     在各自索引 'i' 处的值的差的绝对值。

  注意: `nth i l 0%Z` 用于获取列表 l 在索引 i 处的元素。
  第三个参数 `0%Z` 是一个默认值，如果索引越界则返回该值。
  但在我们的 `forall` 语句中，因为有 `(i < length game)%nat` 的前提条件，
  所以索引 `i` 永远不会越界。
*)
Definition problem_152_spec (game guess result : list Z) : Prop :=
  length game = length guess /\
  length result = length game /\
  forall i, (i < length game)%nat ->
    nth i result 0%Z = Z.abs (nth i game 0%Z - nth i guess 0%Z).