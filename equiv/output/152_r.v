(* 
   This is a complete, standalone Coq file proving that the first specification 
   (LLM-generated) implies the second specification (human-written).
*)

(* ========================================================================= *)
(* 1. Imports                                                                *)
(* ========================================================================= *)

Require Import Coq.ZArith.ZArith.
Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.
Import ListNotations.

(* Set Scope for Z arithmetic *)
Open Scope Z_scope.

(* ========================================================================= *)
(* 2. First Specification (LLM-generated)                                    *)
(* ========================================================================= *)

Definition abs_diff (a b : Z) : Z :=
  Z.abs (a - b).

Fixpoint compare_spec (game : list Z) (guess : list Z) (result : list Z) : Prop :=
  match game, guess, result with
  | [], [], [] => True
  | g :: gs, gu :: gus, r :: rs => r = abs_diff g gu /\ compare_spec gs gus rs
  | _, _, _ => False
  end.

(* ========================================================================= *)
(* 3. Second Specification (Human-written)                                   *)
(* ========================================================================= *)

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

(* ========================================================================= *)
(* 4. Theorem and Proof                                                      *)
(* ========================================================================= *)

(* 
   Theorem: If inputs satisfy compare_spec (LLM), they satisfy problem_152_spec (Human). 
*)
Theorem spec1_implies_spec2 : forall game guess result,
  compare_spec game guess result -> problem_152_spec game guess result.
Proof.
  intros game.
  (* We use induction on the 'game' list. We generalize guess and result 
     so the induction hypothesis applies to the tails of these lists. *)
  induction game as [| g gs IHgs].
  
  (* Base Case: game = [] *)
  - intros guess result Hspec.
    (* Destruct guess and result to handle the match in compare_spec *)
    destruct guess; destruct result; try (simpl in Hspec; contradiction).
    (* Case: game=[], guess=[], result=[] *)
    unfold problem_152_spec.
    simpl.
    repeat split; try reflexivity.
    (* The forall quantifier over an empty range is vacuously true *)
    intros i Hi.
    inversion Hi.

  (* Inductive Step: game = g :: gs *)
  - intros guess result Hspec.
    (* Destruct guess and result to expose heads and tails *)
    destruct guess as [| gu gus]; destruct result as [| r rs]; try (simpl in Hspec; contradiction).
    (* Case: game=g::gs, guess=gu::gus, result=r::rs *)
    simpl in Hspec.
    destruct Hspec as [Hhead Htail].
    
    (* Apply the Induction Hypothesis to the tails *)
    apply IHgs in Htail.
    
    (* Unfold the target spec to prove its components *)
    unfold problem_152_spec in *.
    destruct Htail as [Hlen1 [Hlen2 Hnth]].
    
    split.
    (* 1. length game = length guess *)
    { simpl. f_equal. assumption. }
    
    split.
    (* 2. length result = length game *)
    { simpl. f_equal. assumption. }
    
    (* 3. nth element property *)
    intros i Hi.
    destruct i as [| i'].
    + (* Case i = 0: Check head elements *)
      simpl.
      unfold abs_diff in Hhead.
      rewrite Hhead.
      reflexivity.
    + (* Case i = S i': Check tail elements using IH *)
      simpl.
      apply Hnth.
      simpl in Hi.
      (* Prove i' < length gs from S i' < length (g::gs) using Nat.succ_lt_mono *)
      apply Nat.succ_lt_mono.
      assumption.
Qed.