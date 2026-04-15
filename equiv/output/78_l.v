Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.

Import ListNotations.

(* ================================================================= *)
(* First Specification (Human-written)                               *)
(* ================================================================= *)

Open Scope string_scope.

Definition is_prime_hex_digit (c : ascii) : bool :=
  match c with
  | "2"%char | "3"%char | "5"%char | "7"%char
  | "B"%char | "D"%char => true
  | _ => false
  end.

Fixpoint count_prime_hex (s : string) : nat :=
  match s with
  | "" => 0
  | String h t =>
    (if is_prime_hex_digit h then 1 else 0) +
    count_prime_hex t
  end.

Definition hex_key_impl (s : string) : nat :=
  count_prime_hex s.


Definition problem_78_pre (s : string) : Prop := True.

Definition problem_78_spec (s : string) (output : nat) : Prop :=
  output = hex_key_impl s.

(* ================================================================= *)
(* Second Specification (LLM-generated)                              *)
(* ================================================================= *)

Open Scope char_scope.

Definition is_prime_hex (c : ascii) : bool :=
  match c with
  | "2" => true
  | "3" => true
  | "5" => true
  | "7" => true
  | "B" => true
  | "D" => true
  | _ => false
  end.

Fixpoint count_primes (s : string) : nat :=
  match s with
  | EmptyString => 0
  | String c rest => 
      (if is_prime_hex c then 1 else 0) + count_primes rest
  end.

Definition hex_key_spec (num : string) (count : nat) : Prop :=
  count = count_primes num.

(* ================================================================= *)
(* Proof of Implication                                              *)
(* ================================================================= *)

(* 
   Lemma to prove that the character classification functions are equivalent.
   Since ascii is a finite type (8 bits), we can prove this by exhaustively 
   destructing the boolean components of the ascii character.
*)
Lemma is_prime_hex_equiv : forall c : ascii, is_prime_hex_digit c = is_prime_hex c.
Proof.
  intros c.
  (* 
     Destruct c into its 8 boolean components. 
     This generates 2^8 = 256 cases, which Coq handles easily with reflexivity.
  *)
  destruct c as [b0 b1 b2 b3 b4 b5 b6 b7].
  destruct b0, b1, b2, b3, b4, b5, b6, b7; reflexivity.
Qed.

(*
   Lemma to prove that the recursive counting functions are equivalent
   for any input string.
*)
Lemma count_primes_equiv : forall s : string, count_prime_hex s = count_primes s.
Proof.
  intros s.
  induction s as [| c s' IHs'].
  - (* Base case: Empty string *)
    simpl. reflexivity.
  - (* Inductive step: String c s' *)
    simpl.
    rewrite is_prime_hex_equiv. (* Use the character equivalence lemma *)
    rewrite IHs'.               (* Use the inductive hypothesis *)
    reflexivity.
Qed.

(*
   Main Theorem: Proving that Spec 1 implies Spec 2.
   If problem_78_spec holds (and the trivial precondition), 
   then hex_key_spec must also hold.
*)
Theorem spec1_implies_spec2 : 
  forall (s : string) (output : nat),
  problem_78_pre s ->
  problem_78_spec s output ->
  hex_key_spec s output.
Proof.
  intros s output Pre Spec1.
  
  (* Unfold definitions to expose the underlying equality *)
  unfold problem_78_spec, hex_key_impl in Spec1.
  unfold hex_key_spec.
  
  (* Rewrite the output using the hypothesis from Spec 1 *)
  rewrite Spec1.
  
  (* Apply the equivalence lemma between the two implementations *)
  apply count_primes_equiv.
Qed.