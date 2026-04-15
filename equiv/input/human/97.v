(*def multiply(a, b):
"""Complete the function that takes two integers and returns
the product of their unit digits.
Assume the input is always valid.
Examples:
multiply(148, 412) should return 16.
multiply(19, 28) should return 72.
multiply(2020, 1851) should return 0.
multiply(14,-15) should return 20.
""" *)
(* 导入 Coq 中整数（Z）及其算术运算的库 *)
Require Import ZArith.
Open Scope Z_scope.

(* 任意两个整数输入均合法 *)
Definition problem_97_pre (a b : Z) : Prop := True.

(*
  multiply_spec_abs 定义了程序 multiply 的一个更精确的规约。
  它是一个命题（Prop），描述了输入 a, b 和输出 r 之间的关系。
  - a: 第一个输入整数。
  - b: 第二个输入整数。
  - r: 函数的返回值。
  规约指出，返回值 r 必须等于 a 的绝对值的个位数与 b 的绝对值的个位数的乘积。
  使用 Z.abs 可以确保对于负数（例如 -15），我们取其个位数（5）而不是负的余数。
*)
Definition problem_97_spec (a b r : Z) : Prop :=
  r = (Z.abs a mod 10) * (Z.abs b mod 10).