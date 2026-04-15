#!/usr/bin/env python3
"""
Generate Coq specifications from HumanEval-style JSONL data.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from config import DATASET_PATH, LLMConfig, TYPE as _TYPE
from llm import Chatbot


def open_maybe_gzip(path: str):
    """Open a plain-text or gzipped JSONL file."""
    if path.endswith(".gz"):
        import gzip
        import io

        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dict rows."""
    rows: List[Dict] = []
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def extract_task_id(obj: Dict) -> str:
    """Extract a stable task identifier from a dataset row."""
    return obj.get("task_id") or obj.get("id") or ""


def strip_code_fence(text: str) -> str:
    """Remove common markdown code fences from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```coq"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def build_prompt(problem_prompt: str, reference_impl: str, include_reference_impl: bool) -> str:
    """Build the prompt used for Coq specification generation."""
    shared_prefix = """You are an expert in writing Coq specifications.
Your task is to generate a precise and complete Coq specification for the described program.

Rules
- Include the necessary Require Import statements.
- Output only Coq code.
- Do not generate any natural language comments.
- The result should be a relational specification in Coq rather than an implementation translation.

Before generating the output, refer to the following example:

[Example]
[Input]
Description:
def add(A: int, B: int):
    \"\"\"
    return the sum of A and B.
    \"\"\"

Reference implementation:
    return A + B

[Output]
```coq
Definition add_spec (A B sum : Z) : Prop :=
  sum = A + B.
```"""

    if include_reference_impl:
        return """{shared_prefix}

Now, this is the description:
{problem_prompt}

And this is the reference implementation:
{reference_impl}

Please output only the Coq specification for this program.""".format(
            shared_prefix=shared_prefix,
            problem_prompt=problem_prompt,
            reference_impl=reference_impl,
        )

    return """{shared_prefix}

Now, this is the description:
{problem_prompt}

Do not assume access to any reference implementation beyond the description above.
Please output only the Coq specification for this program.""".format(
        shared_prefix=shared_prefix,
        problem_prompt=problem_prompt,
    )


def parse_tasks(args: argparse.Namespace, total_tasks: int) -> List[int]:
    """Resolve CLI task selectors into a validated task index list."""
    tasks_to_process: List[int] = []

    if args.spec is not None:
        tasks_to_process = [args.spec]
    elif args.specs:
        try:
            tasks_to_process = [int(item.strip()) for item in args.specs.split(",")]
        except ValueError:
            raise ValueError("Invalid --specs format, expected comma-separated integers such as 0,1,2")
    elif args.range:
        try:
            start, end = map(int, args.range.split(":"))
        except ValueError:
            raise ValueError("Invalid --range format, expected START:END such as 0:10")
        if start > end:
            raise ValueError("--range requires START <= END")
        tasks_to_process = list(range(start, end if start == end else end))
        if start == end:
            tasks_to_process = [start]
    elif args.all:
        tasks_to_process = list(range(total_tasks))

    invalid_tasks = [task for task in tasks_to_process if task < 0 or task >= total_tasks]
    if invalid_tasks:
        raise ValueError(f"Task numbers out of range 0-{total_tasks - 1}: {invalid_tasks}")

    return tasks_to_process


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Coq specifications from HumanEval data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model gpt-4o --spec 0
  %(prog)s --model gpt-4o --type llm --spec 0
  %(prog)s --model gpt-4o --spec 0 --without-reference-impl
  %(prog)s --model gpt-4o --specs 0,1,2,5
  %(prog)s --model gpt-4o --range 0:10
  %(prog)s --model gpt-4o --all
        """,
    )

    parser.add_argument("--model", required=True, metavar="MODEL", help="LLM model name")
    parser.add_argument(
        "--type",
        default=_TYPE,
        metavar="TYPE_NAME",
        help=f"Target spec folder name under spec/<type>/input (default: {_TYPE})",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        metavar="PATH",
        help=f"Dataset path (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory override (default: spec/<type>/input)",
    )
    parser.add_argument(
        "--without-reference-impl",
        action="store_true",
        help="Do not include canonical_solution in the generation prompt",
    )

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--spec", type=int, metavar="N", help="Process a single task number")
    task_group.add_argument("--specs", metavar="N1,N2,...", help="Process multiple task numbers")
    task_group.add_argument("--range", metavar="START:END", help="Process task range")
    task_group.add_argument("--all", action="store_true", help="Process all tasks")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: File not found: {args.dataset}", file=sys.stderr)
        return 2

    print(f"Loading JSONL file: {args.dataset}")
    jsonl_data = load_jsonl(args.dataset)
    if not jsonl_data:
        print("Error: JSONL file is empty or invalid", file=sys.stderr)
        return 2

    print(f"Loaded {len(jsonl_data)} tasks")

    try:
        tasks_to_process = parse_tasks(args, len(jsonl_data))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not tasks_to_process:
        print("Error: No tasks specified to process", file=sys.stderr)
        return 2

    print(f"Will process {len(tasks_to_process)} tasks: {tasks_to_process}")
    print(f"Reference implementation in prompt: {not args.without_reference_impl}")
    print(f"Target spec folder type: {args.type}")

    print(f"\nInitializing LLM model: {args.model}")
    llm_config = LLMConfig(api_model=args.model)
    chatbot = Chatbot(llm_config)

    output_dir = Path(args.output_dir) if args.output_dir else Path("spec") / args.type / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    for idx, task_num in enumerate(tasks_to_process, start=1):
        obj = jsonl_data[task_num]
        task_id = extract_task_id(obj)
        output_file = output_dir / f"{task_num}.v"

        print(f"\n[{idx}/{len(tasks_to_process)}] Processing task {task_num} ({task_id})...")

        problem_prompt = obj.get("prompt", "")
        reference_impl = obj.get("canonical_solution", "")

        if not problem_prompt:
            print(f"  Warning: Task {task_num} is missing prompt, skipping", file=sys.stderr)
            error_count += 1
            continue

        if not args.without_reference_impl and not reference_impl:
            print(
                f"  Warning: Task {task_num} is missing canonical_solution required by the current mode, skipping",
                file=sys.stderr,
            )
            error_count += 1
            continue

        full_prompt = build_prompt(
            problem_prompt=problem_prompt,
            reference_impl=reference_impl,
            include_reference_impl=not args.without_reference_impl,
        )

        try:
            print("  Calling LLM...")
            response = chatbot.new_chat(full_prompt)
            output_file.write_text(strip_code_fence(response), encoding="utf-8")
            print(f"  Saved to: {output_file}")
            success_count += 1
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            error_count += 1

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total: {len(tasks_to_process)}")
    print(f"Output directory: {output_dir}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
