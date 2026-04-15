#!/usr/bin/env python3
import os
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from proof_gen import CoqProofGenerator


def list_spec_indices(spec_input_dir: str) -> List[str]:
    """List all specification file identifiers, supporting normal format (e.g., "1.v") and underscore format (e.g., "1_.v")"""
    spec_ids: List[str] = []
    for name in sorted(os.listdir(spec_input_dir)):
        if not name.endswith('.v'):
            continue
        # Match normal format: 1.v
        m1 = re.fullmatch(r'(\d+)\.v', name)
        if m1:
            spec_ids.append(m1.group(1))
        # Match underscore format: 1_.v
        m2 = re.fullmatch(r'(\d+)_\.v', name)
        if m2:
            spec_ids.append(f"{m2.group(1)}_")
    return sorted(spec_ids, key=lambda x: int(re.sub(r'[^0-9]', '', x)))


def run_batch(
    range_str: Optional[str],
    max_attempts: int,
    num_workers: int = 24,
    model_name: Optional[str] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Run batch proof generation
    
    Args:
        range_str: Range string in format "START:END" (both START and END are inclusive)
        max_attempts: Maximum number of attempts per specification
        num_workers: Number of parallel worker threads (default: 24)
    
    Returns:
        (successes, failures): Lists of successful and failed specification identifiers
    """
    generator_kwargs = {"model_name": model_name} if model_name else {}
    spec_ids = list_spec_indices(CoqProofGenerator(**generator_kwargs).spec_input_dir)
    
    # Filter based on numeric part
    if range_str:
        try:
            start, end = map(int, range_str.split(':'))
            # Note: end is inclusive here, different from coq_gen.py's range(start, end)
            spec_ids = [sid for sid in spec_ids 
                       if start <= int(re.sub(r'[^0-9]', '', sid)) <= end]
        except (ValueError, AttributeError):
            raise ValueError(f'--range parameter format invalid, should be START:END (e.g., 1:10)')

    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    
    def process_spec(spec_id: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Wrapper function to process a single spec
        
        Returns:
            (spec_id, status, message): status is 'success', 'failure', 'skip' or 'error'
        """
        try:
            # Each thread creates its own CoqProofGenerator instance
            gen = CoqProofGenerator(**generator_kwargs)
            
            input_path = os.path.join(gen.spec_input_dir, f"{spec_id}.v")
            if not os.path.exists(input_path):
                print(f"[Thread] [SKIP] spec {spec_id} -> Input file does not exist: {input_path}")
                return (spec_id, 'skip', f"Input file does not exist: {input_path}")
            
            print(f"[Thread] [START] Processing spec {spec_id}")
            
            # Check if output already exists
            output_path = os.path.join(gen.spec_output_dir, f"{spec_id}.v")
            if os.path.exists(output_path):
                print(f"[Thread] [SKIP] {spec_id} -> output already exists: {output_path}")
                return (spec_id, 'success', "Output already exists")
            
            res = gen.generate_proof_for_spec(spec_id, max_attempts=max_attempts)
            if res.startswith("Failed to generate valid proof") or res.startswith("Error generating proof"):
                print(f"[Thread] [FAIL] {spec_id}")
                return (spec_id, 'failure', res)
            else:
                print(f"[Thread] [OK] {spec_id}")
                return (spec_id, 'success', res)
        except Exception as e:
            error_msg = f"Error processing spec {spec_id}: {e}"
            print(f"[Thread] [ERROR] {error_msg}")
            return (spec_id, 'error', error_msg)
    
    if num_workers <= 1:
        # Single-threaded mode
        for spec_id in spec_ids:
            spec_id_result, status, message = process_spec(spec_id)
            if status == 'success':
                successes.append(spec_id_result)
            elif status == 'skip':
                successes.append(spec_id_result)  # skip counts as success (already exists)
            elif status == 'failure':
                failures.append((spec_id_result, message or ""))
            elif status == 'error':
                failures.append((spec_id_result, message or ""))
    else:
        # Multi-threaded mode
        print(f"\n=== Using {num_workers} workers for parallel processing ===")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_spec = {
                executor.submit(process_spec, spec_id): spec_id 
                for spec_id in spec_ids
            }
            
            # Collect results
            for future in as_completed(future_to_spec):
                spec_id = future_to_spec[future]
                try:
                    spec_id_result, status, message = future.result()
                    if status == 'success':
                        successes.append(spec_id_result)
                    elif status == 'skip':
                        successes.append(spec_id_result)  # skip counts as success (already exists)
                    elif status == 'failure':
                        failures.append((spec_id_result, message or ""))
                    elif status == 'error':
                        failures.append((spec_id_result, message or ""))
                except Exception as e:
                    error_msg = f"Error processing spec {spec_id}: {e}"
                    print(f"[ERROR] {error_msg}")
                    failures.append((spec_id, error_msg))
    
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Batch run Coq proof generation, processing all specifications in spec/input/*.v',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --range 1:10
  %(prog)s --range 1:100 --max-attempts 5
  %(prog)s --range 1:50 --num-workers 16
  %(prog)s --max-attempts 3
        """
    )
    parser.add_argument(
        '--range',
        type=str,
        default=None,
        metavar='START:END',
        help='Process specification range, format: START:END (both START and END are inclusive, e.g., 1:10)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        metavar='N',
        help='Maximum number of attempts per specification (default: 3)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=24,
        metavar='N',
        help='Number of parallel worker threads (default: 24)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        metavar='MODEL_NAME',
        help='Model name used to generate proofs (default: value from config.py)'
    )
    args = parser.parse_args()

    try:
        successes, failures = run_batch(args.range, args.max_attempts, args.num_workers, args.model)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(2)

    total = len(successes) + len(failures)
    success_rate = (len(successes) / total * 100) if total > 0 else 0.0

    print("\n===== Summary =====")
    print(f"Total: {total}")
    print(f"Success: {len(successes)} ({success_rate:.2f}%) -> {successes}")
    print(f"Fail: {len(failures)}")
    for spec_id, msg in failures:
        short = msg[:200].replace("\n", " ").replace("\r", " ")
        print(f"- {spec_id}: {short}...")


if __name__ == '__main__':
    main()
