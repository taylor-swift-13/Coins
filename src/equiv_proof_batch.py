#!/usr/bin/env python3
"""
Batch script to run equiv_proof_gen.py
Uses hardcoded spec_id list
"""
import subprocess
import sys
import os
import re

# Hardcoded spec_id list
SPEC_ID_LIST = [
    "1", "6", "7", "15", "17", "19", "22", "29", "30", "32", "36", "39",
    "46", "55", "56", "61", "63", "64", "65", "66", "69", "70", "73",
    "78", "79", "84", "85", "86", "87", "91", "94", "101", "103", "104",
    "107", "108", "109", "112", "113", "116", "117", "119", "121", "123",
    "129", "130", "131", "132", "137", "139", "140", "142", "143", "144",
    "145", "146", "149", "155", "158", "160", "161"
]

# Underscore version list (if needed)
SPEC_ID_LIST_UNDERSCORE = [
    "137_", "160_"
]


def run_equiv_proof(spec_id: str, model_name: str = None, max_attempts: int = 3):
    """Run equiv_proof_gen.py for a single spec_id (using equiv directory)"""
    script_path = os.path.join(os.path.dirname(__file__), "equiv_proof_gen.py")
    
    cmd = ["python3", script_path, "--spec", spec_id]
    
    if model_name:
        cmd.extend(["--model", model_name])
    
    if max_attempts != 3:
        cmd.extend(["--max-attempts", str(max_attempts)])
    
    print(f"\n{'='*60}")
    print(f"Processing spec_id: {spec_id}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✓ {spec_id} processing complete")
        return True
    else:
        print(f"✗ {spec_id} processing failed")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch run equiv_proof_gen.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        metavar='MODEL_NAME',
        help='Model name (if not specified, uses default from equiv_proof_gen.py)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        metavar='N',
        help='Maximum number of attempts per direction (default: 3)'
    )
    parser.add_argument(
        '--include-underscore',
        action='store_true',
        help='Include underscore versions (e.g., 137_, 160_)'
    )
    
    args = parser.parse_args()
    
    # Hardcoded to use equiv directory, only process if files exist in both directories
    equiv_human_dir = os.path.join(os.path.dirname(__file__), "..", "equiv", "input", "human")
    equiv_llm_dir = os.path.join(os.path.dirname(__file__), "..", "equiv", "input", "llm")
    
    human_specs = set()
    llm_specs = set()
    
    if os.path.exists(equiv_human_dir):
        for f in os.listdir(equiv_human_dir):
            if f.endswith('.v'):
                spec_id = f[:-2]
                human_specs.add(spec_id)

    if os.path.exists(equiv_llm_dir):
        for f in os.listdir(equiv_llm_dir):
            if f.endswith('.v'):
                spec_id = f[:-2]
                llm_specs.add(spec_id)
    
    # Only process if files exist in both directories
    spec_ids = sorted(human_specs & llm_specs, key=lambda x: int(re.sub(r'[^0-9]', '', x)) if re.sub(r'[^0-9]', '', x) else 0)
    print(f"Using equiv directory mode")
    print(f"human directory file count: {len(human_specs)}")
    print(f"llm directory file count: {len(llm_specs)}")
    print(f"Common files count: {len(spec_ids)}")
    
    print(f"Starting batch processing of {len(spec_ids)} spec_ids...")
    print(f"Model: {args.model if args.model else 'default'}")
    print(f"Max attempts: {args.max_attempts}")
    
    success_count = 0
    fail_count = 0
    
    for spec_id in spec_ids:
        success = run_equiv_proof(spec_id, args.model, args.max_attempts)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"{'='*60}")
    print(f"Total: {len(spec_ids)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
