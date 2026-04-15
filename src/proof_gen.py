#!/usr/bin/env python3
import argparse
import os
import re
import sys
import subprocess
import tempfile
from typing import Any, List, Tuple
from extract_pairs import HumanEvalPairs
from config import (
    LLMConfig,
    TYPE as _TYPE,
    MODEL_NAME as _MODEL_NAME,
    DATASET_PATH as _DATASET_PATH,
    SPEC_INPUT_DIR as _SPEC_INPUT_DIR,
    SPEC_OUTPUT_DIR as _SPEC_OUTPUT_DIR,
    ROOT_DIR as _ROOT_DIR,
    MAX_ITERATIONS as _MAX_ITERATIONS,
)
from llm import Chatbot
from logger import get_logger


class CoqProofGenerator:
    """Coq proof generator: generates proof for test case 0 (specification/proof)"""
    
    def __init__(
        self,
        dataset_path: str = _DATASET_PATH,
        spec_input_dir: str = _SPEC_INPUT_DIR,
        spec_output_dir: str = _SPEC_OUTPUT_DIR,
        max_iterations: int = _MAX_ITERATIONS,
        model_name: str = _MODEL_NAME,
    ):
        self.model_name = model_name
        self.type_name = _TYPE
        self.root_dir = _ROOT_DIR
        self.dataset_path = dataset_path
        self.max_iterations = max_iterations
        self.pairs_helper = HumanEvalPairs(dataset_path)
        self.llm_config = LLMConfig(api_model=self.model_name)
        self.llm = Chatbot(self.llm_config)
        self.logger = None  # Will be initialized when needed

        # TYPE only determines input directory path, output structure is unified
        self.spec_input_dir = spec_input_dir
        self.spec_output_dir = os.path.join(spec_output_dir, self.model_name)
        self.spec_bad_dir = os.path.join(os.path.dirname(spec_output_dir), "bad", self.model_name)
    
    def get_pairs(self, i: int, one_based: bool = False) -> List[Any]:
        """Get test case pairs for program i"""
        return self.pairs_helper.get_pairs(i, one_based=one_based)
    
    def get_pair_line(self, i: int, j: int, one_based: bool = False) -> Any:
        """Get the j-th test case for program i"""
        results = self.get_pairs(i, one_based=one_based)
        idx = j - 1 if one_based else j
        if idx < 0 or idx >= len(results):
            raise IndexError(f'j={j} out of range (n={len(results)})')
        return results[idx]
    
    def read_coq_spec(self, spec_id: str) -> str:
        """Read Coq specification file (from input directory)
        
        Args:
            spec_id: Specification identifier, can be a number (e.g., "1") or underscore format (e.g., "1_")
        """
        spec_path = os.path.join(self.spec_input_dir, f"{spec_id}.v")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path}")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def format_example_for_coq(self, example: List[Any]) -> str:
        """Convert Python test case to Coq format"""
        def convert_value(val):
            if isinstance(val, bool):
                return "true" if val else "false"
            elif isinstance(val, int):
                return f"{val}%Z"
            elif isinstance(val, float):
                return f"{val}%R"
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, list):
                if not val:
                    return "[]"
                elements = [convert_value(x) for x in val]
                return f"[{'; '.join(elements)}]"
            else:
                return str(val)
        
        if len(example) == 2:
            input_val, output_val = example
            return f"input = {convert_value(input_val)}, output = {convert_value(output_val)}"
        else:
            return str(example)
    
    def call_llm_for_proof(self, spec_content: str, example: str, error_info: str = "") -> str:
        """Call LLM to generate Coq proof"""

        base_prompt = f"""Please generate a complete Coq proof for the following specification and test case.
Specification definition:
{spec_content}

Test case:
{example}

Please generate an Example proof, including the full Proof steps and ending with Qed. If additional imports or lemmas are required, please also provide them.

Notes:
1. Use appropriate Coq tactics (such as unfold, simpl, reflexivity, lia, lra, etc.)
2. Ensure the proof is complete and correct
3. If real number operations are involved, please use R_scope
4. If integer operations are involved, please use Z_scope
5. For list operations, please use the appropriate List library functions

Please return Coq code directly, without any additional explanations:
"""

        if error_info:
            prompt = f"""{base_prompt}

IMPORTANT: The previous proof attempt failed with the following error:
{error_info}

Please fix the proof based on this error information and generate a corrected version:
"""     
        else:
            prompt = base_prompt

        proof = self.llm.chat(prompt)
        
        # Remove possible code block markers
        proof = proof.strip()
        if proof.startswith('```coq'):
            proof = proof[6:]
        if proof.startswith('```'):
            proof = proof[3:]
        if proof.endswith('```'):
            proof = proof[:-3]
        proof = proof.strip()
        
        return proof
    
    def verify_coq_proof(self, combined_content: str) -> Tuple[bool, str]:
        """Verify if Coq proof is correct: verify (original spec + generated proof) combined content"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(combined_content)
                temp_file.flush()
                result = subprocess.run(
                    ['coqtop', '-batch', '-l', temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                os.unlink(temp_file.name)
                if result.returncode == 0:
                    return True, ""
                else:
                    return False, result.stderr + result.stdout
        except subprocess.TimeoutExpired:
            return False, "Coq verification timed out"
        except Exception as e:
            return False, f"Error during verification: {e}"
    
    def save_bad_proof(self, spec_id: str, test_case_id: int, proof: str) -> None:
        """Save failed proof to bad folder
        
        Args:
            spec_id: Specification identifier
            test_case_id: Test case ID (for proof_gen, this is the first test case, usually 1)
            proof: Failed proof content
        """
        os.makedirs(self.spec_bad_dir, exist_ok=True)
        bad_file = os.path.join(self.spec_bad_dir, f"{spec_id}_{test_case_id}.v")
        with open(bad_file, 'w', encoding='utf-8') as f:
            f.write(proof)
        if self.logger:
            self.logger.info(f"Saved bad proof to: {bad_file}")
    
    def generate_proof_for_spec(self, spec_id: str, max_attempts: int = None) -> str:
        """Generate proof for specification with iterative fixing. Reads input/{spec_id}.v, writes to output/{spec_id}.v on success.
        
        Args:
            spec_id: Specification identifier, can be a number (e.g., "1") or underscore format (e.g., "1_")
            max_attempts: Maximum number of attempts
        """
        if max_attempts is None:
            max_attempts = self.max_iterations

        test_id = 1
        
        # Extract numeric part from spec_id for getting test cases
        # e.g., "1_" -> 1, "123_" -> 123, "1" -> 1
        numeric_id = int(re.sub(r'[^0-9]', '', spec_id)) if spec_id else 0
        
        # Initialize logger
        self.logger = get_logger(model_name=f"{self.model_name}", spec_id=numeric_id, test_id=test_id, type_name=self.type_name)
        
        try:
            # Paths and content
            input_path = os.path.join(self.spec_input_dir, f"{spec_id}.v")
            output_path = os.path.join(self.spec_output_dir, f"{spec_id}.v")
            spec_content = self.read_coq_spec(spec_id)
            
            # Get first test case (using numeric ID)
            example = self.get_pair_line(numeric_id, test_id-1)
            example_str = self.format_example_for_coq(example)
            print(f"Example: {example_str}")

            self.logger.print_section(f"Starting proof generation for Spec {spec_id}")
            self.logger.info(f"Input file: {input_path}")
            self.logger.info(f"Output file: {output_path}")
            self.logger.info(f"Test case: {example_str}")
            
            error_info = ""
            
            for attempt in range(max_attempts):
                self.logger.info(f"Attempt {attempt + 1}/{max_attempts} for spec {spec_id}")
                
                # Generate proof
                proof = self.call_llm_for_proof(spec_content, example_str, error_info)
                
                # Output generated proof
                self.logger.print_section(f"Generated Proof for Attempt {attempt + 1}")
                self.logger.info(proof)
                
                # Combine content for verification (original spec + proof)
                proof = proof.strip()
                is_valid, error_output = self.verify_coq_proof(proof)
                
                if is_valid:
                    self.logger.info(f"✓ Proof generated successfully for spec {spec_id}")
                    # Write to output/{spec_id}.v
                    os.makedirs(self.spec_output_dir, exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(proof)
                    self.logger.info(f"✓ Proof saved to: {output_path}")
                    self.logger.info(f"Run command: coqtop -batch -l {output_path}")
                    self.logger.close()
                    return proof
                else:
                    self.logger.error(f"✗ Proof failed for spec {spec_id}, attempt {attempt + 1}")
                    self.logger.error(f"Error: {error_output}")
                    error_info = error_output
                    if attempt == max_attempts - 1:
                        # Save last failed proof to bad folder
                        self.save_bad_proof(spec_id, test_id, proof)
                        self.logger.error(f"✗ Failed to generate valid proof after {max_attempts} attempts")
                        self.logger.close()
                        return f"Failed to generate valid proof after {max_attempts} attempts. Last error: {error_output}"
            
            self.logger.error(f"✗ Unexpected error in proof generation for spec {spec_id}")
            self.logger.close()
            return f"Unexpected error in proof generation for spec {spec_id}"
            
        except Exception as e:
            self.logger.error(f"✗ Error generating proof for spec {spec_id}: {e}")
            self.logger.close()
            return f"Error generating proof for spec {spec_id}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Coq proof for HumanEval specification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1
  %(prog)s 1_ --max-attempts 5
  %(prog)s 42 --output proof.v
        """
    )
    parser.add_argument(
        'spec_id',
        type=str,
        metavar='SPEC_ID',
        help='Specification identifier, can be a number (e.g., "1") or underscore format (e.g., "1_")'
    )
    parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='Output file path (default: output to stdout)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=_MAX_ITERATIONS,
        metavar='N',
        help=f'Maximum number of attempts (default: {_MAX_ITERATIONS})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=_MODEL_NAME,
        metavar='MODEL_NAME',
        help=f'Model name used to generate proofs (default: {_MODEL_NAME})'
    )
    args = parser.parse_args()
    
    generator = CoqProofGenerator(model_name=args.model)
    proof = generator.generate_proof_for_spec(args.spec_id, args.max_attempts)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(proof)
        print(f"Proof written to {args.output}")
    else:
        print(proof)


if __name__ == '__main__':
    main()
