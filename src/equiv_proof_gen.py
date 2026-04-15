#!/usr/bin/env python3
"""
Generate implication proof script between two specs
- Input: equiv/input/human/id.v and equiv/input/llm/id.v
- Output:
  - equiv/output/id_l.v: proof that human implies llm
  - equiv/output/id_r.v: proof that llm implies human
"""
import argparse
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
from config import LLMConfig, MODEL_NAME as _MODEL_NAME
from llm import Chatbot
from logger import get_logger


class EquivProofGenerator:
    """Generator for implication proofs between two specs"""
    
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        
        # Hardcoded to use equiv directory
        script_dir = Path(__file__).parent.parent
        equiv_dir = script_dir / "equiv"
        self.root_dir = str(equiv_dir)
        self.human_input_dir = os.path.join(self.root_dir, "input", "human")
        self.llm_input_dir = os.path.join(self.root_dir, "input", "llm")
        self.equiv_output_dir = os.path.join(self.root_dir, "output")

        # Ensure output directory exists
        os.makedirs(self.equiv_output_dir, exist_ok=True)
        
        # Initialize LLM
        self.llm_config = LLMConfig(api_model=self.model_name)
        self.llm = Chatbot(self.llm_config)
        self.logger = None
    
    def read_spec(self, spec_path: str) -> str:
        """Read spec file content"""
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path}")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_implication_proof(self, spec1: str, spec2: str, direction: str, error_info: str = "") -> str:
        """Use LLM to generate proof that spec1 implies spec2
        
        Args:
            spec1: First spec content
            spec2: Second spec content
            direction: "l" means human -> llm, "r" means llm -> human
            error_info: Previous error information (if any)
        """
        if direction == "l":
            spec1_name = "human-written spec"
            spec2_name = "LLM-generated spec"
        else:
            spec1_name = "LLM-generated spec"
            spec2_name = "human-written spec"
        
        base_prompt = f"""
### Role: ###
You are a Coq proof assistant. Given two Coq specifications, your task is to generate a COMPLETE, STANDALONE Coq file that proves the first specification logically implies the second specification.

This means: if all the definitions, properties, and constraints in the first specification hold, then all the definitions, properties, and constraints in the second specification should also hold.

First specification ({spec1_name}):
```coq
{spec1}
```

Second specification ({spec2_name}):
```coq
{spec2}
```

Your task is to generate a COMPLETE Coq file that:
1. Includes ALL necessary Require Import statements from both specifications
2. Includes ALL definitions from the first specification ({spec1_name})
3. Includes ALL definitions from the second specification ({spec2_name})
4. States a theorem that expresses the implication relationship (spec1 implies spec2)
5. Provides a complete proof using appropriate Coq tactics
6. Is syntactically correct and can be verified by Coq independently

### Rules: ###
- The generated code must be a COMPLETE, STANDALONE Coq file
- It must include all Require statements needed
- It must include all definitions from both specs (copy them into the proof file)
- The proof should be self-contained and verifiable without needing to reference the original spec files
- Use appropriate tactics like unfold, simpl, reflexivity, apply, rewrite, etc.
- If both specs define the same concepts with different names, you may need to prove equivalence or rename them appropriately

Generate ONLY the complete Coq code, without any markdown formatting or explanations outside the code. The output should be a valid, complete Coq file that can be directly compiled."""
        
        if error_info:
            prompt = f"""{base_prompt}

IMPORTANT: The previous proof attempt failed with the following error:
{error_info}

Please fix the proof based on this error information and generate a corrected version:"""
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
    
    def verify_coq_proof(self, proof_content: str) -> Tuple[bool, str]:
        """Verify if Coq proof is correct
        
        Args:
            proof_content: Generated complete proof content (should be a complete, standalone Coq file)
        
        Returns:
            (success, error_message)
        """
        try:
            # Directly verify generated proof file, no need to combine with other content
            # Because generated proof should be complete and standalone
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(proof_content)
                temp_file.flush()
                result = subprocess.run(
                    ['coqtop', '-batch', '-l', temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=60
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
    
    def generate_proofs_for_spec(self, spec_id: str, max_attempts: int = 3) -> Tuple[bool, bool]:
        """Generate implication proofs in both directions for specified spec_id
        
        Args:
            spec_id: Specification identifier (e.g., "1", "122")
            max_attempts: Maximum number of attempts
        
        Returns:
            (l_proof_success, r_proof_success) Whether proofs in both directions succeeded
        """
        # Initialize logger
        numeric_id = int(re.sub(r'[^0-9]', '', spec_id)) if spec_id else 0
        self.logger = get_logger(model_name=f"{self.model_name}", spec_id=numeric_id, test_id=0, type_name="equiv")
        
        # Read both spec files
        human_spec_path = os.path.join(self.human_input_dir, f"{spec_id}.v")
        llm_spec_path = os.path.join(self.llm_input_dir, f"{spec_id}.v")
        
        try:
            human_spec = self.read_spec(human_spec_path)
            llm_spec = self.read_spec(llm_spec_path)
        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Failed to read spec files: {e}")
            print(f"Error: Failed to read spec files: {e}", file=sys.stderr)
            return False, False
        
        # Generate l direction proof (human -> llm)
        l_proof_success = False
        l_output_path = os.path.join(self.equiv_output_dir, f"{spec_id}_l.v")
        
        if self.logger:
            self.logger.info(f"Generating proof for {spec_id}_l.v (human -> llm)")
        
        error_info_l = ""
        for attempt in range(1, max_attempts + 1):
            try:
                proof_l = self.generate_implication_proof(human_spec, llm_spec, "l", error_info_l)
                
                # Verify proof (generated proof should be complete and standalone)
                is_valid, error_info = self.verify_coq_proof(proof_l)
                
                if is_valid:
                    # Save successful proof
                    with open(l_output_path, 'w', encoding='utf-8') as f:
                        f.write(proof_l)
                    l_proof_success = True
                    if self.logger:
                        self.logger.info(f"Successfully generated and verified proof for {spec_id}_l.v")
                    break
                else:
                    error_info_l = error_info
                    if self.logger:
                        self.logger.warning(f"Attempt {attempt}/{max_attempts} failed for {spec_id}_l.v: {error_info}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error generating proof for {spec_id}_l.v: {e}")
                print(f"Error: Error generating proof for {spec_id}_l.v: {e}", file=sys.stderr)
        
        if not l_proof_success:
            if self.logger:
                self.logger.error(f"Failed to generate valid proof for {spec_id}_l.v after {max_attempts} attempts")
            print(f"Warning: Failed to generate valid proof for {spec_id}_l.v", file=sys.stderr)
        
        # Generate r direction proof (llm -> human)
        r_proof_success = False
        r_output_path = os.path.join(self.equiv_output_dir, f"{spec_id}_r.v")
        
        if self.logger:
            self.logger.info(f"Generating proof for {spec_id}_r.v (llm -> human)")
        
        error_info_r = ""
        for attempt in range(1, max_attempts + 1):
            try:
                proof_r = self.generate_implication_proof(llm_spec, human_spec, "r", error_info_r)
                
                # Verify proof (generated proof should be complete and standalone)
                is_valid, error_info = self.verify_coq_proof(proof_r)
                
                if is_valid:
                    # Save successful proof
                    with open(r_output_path, 'w', encoding='utf-8') as f:
                        f.write(proof_r)
                    r_proof_success = True
                    if self.logger:
                        self.logger.info(f"Successfully generated and verified proof for {spec_id}_r.v")
                    break
                else:
                    error_info_r = error_info
                    if self.logger:
                        self.logger.warning(f"Attempt {attempt}/{max_attempts} failed for {spec_id}_r.v: {error_info}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error generating proof for {spec_id}_r.v: {e}")
                print(f"Error: Error generating proof for {spec_id}_r.v: {e}", file=sys.stderr)
        
        if not r_proof_success:
            if self.logger:
                self.logger.error(f"Failed to generate valid proof for {spec_id}_r.v after {max_attempts} attempts")
            print(f"Warning: Failed to generate valid proof for {spec_id}_r.v", file=sys.stderr)
        
        return l_proof_success, r_proof_success


def main():
    parser = argparse.ArgumentParser(
        description='Generate implication proofs between two specs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --spec 1                    # Generate proof for spec 1
  %(prog)s --spec 122 --model gpt-4o   # Generate proof for spec 122, using gpt-4o model
  %(prog)s --all                       # Generate proofs for all available specs
        """
    )
    parser.add_argument(
        '--spec',
        type=str,
        default=None,
        metavar='SPEC_ID',
        help='Specify spec_id to generate proof for (e.g., "1", "122")'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=_MODEL_NAME,
        metavar='MODEL_NAME',
        help=f'Model name (default: {_MODEL_NAME})'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate proofs for all available specs'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        metavar='N',
        help='Maximum number of attempts per direction (default: 3)'
    )
    
    args = parser.parse_args()
    
    if not args.spec and not args.all:
        parser.error("Must specify --spec or --all")
    
    generator = EquivProofGenerator(model_name=args.model)
    
    if args.all:
        # Get all available spec_ids
        human_specs = set()
        llm_specs = set()
        
        if os.path.exists(generator.human_input_dir):
            for f in os.listdir(generator.human_input_dir):
                if f.endswith('.v') and f != 'lib.v':
                    spec_id = f[:-2]  # Remove .v suffix
                    human_specs.add(spec_id)
        
        if os.path.exists(generator.llm_input_dir):
            for f in os.listdir(generator.llm_input_dir):
                if f.endswith('.v'):
                    spec_id = f[:-2]  # Remove .v suffix
                    llm_specs.add(spec_id)
        
        # Find spec_ids that exist in both directories
        common_specs = sorted(human_specs & llm_specs, key=lambda x: int(re.sub(r'[^0-9]', '', x)) if re.sub(r'[^0-9]', '', x) else 0)
        
        print(f"Found {len(common_specs)} common specs, starting proof generation...", file=sys.stderr)
        
        success_count = 0
        l_success_count = 0
        r_success_count = 0
        
        for spec_id in common_specs:
            print(f"Processing spec {spec_id}...", file=sys.stderr)
            l_success, r_success = generator.generate_proofs_for_spec(spec_id, max_attempts=args.max_attempts)
            
            if l_success:
                l_success_count += 1
            if r_success:
                r_success_count += 1
            if l_success and r_success:
                success_count += 1
        
        print(f"\nComplete!", file=sys.stderr)
        print(f"  Total specs: {len(common_specs)}", file=sys.stderr)
        print(f"  Both directions succeeded: {success_count}", file=sys.stderr)
        print(f"  l direction succeeded (human -> llm): {l_success_count}", file=sys.stderr)
        print(f"  r direction succeeded (llm -> human): {r_success_count}", file=sys.stderr)
    else:
        # Process single spec
        l_success, r_success = generator.generate_proofs_for_spec(args.spec, max_attempts=args.max_attempts)
        
        if l_success:
            print(f"✓ Successfully generated {args.spec}_l.v (human -> llm)")
        else:
            print(f"✗ Failed to generate {args.spec}_l.v (human -> llm)")
        
        if r_success:
            print(f"✓ Successfully generated {args.spec}_r.v (llm -> human)")
        else:
            print(f"✗ Failed to generate {args.spec}_r.v (llm -> human)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
