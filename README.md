
# Coins

![Coins Overview](images/main.png)

Coins is a Coq-based formal verification framework for evaluating HumanEval code specifications. It uses LLMs to automatically generate Coq proofs and verify the correctness and robustness of specifications.

## Project Structure

```
coins/
├── src/                          # Core modules
│   ├── __init__.py               # Module initialization
│   ├── config.py                 # Configuration (models, paths, parameters)
│   ├── llm.py                    # LLM interface wrapper
│   ├── logger.py                 # Logging utility
│   ├── extract_pairs.py          # Extract test cases from HumanEval dataset
│   ├── verify.py                 # Coq verification tool
│   ├── proof_gen.py              # Single specification proof generation
│   ├── batch_proof_gen.py        # Batch proof generation
│   ├── spec_gen.py               # Coq specification generation from HumanEval
│   ├── tester.py                 # Test case verifier
│   ├── negative_proof.py         # Single specification negative testing
│   ├── negative_batch_proof.py   # Batch negative testing
│   ├── equiv_proof_gen.py        # Specification implication proof generation
│   └── equiv_proof_batch.py      # Batch implication proof generation
├── spec/                         # Specifications, proofs, tests, and model outputs
│   ├── human/
│   │   ├── input/                # Human-written input specs
│   │   ├── output/               # Proofs for human specs
│   │   ├── test/                 # Additional test proofs
│   │   ├── bad/                  # Failed proofs
│   │   └── pass_all/             # Specs that passed all tests
│   ├── gpt-4o/, gpt-5/, gemini-3-pro-preview/, claude-opus-4-5-20251101/, ...
│   │   ├── input/
│   │   ├── output/
│   │   ├── test/
│   │   ├── bad/
│   │   └── pass_all/
├── equiv/                        # Equivalence/implication proofs
│   ├── input/                    # Input specs (human/, llm/)
│   └── output/                   # Generated implication proofs
├── negative/                     # Negative cases testing
│   ├── input/                    # Input by model type
│   └── negative_cases.jsonl      # Negative test cases
├── images/                       # Documentation images
├── log/                          # Log output directory
└── README.md
```

## Requirements

- Python 3.8+
- Coq (coqtop)
- OpenAI-compatible LLM API

### Python Dependencies

```bash
pip install openai
```

## Configuration

Configuration file is located at `src/config.py`:

```python
TYPE = "human"
MODEL_NAME = "gpt-4o"
DATASET_PATH = "<repo-root>/HumanEvalPlus.jsonl"

@dataclass
class LLMConfig:
    use_api_model: bool = True
    api_model: str = "gemini-3-pro-preview"
    api_key: str = "<api-key>"
    base_url: str = "<base-url>"
    api_temperature: float = 0.7
```

## Core Modules

### 1. Proof Generation (proof_gen.py)

Generates Coq proofs for a single specification using LLM with iterative error fixing.

```bash
# Generate proof for specification 1
python3 src/proof_gen.py 1

# Use underscore format with custom max attempts
python3 src/proof_gen.py 1_ --max-attempts 5

# Output to file
python3 src/proof_gen.py 42 --output proof.v
```

### 2. Spec Generation (spec_gen.py)

Generates Coq specifications from HumanEval-style JSONL tasks.

```bash
# Generate a single spec with reference implementation in the prompt
python3 src/spec_gen.py --model gpt-4o --type llm --spec 0

# Generate a single spec without reference implementation in the prompt
python3 src/spec_gen.py --model gpt-4o --type llm --spec 0 --without-reference-impl
```

### 3. Batch Proof Generation (batch_proof_gen.py)

Parallel processing of multiple specification proofs.

```bash
# Process specifications 1-10
python3 src/batch_proof_gen.py --range 1:10

# Specify number of parallel workers
python3 src/batch_proof_gen.py --range 1:100 --num-workers 16

# Specify max attempts
python3 src/batch_proof_gen.py --range 1:50 --max-attempts 5
```

### 4. Test Verification (tester.py)

Current script entry prints the number of available test cases for a specification.

```bash
# Check test case count for specification
python3 src/tester.py 1

# Underscore variant
python3 src/tester.py 1_
```

### 5. Coq Verification (verify.py)

Verifies the correctness of Coq files.

```bash
# Verify single file
python3 src/verify.py --file spec/human/input/1.v

# Batch verify files in directory
python3 src/verify.py --batch --directory spec/gpt-4o/output/gpt-4o --range 1:10
```

### 6. Negative Testing (negative_proof.py / negative_batch_proof.py)

Tests specification robustness against negative test cases. If all negative proofs fail, the specification is considered robust.

```bash
# Single specification negative test
python3 src/negative_proof.py --spec-id 42

# Batch negative testing
python3 src/negative_batch_proof.py --range 1:10 --num-workers 8
```

### 7. Implication Proofs (equiv_proof_gen.py / equiv_proof_batch.py)

Generates implication proofs between two specifications, used to compare human-written and LLM-generated specifications.

```bash
# Generate implication proof for single spec
python3 src/equiv_proof_gen.py --spec 1

# Specify model to use
python3 src/equiv_proof_gen.py --spec 122 --model gpt-4o

# Generate proofs for all available specs
python3 src/equiv_proof_gen.py --all

# Batch processing
python3 src/equiv_proof_batch.py --model gemini-3-pro-preview
```

### 8. Test Case Extraction (extract_pairs.py)

Extracts input-output pairs from HumanEval dataset.

```bash
# Extract test pairs for specification 0
python3 src/extract_pairs.py -i 0 -p /path/to/HumanEvalPlus.jsonl

# Use 1-based indexing
python3 src/extract_pairs.py -i 1 --one-based

# Extract results only
python3 src/extract_pairs.py -i 0 --mode results
```

## Workflows

### 1. Basic Proof Generation Workflow

```
Input specification (spec/input/*.v)
        ↓
  LLM generates proof
        ↓
  Coq verification (coqtop)
        ↓
    Failed? → Extract error info → Iterative fix
        ↓
    Success → Save to output
```

### 2. Negative Testing Workflow

```
Read negative test cases (negative/negative_cases.jsonl)
        ↓
Generate proof for each negative case
        ↓
All proofs failed? → GOOD (spec is robust)
        ↓
Any proof succeeded? → BAD (spec too weak)
```

### 3. Implication Verification Workflow

```
Human spec (equiv/input/human/*.v)
LLM spec (equiv/input/llm/*.v)
        ↓
Generate human → llm proof (*_l.v)
Generate llm → human proof (*_r.v)
        ↓
Output to equiv/output/
```

## Directory Conventions

| Directory | Purpose |
|-----------|---------|
| `spec/{model}/input/` | Generated or human-written input specs |
| `spec/{model}/output/` | Proofs generated from the input specs |
| `spec/{model}/test/` | Proofs for additional test cases |
| `spec/{model}/bad/` | Failed proofs |
| `spec/{model}/pass_all/` | Specs that passed all tests |
| `negative/input/{model}/` | Negative testing input |
| `negative/negative_cases.jsonl` | Negative test cases data |
| `equiv/input/human/` | Human-written specifications for equivalence |
| `equiv/input/llm/` | LLM-generated specifications for equivalence |
| `equiv/output/` | Implication proofs |
| `log/` | Runtime logs |

## File Naming Conventions

- `{id}.v` - Standard specification file (e.g., `1.v`, `42.v`)
- `{id}_.v` - Inductive specification (e.g., `1_.v`, `137_.v`)
- `{id}_l.v` - Left implication proof (human → llm)
- `{id}_r.v` - Right implication proof (llm → human)
- `{id}_{n}.v` - Negative test proof (nth negative case)

## FAQ

### 1. How to modify LLM API configuration?

Edit the `LLMConfig` class in `src/config.py`:

```python
@dataclass
class LLMConfig:
    api_model: str = "your-model-name"
    api_key: str = "your-api-key"
    base_url: str = "https://your-api-url/v1"
```

### 2. How to increase parallelism?

Use the `--num-workers` parameter:

```bash
python3 src/batch_proof_gen.py --range 1:100 --num-workers 32
```

### 3. What if proof verification times out?

Default timeout is 30 seconds. You can adjust the `timeout` parameter in the source code.

## License

MIT License
