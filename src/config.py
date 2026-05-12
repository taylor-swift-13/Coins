from dataclasses import dataclass
from pathlib import Path

# Global proof/spec settings
# TYPE only determines the input directory path, no other logic differences
TYPE = "human"
MODEL_NAME = "deepseek-v4-pro"
INPUT_MODEL_NAME = MODEL_NAME

# Paths based on TYPE (TYPE only affects input directory)
# Use relative paths based on project root (parent of src/)
BASE_DIR = Path(__file__).parent.parent.resolve()
DATASET_PATH = str(BASE_DIR / "HumanEvalPlus.jsonl")
SPEC_INPUT_DIR = str(BASE_DIR / "spec" / TYPE / "input")
SPEC_OUTPUT_DIR = str(BASE_DIR / "spec" / TYPE / "output")
ROOT_DIR = str(BASE_DIR / "spec" / TYPE)

# Proof generation settings
MAX_ITERATIONS = 3  # Maximum iterations for proof generation (test case 0)
SKIP_ON_FIRST_FAIL = True  # For batch_test: stop at first failed test case
MAX_TEST_CASES = 0  # Maximum number of test cases to process (2-MAX_TEST_CASES, default: 0 means all)


@dataclass
class LLMConfig:
    """LLM API configuration"""
    use_api_model: bool = True  # Control whether to use API model or local Transformers model
    api_model: str = "deepseek-v4-pro"  # Default API model
    api_key: str = "sk-9dnxPHHwjlTiAlH0uUoKwiW79Hs51AGpqUSRj9DvZurLZx1R"
    base_url: str = "https://yunwu.ai/v1"
    api_temperature: float = 0.7  # Temperature parameter for API calls
