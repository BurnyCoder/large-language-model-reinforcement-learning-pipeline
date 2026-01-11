"""Configure HuggingFace cache directory to project folder."""
import os
from pathlib import Path

# Set cache to ./cache/ relative to project root
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"

# Set HF_HOME before any HuggingFace imports
os.environ["HF_HOME"] = str(CACHE_DIR)
