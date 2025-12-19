import sys
import os
from pathlib import Path

# Add the project root to the python path so that 'app' and 'ml' can be imported
# Get the directory containing this file (tests/)
current_dir = Path(__file__).parent
# Get the project root (one level up)
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


