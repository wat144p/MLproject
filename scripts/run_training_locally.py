import sys
import os
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from flows.training_flow import training_flow

if __name__ == "__main__":
    print("Starting local training run...")
    training_flow()
