
import sys
import traceback

print(f"Python Executable: {sys.executable}")
print("Attempting to import deepchecks...")

try:
    import deepchecks
    print(f"DeepChecks Version: {deepchecks.__version__}")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import train_test_validation
    print("SUCCESS: DeepChecks imported correctly!")
except ImportError as e:
    print("\nFAILURE: ImportError caught!")
    print(e)
    traceback.print_exc()
except Exception as e:
    print(f"\nFAILURE: Unexpected error: {e}")
    traceback.print_exc()


