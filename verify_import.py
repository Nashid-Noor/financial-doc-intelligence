import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Attempting to import src.api.app...")
try:
    from src.api.app import app
    print("SUCCESS: src.api.app imported successfully.")
except Exception as e:
    print(f"FAILURE: Could not import src.api.app. Error: {e}")
    import traceback
    traceback.print_exc()
