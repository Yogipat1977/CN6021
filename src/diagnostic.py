import os
import sys

def check_imports():
    print("Checking imports...")
    try:
        import monai
        print(f"  monai version: {monai.__version__}")
    except ImportError as e:
        print(f"  FAILED to import monai: {e}")

    try:
        import nibabel
        print(f"  nibabel version: {nibabel.version.version}")
    except ImportError as e:
        print(f"  FAILED to import nibabel: {e}")

    try:
        import kagglehub
        print("  kagglehub imported successfully")
    except ImportError as e:
        print(f"  FAILED to import kagglehub: {e}")

    try:
        import einops
        print("  einops imported successfully")
    except ImportError as e:
        print(f"  FAILED to import einops: {e}")

def check_paths():
    print("\nChecking paths...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"  Current file dir: {current_dir}")
    eda_path = os.path.join(current_dir, "eda.py")
    print(f"  eda.py exists: {os.path.exists(eda_path)}")

if __name__ == "__main__":
    check_imports()
    check_paths()
