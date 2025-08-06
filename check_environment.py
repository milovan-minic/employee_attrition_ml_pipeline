"""
Pre-flight Environment Check Script for ML Project
Ensures all requirements are met before running the main pipeline.
"""

import os
import sys
import importlib.util
import shutil
import platform
import psutil

REQUIRED_FILES = [
    'data/in/train.csv',
    'data/in/test.csv'
]
REQUIRED_DIRS = [
    'data/out',
    'results',
    'models'
]
REQUIRED_PACKAGES = [
    # pip name           # import name
    ('pandas',           'pandas'),
    ('numpy',            'numpy'),
    ('scikit-learn',     'sklearn'),
    ('matplotlib',       'matplotlib'),
    ('seaborn',          'seaborn'),
    ('xgboost',          'xgboost'),
    ('lightgbm',         'lightgbm'),
    ('catboost',         'catboost'),
    ('imbalanced-learn', 'imblearn'),
    ('joblib',           'joblib'),
    ('scipy',            'scipy')
]
MIN_PYTHON = (3, 8)
MIN_RAM_GB = 4
MIN_DISK_GB = 2


def check_python_version():
    print(f"Python version: {platform.python_version()}")
    if sys.version_info < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
        return False
    print("✅ Python version is sufficient.")
    return True

def check_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    print("✅ All required data files are present.")
    return True

def check_dirs():
    for d in REQUIRED_DIRS:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"🛠️ Created missing directory: {d}")
    print("✅ All required directories are present or created.")
    return True

def check_packages():
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    if missing:
        print(f"❌ Missing required Python packages: {missing}")
        print("   Please run: pip install -r requirements.txt")
        return False
    print("✅ All required Python packages are installed.")
    return True

def check_ram():
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"System RAM: {ram_gb:.1f} GB")
    if ram_gb < MIN_RAM_GB:
        print(f"❌ At least {MIN_RAM_GB} GB RAM required.")
        return False
    print("✅ Sufficient RAM available.")
    return True

def check_disk():
    disk = shutil.disk_usage('.')
    free_gb = disk.free / (1024 ** 3)
    print(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < MIN_DISK_GB:
        print(f"❌ At least {MIN_DISK_GB} GB free disk space required.")
        return False
    print("✅ Sufficient disk space available.")
    return True

def check_write_permissions():
    try:
        for d in REQUIRED_DIRS:
            testfile = os.path.join(d, 'test_write.tmp')
            with open(testfile, 'w') as f:
                f.write('test')
            os.remove(testfile)
        print("✅ Write permissions for output directories confirmed.")
        return True
    except Exception as e:
        print(f"❌ Write permission error: {e}")
        return False

def check_virtualenv():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected.")
        return True
    else:
        print("⚠️  Not running inside a virtual environment. (Recommended)")
        return True  # Not fatal, just a warning

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  No GPU detected. (Not required, but will speed up training)")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping GPU check.")
        return True

def check_env_vars():
    # Example: check for custom environment variables if needed
    # Not required for this project, but placeholder for best practices
    return True

if __name__ == '__main__':
    print("🔎 Running pre-flight environment check...")
    ok = (
        check_python_version() and
        check_files() and
        check_dirs() and
        check_packages() and
        check_ram() and
        check_disk() and
        check_write_permissions() and
        check_virtualenv() and
        check_gpu() and
        check_env_vars()
    )
    if ok:
        print("\n🚦 Environment check PASSED. You are ready to run main.py!")
        sys.exit(0)
    else:
        print("\n🚨 Environment check FAILED. Please fix the above issues before running main.py.")
        sys.exit(1)