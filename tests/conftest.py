# tests/conftest.py
"""
Pytest configuration and shared fixtures for HKB-BV test suite.
Resolves module import paths for backend, evaluation, and models.
"""

import sys
import os

# ============================================================
# Path Resolution
# Add project root to sys.path so all modules are importable
# ============================================================

# Get absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path if not already present
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Verify critical modules are importable
def pytest_configure(config):
    """Called after command line options have been parsed."""
    print(f"\n[conftest] Project root: {PROJECT_ROOT}")
    print(f"[conftest] sys.path[0]: {sys.path[0]}")
