#!/usr/bin/env python3
"""Script to update import paths in test files after moving them to the top-level tests directory."""

import os
import re
import sys
from pathlib import Path

def update_imports(file_path):
    """Update import paths in a single file."""
    # Skip non-Python files
    if not file_path.endswith('.py') or file_path.endswith('__init__.py'):
        return False
    
    print(f"Checking {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track if we made any changes
    updated_content = content
    changes_made = False
    
    # Update imports that reference the old scripts.tests path
    updated_content, n = re.subn(
        r'from\s+scripts\.tests\.', 
        'from tests.', 
        updated_content
    )
    changes_made = changes_made or (n > 0)
    
    # Update relative imports within the tests directory
    updated_content, n = re.subn(
        r'from\s+\.(\..+?)\s+import', 
        r'from tests\1 import', 
        updated_content
    )
    changes_made = changes_made or (n > 0)
    
    # Update sys.path modifications if they reference the old location
    updated_content, n = re.subn(
        r'sys\.path\.insert\s*\(\s*0\s*,\s*[\'"].*?scripts/tests[\'"]\)',
        "# sys.path.insert(0, str(Path(__file__).parent.parent))  # Updated path",
        updated_content
    )
    changes_made = changes_made or (n > 0)
    
    # Only write if changes were made
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"  Updated {file_path}")
        return True
    
    return False

def update_conftest_imports(file_path):
    """Special handling for conftest.py files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update sys.path.insert to point to the new location
    updated_content = re.sub(
        r'sys\.path\.insert\s*\(\s*0\s*,\s*[\'"].*?scripts/tests[\'"]\)',
        "sys.path.insert(0, str(Path(__file__).parent.parent))  # Updated path",
        content
    )
    
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"  Updated {file_path}")
        return True
    
    return False

def main():
    """Main function to update imports in all test files."""
    tests_dir = Path('tests')
    updated_files = []
    
    # First, handle conftest.py files specially
    for conftest_path in tests_dir.rglob('conftest.py'):
        if update_conftest_imports(conftest_path):
            updated_files.append(str(conftest_path))
    
    # Then handle all other Python files
    for py_file in tests_dir.rglob('*.py'):
        if py_file.name != 'conftest.py' and py_file.name != '__init__.py':
            if update_imports(str(py_file)):
                updated_files.append(str(py_file))
    
    print(f"\nUpdated imports in {len(updated_files)} files:")
    for file in sorted(updated_files):
        print(f"- {file}")

if __name__ == '__main__':
    main()
