#!/usr/bin/env python3
"""Script to clean up the root directory by organizing and removing files."""

import os
import shutil
from pathlib import Path

def main():
    root = Path('.')
    
    # Create necessary directories
    Path('scripts').mkdir(exist_ok=True)
    Path('tests/temp').mkdir(parents=True, exist_ok=True)
    
    # Files to keep in root
    keep_files = {
        'setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt',
        'README.md', 'LICENSE', '.gitignore', 'MANIFEST.in', 'Makefile'
    }
    
    # Files to move to scripts/
    script_files = {
        'run_all_tests.sh', 'run_tests.sh', 'run_benchmarks.sh',
        'run_tests_directly.py', 'run_tests_with_logging.py'
    }
    
    # Files to delete (redundant or temporary test files)
    delete_files = {
        'test_import.py', 'test_imports.py', 'test_hello.py', 'test_minimal.py',
        'test_simple.py', 'test_env.py', 'test_python_path.py',
        'test_direct_import.py', 'test_import_issues.py',
        'test_minimal_imports.py', 'test_patched_import.py',
        'test_subprocess.py', 'test_transformers.py',
        'test_transformers_import.py', 'test_transformers_import_script.py',
        'test_biobert_import.py', 'test_biobert_imports.py',
        'test_biobert_init.py', 'test_biobert_minimal.py',
        'test_biobert_simple.py', 'test_biobert_verbose.py',
        'test_biobert_direct.py', 'test_biobert_directly.py',
        'test_biobert_detailed.py', 'test_biobert_manual.py',
        'test_llm_engine_import.py'
    }
    
    # Move script files to scripts/
    for filename in script_files:
        if (root / filename).exists():
            shutil.move(str(filename), f'scripts/{filename}')
            print(f"Moved {filename} to scripts/")
    
    # Move test files to tests/temp/
    for filename in delete_files:
        if (root / filename).exists():
            shutil.move(str(filename), f'tests/temp/{filename}')
            print(f"Moved {filename} to tests/temp/")
    
    # List remaining files
    print("\nRemaining files in root directory:")
    for f in sorted(root.iterdir()):
        if f.is_file() and f.name not in keep_files and f.name not in script_files:
            print(f"  - {f.name} (consider moving or deleting)")
    
    print("\nCleanup complete. Review the moved files in scripts/ and tests/temp/")
    print("Run 'rm -rf tests/temp' to delete the temporary test files.")

if __name__ == '__main__':
    main()
