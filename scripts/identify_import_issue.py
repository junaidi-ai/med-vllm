#!/usr/bin/env python3
"""Script to help identify which test file is causing the import issue."""

import argparse
import importlib
import os
import sys
import traceback


def test_import_file(file_path: str) -> bool:
    """Test importing a single test file and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Testing import of: {file_path}")
    print(f"{'='*80}")

    # Convert file path to module path
    rel_path = os.path.relpath(file_path, os.getcwd())
    module_path = rel_path.replace("/", ".").replace("\\", ".").replace(".py", "")

    try:
        # Add the directory containing the test file to Python path
        test_dir = os.path.dirname(os.path.abspath(file_path))
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)

        # Try to import the module
        module = importlib.import_module(module_path)
        print(f"✅ Successfully imported {module_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_path}")
        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False


def main():
    """Main function to run the import tests."""
    parser = argparse.ArgumentParser(description="Test imports in test files.")
    parser.add_argument("test_files", nargs="*", help="Test files to check")
    args = parser.parse_args()

    # If no files provided, find all test files in the tests directory
    if not args.test_files:
        test_dir = os.path.join(os.getcwd(), "tests")
        test_files = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
    else:
        test_files = [os.path.abspath(f) for f in args.test_files]

    print(f"Found {len(test_files)} test files to check...")

    # Test each file
    success_count = 0
    failed_files = []

    for test_file in test_files:
        if test_import_file(test_file):
            success_count += 1
        else:
            failed_files.append(test_file)

    # Print summary
    print("\n" + "=" * 80)
    print("Import Test Summary:")
    print("=" * 80)
    print(f"Total files tested: {len(test_files)}")
    print(f"✅ Successfully imported: {success_count}")
    print(f"❌ Failed to import: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    return 0 if not failed_files else 1


if __name__ == "__main__":
    sys.exit(main())
