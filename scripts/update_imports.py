"""Script to update import paths from medvllm to medvllm."""

import os
import re
from pathlib import Path

# Directory to search for Python files
ROOT_DIR = Path(__file__).parent

# Files to exclude from processing
EXCLUDED_FILES = {
    # Add any files that should be excluded from processing
}

def update_imports_in_file(file_path: Path) -> bool:
    """Update import statements in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace import statements
        updated_content = re.sub(
            r'from\s+nanovllm(\.|\s+)', 
            r'from medvllm\1', 
            content
        )
        
        # Only write if changes were made
        if updated_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to update imports in all Python files."""
    updated_count = 0
    
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if not file.endswith('.py') or file in EXCLUDED_FILES:
                continue
                
            file_path = Path(root) / file
            relative_path = file_path.relative_to(ROOT_DIR)
            
            if update_imports_in_file(file_path):
                print(f"Updated imports in: {relative_path}")
                updated_count += 1
    
    print(f"\nUpdated imports in {updated_count} files.")

if __name__ == "__main__":
    main()
