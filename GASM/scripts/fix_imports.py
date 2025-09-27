#!/usr/bin/env python3
"""
Fix import paths after project restructuring
"""

import os
import re
import sys

def fix_imports_in_file(file_path):
    """Fix import statements in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import patterns
        fixes = [
            # src.spatial_agent -> spatial_agent when running from src/
            (r'from src\.spatial_agent\.', 'from spatial_agent.'),
            (r'import src\.spatial_agent\.', 'import spatial_agent.'),
            
            # src.gasm -> gasm when running from src/  
            (r'from src\.gasm\.', 'from gasm.'),
            (r'import src\.gasm\.', 'import gasm.'),
            
            # src.api -> api when running from src/
            (r'from src\.api\.', 'from api.'),
            (r'import src\.api\.', 'import api.'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        # Add sys.path modifications at the top for test files
        if 'test_' in os.path.basename(file_path) and '/tests/' in file_path:
            if 'sys.path' not in content:
                # Add path setup for tests
                path_setup = '''import sys
import os
# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

'''
                # Insert after existing imports but before test imports
                lines = content.split('\n')
                import_end_idx = 0
                for i, line in enumerate(lines):
                    if (line.startswith('import ') or line.startswith('from ')) and 'src.' not in line:
                        import_end_idx = i + 1
                
                lines.insert(import_end_idx, path_setup.strip())
                content = '\n'.join(lines)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⏭️  No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def find_python_files(root_dir):
    """Find all Python files that might need fixing"""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', 'dist', 'build'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    print("🔧 Fixing import paths after project restructuring...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"📁 Project root: {project_root}")
    
    python_files = find_python_files(project_root)
    print(f"📄 Found {len(python_files)} Python files")
    
    fixed_count = 0
    
    for file_path in python_files:
        # Skip certain files
        skip_patterns = [
            'scripts/fix_imports.py',  # Don't fix this script itself
            'venv/',
            '.venv/',
            '__pycache__/',
        ]
        
        if any(pattern in file_path for pattern in skip_patterns):
            continue
            
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\n🎉 Fixed imports in {fixed_count} files!")
    print("\n🧪 Test the fixes:")
    print("python tests/test_se3_utils.py")
    print("python scripts/run_simple_demo.py 'box above robot'")

if __name__ == "__main__":
    main()