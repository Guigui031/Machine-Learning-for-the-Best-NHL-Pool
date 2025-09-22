#!/usr/bin/env python3
"""
Analyze actual dependencies in notebooks and Python files
"""

import os
import re
import json

def extract_imports_from_notebook(notebook_path):
    """Extract import statements from Jupyter notebook"""
    imports = set()

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find import statements in code cells
        import_patterns = [
            r'import (\w+)',
            r'from (\w+) import',
            r'from \.(\w+) import',
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)

    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")

    return imports

def extract_imports_from_python(python_path):
    """Extract import statements from Python file"""
    imports = set()

    try:
        with open(python_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Extract module names
                if line.startswith('import '):
                    module = line.replace('import ', '').split()[0].split('.')[0]
                    imports.add(module)
                elif line.startswith('from ') and ' import ' in line:
                    module = line.split(' import ')[0].replace('from ', '').split('.')[0]
                    imports.add(module)

    except Exception as e:
        print(f"Error reading {python_path}: {e}")

    return imports

def analyze_project_dependencies():
    """Analyze all dependencies in the project"""

    # Get all notebooks and Python files
    notebooks = [f for f in os.listdir('.') if f.endswith('.ipynb')]
    python_files = [f for f in os.listdir('.') if f.endswith('.py')]

    dependency_map = {
        'notebooks': {},
        'python_files': {},
        'local_modules': set(),
        'external_packages': set()
    }

    # Analyze notebooks
    for notebook in notebooks:
        print(f"Analyzing notebook: {notebook}")
        imports = extract_imports_from_notebook(notebook)
        dependency_map['notebooks'][notebook] = list(imports)

    # Analyze Python files
    for py_file in python_files:
        if py_file != 'analyze_dependencies.py':  # Skip this script
            print(f"Analyzing Python file: {py_file}")
            imports = extract_imports_from_python(py_file)
            dependency_map['python_files'][py_file] = list(imports)

    # Categorize imports
    local_py_files = {f.replace('.py', '') for f in python_files}

    all_imports = set()
    for imports_list in dependency_map['notebooks'].values():
        all_imports.update(imports_list)
    for imports_list in dependency_map['python_files'].values():
        all_imports.update(imports_list)

    for imp in all_imports:
        if imp in local_py_files or imp in ['ml_models', 'src']:
            dependency_map['local_modules'].add(imp)
        else:
            dependency_map['external_packages'].add(imp)

    dependency_map['local_modules'] = list(dependency_map['local_modules'])
    dependency_map['external_packages'] = list(dependency_map['external_packages'])

    return dependency_map

def find_unused_files(dependency_map):
    """Find files that are not imported by any notebook or key Python file"""

    # Key entry points (notebooks and main scripts)
    key_files = list(dependency_map['notebooks'].keys())

    # Get all imports from key entry points
    used_modules = set()
    for notebook_imports in dependency_map['notebooks'].values():
        used_modules.update(notebook_imports)

    # Python files in root directory
    python_files = [f.replace('.py', '') for f in os.listdir('.') if f.endswith('.py')]

    # Find unused files
    unused_files = []
    for py_file in python_files:
        if py_file not in used_modules and py_file != 'analyze_dependencies':
            # Check if it's used by any other local module
            is_used = False
            for other_file, imports in dependency_map['python_files'].items():
                if py_file in imports:
                    is_used = True
                    break

            if not is_used:
                unused_files.append(f"{py_file}.py")

    return unused_files

def main():
    print("=" * 60)
    print("NHL Pool Optimization - Dependency Analysis")
    print("=" * 60)

    # Analyze dependencies
    dependency_map = analyze_project_dependencies()

    print("\n📊 DEPENDENCY SUMMARY:")
    print(f"Notebooks analyzed: {len(dependency_map['notebooks'])}")
    print(f"Python files analyzed: {len(dependency_map['python_files'])}")
    print(f"Local modules found: {len(dependency_map['local_modules'])}")
    print(f"External packages: {len(dependency_map['external_packages'])}")

    print("\n📋 NOTEBOOK DEPENDENCIES:")
    for notebook, imports in dependency_map['notebooks'].items():
        local_imports = [imp for imp in imports if imp in dependency_map['local_modules']]
        print(f"{notebook}:")
        print(f"  Local: {local_imports}")

    print("\n🔍 LOCAL MODULE USAGE:")
    for module in sorted(dependency_map['local_modules']):
        used_by = []
        for notebook, imports in dependency_map['notebooks'].items():
            if module in imports:
                used_by.append(notebook)
        for py_file, imports in dependency_map['python_files'].items():
            if module in imports:
                used_by.append(py_file)

        if used_by:
            print(f"{module}.py: Used by {used_by}")
        else:
            print(f"{module}.py: ⚠️ Not directly used by notebooks")

    # Find potentially unused files
    unused_files = find_unused_files(dependency_map)

    print("\n🗑️ POTENTIALLY UNUSED FILES:")
    if unused_files:
        for file in unused_files:
            print(f"  - {file}")
    else:
        print("  None found - all files appear to be used")

    print("\n💾 Saving detailed analysis...")
    with open('dependency_analysis.json', 'w') as f:
        json.dump(dependency_map, f, indent=2)

    print("✅ Analysis complete! See dependency_analysis.json for details")

if __name__ == "__main__":
    main()