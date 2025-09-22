#!/usr/bin/env python3
"""
Simple dependency analysis for refactoring
"""

import os
import re

def check_notebook_imports(notebook_path):
    """Check what local modules a notebook imports"""
    local_imports = []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Local module patterns
        local_modules = [
            'config', 'data_download', 'data_pipeline', 'data_validation',
            'ensemble_learning', 'model_predictor', 'player', 'pool_classifier',
            'process_data', 'team', 'ml_models'
        ]

        for module in local_modules:
            patterns = [
                f'import {module}',
                f'from {module} import',
                f'from {module}.'
            ]
            for pattern in patterns:
                if pattern in content:
                    if module not in local_imports:
                        local_imports.append(module)

    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")

    return local_imports

def main():
    print("=" * 50)
    print("Dependency Analysis for Notebooks")
    print("=" * 50)

    notebooks = [
        'ML_for_NHL.ipynb',
        'NHL_ML_Models_Complete.ipynb',
        'Team_Optimization_Notebook.ipynb'
    ]

    all_used_modules = set()

    for notebook in notebooks:
        if os.path.exists(notebook):
            imports = check_notebook_imports(notebook)
            all_used_modules.update(imports)
            print(f"{notebook}:")
            print(f"  Uses: {imports}")
        else:
            print(f"{notebook}: NOT FOUND")

    print(f"\nAll modules used by notebooks: {sorted(all_used_modules)}")

    # Check which files exist
    all_py_files = [f.replace('.py', '') for f in os.listdir('.') if f.endswith('.py')]
    print(f"\nAll Python files in root: {sorted(all_py_files)}")

    # Find potentially unused
    unused = set(all_py_files) - all_used_modules - {'analyze_dependencies', 'simple_dependency_check'}
    print(f"\nPotentially unused by notebooks: {sorted(unused)}")

    # Check if unused files are imported by other modules
    print(f"\nChecking cross-dependencies...")
    for unused_file in unused:
        used_by = []
        for py_file in all_py_files:
            if py_file != unused_file and py_file not in ['analyze_dependencies', 'simple_dependency_check']:
                try:
                    with open(f'{py_file}.py', 'r', encoding='utf-8') as f:
                        content = f.read()
                        if f'import {unused_file}' in content or f'from {unused_file}' in content:
                            used_by.append(py_file)
                except:
                    pass

        if used_by:
            print(f"  {unused_file}.py: Used by {used_by}")
        else:
            print(f"  {unused_file}.py: Not used by other modules either")

if __name__ == "__main__":
    main()