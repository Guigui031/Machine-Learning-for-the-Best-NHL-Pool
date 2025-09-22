# Archives

This folder contains files that have been moved out of the main codebase during the refactoring process. These files are preserved for historical reference but are not part of the active development workflow.

## 📁 Archive Structure

### `old_notebooks/`
Historical Jupyter notebooks that have been superseded by newer versions:
- `NHL_Data_Pipeline_Demo.ipynb` - Legacy data pipeline demo

### `test_files/`
Development and testing files:
- `test_*.py` - Various test scripts
- `debug_features.py` - Feature debugging utilities

### `result_files/`
Output files and generated results:
- `*.csv` - Historical optimization results
- `*.png` - Generated plots and charts
- `*.pdf` - Project documentation and reports

### `legacy_code/`
Deprecated code modules:
- `scrape_salaries.py` - Legacy salary scraping functionality
- `data_pipeline.py` - Old data pipeline implementation

### `documentation/`
Outdated documentation and guides:
- `MIGRATION_GUIDE.md` - Historical migration instructions
- `NHL_ML_SYSTEM_DOCUMENTATION.md` - Legacy system documentation
- `README_ML_SYSTEM.md` - Old README variants
- `nhl_data_pipeline_summary.json` - Pipeline summary
- `run_tests.bat` - Windows batch test runner

## 🔄 Refactoring Summary

**Date**: September 2024
**Reason**: Clean up codebase and improve maintainability

### Files Moved
- **Total**: 25+ files archived
- **Categories**: Old notebooks, test files, results, legacy code, documentation

### Active Codebase
The following files remain in the main directory:
- Core notebooks (`NHL_ML_Models_Complete.ipynb`, `Team_Optimization_Notebook.ipynb`, `ML_for_NHL.ipynb`)
- Core modules (`data_download.py`, `process_data.py`, `player.py`, etc.)
- Modern ML architecture (`ml_models/`, `models_saved/`)
- Updated documentation (`docs/`, `CLAUDE.md`)

## 🗂️ File Recovery

If any archived file is needed for active development:
1. Locate the file in the appropriate archive subdirectory
2. Copy (don't move) back to the main directory
3. Update any import statements or dependencies
4. Test functionality with current codebase

## 📚 Historical Context

These archived files represent the evolution of the NHL Pool Optimization project:
- Early experimental notebooks
- Various testing approaches
- Multiple documentation iterations
- Legacy implementation patterns

While not actively maintained, they provide valuable context for understanding the project's development history and may contain useful code snippets or methodologies.