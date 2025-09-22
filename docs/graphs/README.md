# NHL Pool Optimization - Visualization Graphs

This directory contains Python scripts to generate comprehensive visualization graphs for the NHL Pool Optimization system. The graphs provide visual insights into system architecture, data flow, machine learning performance, and optimization results.

## Available Visualizations

### 1. System Architecture (`create_system_architecture.py`)
**Output**: `system_architecture.png/pdf`
- Complete system overview showing all modules and their relationships
- Color-coded layers (Data, Processing, ML, Optimization)
- Data flow arrows between components
- Legend and detailed module descriptions

### 2. Data Flow Diagram (`create_data_flow.py`)
**Output**: `data_flow_diagram.png/pdf`
- Detailed data flow from NHL API to final results
- Shows data transformation steps
- Includes process annotations and data format information
- Highlights key decision points in the pipeline

### 3. Machine Learning Pipeline (`create_ml_pipeline.py`)
**Output**: `ml_pipeline_flowchart.png/pdf`
- Complete ML workflow from feature engineering to predictions
- Individual model training paths (XGBoost, SVR, SGD, Logistic Regression)
- Hyperparameter tuning and cross-validation processes
- Ensemble model creation and final prediction generation

### 4. Performance Analysis (`create_performance_analysis.py`)
**Output**: `performance_analysis_charts.png/pdf`, `optimization_results.png/pdf`
- Model performance comparisons (MSE, R² scores)
- Prediction accuracy scatter plots
- Cross-validation performance across folds
- Feature importance rankings
- Salary vs performance analysis
- Position-specific performance distributions

### 5. Optimization Comparison (`create_optimization_comparison.py`)
**Output**: `optimization_comparison.png/pdf`, `constraint_analysis.png/pdf`
- Algorithm performance comparison (LP vs Branch & Bound vs Greedy)
- Computational efficiency metrics (solve time, memory usage)
- Optimality gap analysis
- Branch and bound convergence visualization
- Constraint sensitivity analysis
- Budget utilization optimization

### 6. Feature Importance Analysis (`create_feature_importance.py`)
**Output**: `feature_importance_analysis.png/pdf`, `data_exploration_charts.png/pdf`
- Feature importance across different ML models
- SHAP-like feature impact analysis
- Feature correlation heatmaps
- Performance by player type analysis
- Data distribution exploration
- Age vs performance relationships

## Quick Start

### Generate All Graphs
```bash
# Navigate to the graphs directory
cd docs/graphs

# Run the master generation script
python generate_all_graphs.py
```

### Generate Individual Graph Sets
```bash
# System architecture only
python create_system_architecture.py

# ML pipeline analysis
python create_ml_pipeline.py

# Performance analysis
python create_performance_analysis.py

# Optimization comparison
python create_optimization_comparison.py

# Feature analysis
python create_feature_importance.py

# Data flow diagram
python create_data_flow.py
```

## Requirements

### Python Packages
```bash
pip install matplotlib numpy pandas seaborn
```

### Optional (for enhanced visualizations)
```bash
pip install plotly  # For interactive graphs (future enhancement)
pip install networkx  # For network-style diagrams
```

## Output Files

All scripts generate both PNG (high-resolution) and PDF (vector) formats:

```
docs/graphs/
├── system_architecture.png
├── system_architecture.pdf
├── data_flow_diagram.png
├── data_flow_diagram.pdf
├── ml_pipeline_flowchart.png
├── ml_pipeline_flowchart.pdf
├── performance_analysis_charts.png
├── performance_analysis_charts.pdf
├── optimization_results.png
├── optimization_results.pdf
├── optimization_comparison.png
├── optimization_comparison.pdf
├── constraint_analysis.png
├── constraint_analysis.pdf
├── feature_importance_analysis.png
├── feature_importance_analysis.pdf
├── data_exploration_charts.png
└── data_exploration_charts.pdf
```

## Customization

### Colors and Styling
Each script uses consistent color schemes:
- **Data Layer**: Light blue (#E3F2FD)
- **Processing**: Light orange (#FFF3E0)
- **ML Models**: Light purple (#F3E5F5)
- **Optimization**: Light green (#E8F5E8)
- **Output**: Light red (#FFEBEE)

### Modifying Graphs
To customize visualizations:

1. **Change Colors**: Modify the `colors` dictionary in each script
2. **Adjust Layout**: Modify component positions and sizes
3. **Add Components**: Add new elements to the `components` dictionary
4. **Update Data**: Replace sample data with actual model results

### Example Customization
```python
# In create_system_architecture.py
colors = {
    'data': '#YOUR_COLOR_HERE',
    'processing': '#YOUR_COLOR_HERE',
    # ... other colors
}
```

## Graph Descriptions

### System Architecture Features
- **Layered Design**: Clear separation of system layers
- **Data Flow**: Arrows showing information flow
- **Module Details**: Detailed descriptions of each component
- **Technology Stack**: Shows technologies used (PuLP, scikit-learn, etc.)

### Data Flow Features
- **API Integration**: NHL API endpoints and data structures
- **Processing Steps**: Data transformation and validation
- **Storage Format**: JSON file organization
- **Error Handling**: Shows validation and error handling steps

### ML Pipeline Features
- **Model Diversity**: Shows different algorithm types
- **Hyperparameter Space**: Visualizes tuning parameters
- **Cross-Validation**: 5-fold validation process
- **Ensemble Method**: VotingRegressor combination

### Performance Analysis Features
- **Comparative Metrics**: Side-by-side model comparison
- **Statistical Validation**: R² scores and error metrics
- **Distribution Analysis**: Performance across different player segments
- **Feature Ranking**: Most important predictive features

### Optimization Features
- **Algorithm Comparison**: LP vs heuristic methods
- **Efficiency Metrics**: Time and memory usage
- **Solution Quality**: Optimality gap analysis
- **Constraint Impact**: Which constraints limit performance most

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Install missing packages
pip install matplotlib numpy pandas seaborn

# For macOS, you might need:
pip install matplotlib --upgrade
```

**Display Issues**:
```python
# If running on headless server, add this to scripts:
import matplotlib
matplotlib.use('Agg')
```

**Memory Issues**:
```python
# Close figures after saving to free memory
plt.close(fig)
```

**Font Issues**:
```python
# Use system fonts if custom fonts cause issues
plt.rcParams['font.family'] = 'DejaVu Sans'
```

## Advanced Usage

### Batch Generation with Custom Parameters
```python
from generate_all_graphs import generate_all_graphs

# Generate with custom settings
success = generate_all_graphs()
if success:
    print("All graphs generated successfully!")
```

### Integration with Actual Model Results
To use real model results instead of sample data:

1. **Train Your Models**: Run the complete ML pipeline
2. **Export Results**: Save model performance, predictions, feature importance
3. **Update Scripts**: Replace sample data with actual results
4. **Generate Graphs**: Run visualization scripts

### Automated Report Generation
```bash
# Generate graphs and compile into report
python generate_all_graphs.py
# Then use generated images in documentation/reports
```

## Future Enhancements

- **Interactive Visualizations**: Plotly-based interactive charts
- **Real-time Updates**: Live performance monitoring dashboards
- **3D Visualizations**: Advanced feature space exploration
- **Animation**: Time-series progression animations
- **Custom Themes**: Multiple visualization themes

This visualization suite provides comprehensive insights into every aspect of the NHL Pool Optimization system, from high-level architecture to detailed performance metrics.