# NHL Pool Optimization System

A machine learning-powered system for optimizing NHL fantasy hockey team selection using predictive analytics and mathematical optimization.

## 🏒 Overview

This system predicts NHL player performance and selects optimal fantasy teams within salary and position constraints using:
- **Machine Learning**: Ensemble models (XGBoost, SVR, SGD) for PPG prediction
- **Optimization**: Linear Programming and Branch & Bound algorithms
- **Data Pipeline**: Automated NHL API data collection and processing

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Main Workflows

#### 1. Complete ML Pipeline
```bash
jupyter notebook NHL_ML_Models_Complete.ipynb
```
- Data collection and preprocessing
- Feature engineering and model training
- Performance evaluation and model selection

#### 2. Team Optimization
```bash
jupyter notebook Team_Optimization_Notebook.ipynb
```
- Load trained models and generate predictions
- Apply salary cap and position constraints
- Generate optimal team composition

#### 3. Original Analysis
```bash
jupyter notebook ML_for_NHL.ipynb
```
- Historical analysis and model development
- Exploratory data analysis
- Algorithm comparison

### Individual Components
```bash
# Test data download
python data_download.py

# Test data processing
python process_data.py

# Test ML models
python model_predictor.py

# Test optimization
python pool_classifier.py
```

## 📁 Project Structure

```
├── 📊 Core Notebooks
│   ├── NHL_ML_Models_Complete.ipynb    # Complete ML pipeline
│   ├── Team_Optimization_Notebook.ipynb  # Team optimization
│   └── ML_for_NHL.ipynb                # Original analysis
│
├── 🔧 Core Modules
│   ├── data_download.py                # NHL API data fetching
│   ├── process_data.py                 # Data processing & normalization
│   ├── player.py                       # Player & Season classes
│   ├── team.py                         # Team data model
│   ├── ensemble_learning.py            # ML ensemble methods
│   ├── model_predictor.py             # Model prediction interface
│   ├── pool_classifier.py             # Team optimization
│   ├── data_validation.py             # Data quality validation
│   └── config.py                      # Configuration settings
│
├── 🤖 ML Models (Structured)
│   ├── ml_models/                     # Modern ML architecture
│   │   ├── models/                    # Model implementations
│   │   ├── features/                  # Feature engineering
│   │   ├── evaluation/                # Model evaluation
│   │   └── utils/                     # ML utilities
│   └── models_saved/                  # Trained model artifacts
│
├── 📚 Documentation
│   ├── docs/                          # Comprehensive documentation
│   │   ├── data-module.md
│   │   ├── models-module.md
│   │   ├── optimizer-module.md
│   │   ├── core-classes.md
│   │   ├── workflow-documentation.md
│   │   └── graphs/                    # Visualization assets
│   └── CLAUDE.md                      # Development guide
│
├── 🏗️ Infrastructure
│   ├── src/                           # Legacy structured code
│   ├── requirements.txt               # Dependencies
│   └── archives/                      # Archived files
│       ├── old_notebooks/
│       ├── test_files/
│       ├── result_files/
│       ├── legacy_code/
│       └── documentation/
```

## 🎯 Key Features

### Machine Learning Pipeline
- **Ensemble Methods**: XGBoost, SVR, SGD, Logistic Regression
- **Feature Engineering**: Historical performance, player attributes, team context
- **Model Validation**: Cross-validation with hyperparameter tuning
- **Performance**: R² scores up to 0.768 for PPG prediction

### Optimization Engine
- **Linear Programming**: Guaranteed optimal solutions using PuLP
- **Branch & Bound**: Custom implementation for educational purposes
- **Constraints**: $88M salary cap, position limits (12A/6D/2G)
- **Efficiency**: Optimal team selection in <1 second

### Data Management
- **NHL API Integration**: Official data with robust error handling
- **Multi-Season Support**: Historical data spanning 4+ seasons
- **Data Validation**: Comprehensive quality checks and cleaning
- **Caching**: Efficient data storage and retrieval

## 📊 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **ML Models** | Best R² Score | 0.768 (Ensemble) |
| **Optimization** | Solve Time | <0.2 seconds (LP) |
| **Data Coverage** | Players | ~800 active NHL players |
| **Constraints** | Salary Cap | $88M with position limits |

## 🔧 Configuration

Key settings in `config.py` and `CLAUDE.md`:
- NHL API endpoints and rate limits
- Model hyperparameter spaces
- Optimization constraints
- Data validation rules

## 🚧 Development

### Testing
```bash
# Individual component testing (archived)
# See archives/test_files/ for historical tests
```

### Documentation
- **Module Documentation**: See `docs/` folder
- **Visual Guides**: Generated charts in `docs/graphs/`
- **API Reference**: Docstrings in all core modules

## 📈 Results

The system typically generates:
- **Optimal Teams**: 20 players maximizing predicted PPG
- **Performance**: ~250 total PPG for optimal lineups
- **Efficiency**: 95-98% salary cap utilization
- **Accuracy**: Validated through backtesting on historical data

## 🤝 Contributing

1. Follow existing code structure and documentation standards
2. Add comprehensive docstrings and type hints
3. Include tests for new functionality
4. Update relevant documentation in `docs/`

## 📞 Support

- **Documentation**: See `docs/` folder for detailed guides
- **Configuration**: Check `CLAUDE.md` for development setup
- **Issues**: Review code structure and validation logic

---

**Built for optimal NHL fantasy team selection through data-driven decision making.**