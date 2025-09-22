# NHL Machine Learning System Documentation

## Documentation Overview

This documentation provides comprehensive coverage of the NHL Machine Learning System, organized by module for easy navigation.

## Module Documentation

### Core Modules

| Module | Description | Documentation |
|--------|-------------|---------------|
| **Data Pipeline** | NHL API integration, data models, processing | [`data.md`](data.md) |
| **Feature Engineering** | Hockey-specific features, preprocessing | [`features.md`](features.md) |
| **Machine Learning** | Models, ensemble methods, training | [`models.md`](models.md) |
| **Predictions** | Production pipeline, model loading | [`predictions.md`](predictions.md) |
| **Optimization** | Fantasy team selection, constraints | [`optimization.md`](optimization.md) |
| **Evaluation** | Cross-validation, metrics, performance | [`evaluation.md`](evaluation.md) |

### Visual Documentation

| Graphics | Description | Location |
|----------|-------------|----------|
| **Architecture** | System flow diagrams | [`graphics/architecture.md`](graphics/architecture.md) |
| **Data Flow** | Data pipeline visualizations | [`graphics/data-flow.md`](graphics/data-flow.md) |
| **Feature Engineering** | Feature creation process | [`graphics/feature-engineering.md`](graphics/feature-engineering.md) |
| **Model Performance** | Performance charts and comparisons | [`graphics/model-performance.md`](graphics/model-performance.md) |
| **Hockey Analytics** | Domain-specific visualizations | [`graphics/hockey-analytics.md`](graphics/hockey-analytics.md) |

## Quick Navigation

### Getting Started
- [**System Overview**](overview.md) - High-level system description
- [**Installation Guide**](installation.md) - Setup and dependencies
- [**Quick Start**](quickstart.md) - First time usage examples

### Advanced Topics
- [**Production Deployment**](production.md) - Deployment considerations
- [**Performance Tuning**](performance.md) - Optimization strategies
- [**Troubleshooting**](troubleshooting.md) - Common issues and solutions

### Reference
- [**API Reference**](api-reference.md) - Complete method documentation
- [**Configuration**](configuration.md) - System configuration options
- [**Glossary**](glossary.md) - Hockey and ML terminology

## System Performance

**Current Best Model**: Lasso Regression
- **RMSE**: 0.209 PPG
- **R²**: 0.728 (72.8% variance explained)
- **Features**: 86 hockey-specific engineered features
- **Training Data**: 858 player-seasons

## Architecture Summary

```
NHL API → Data Processing → Feature Engineering → ML Training → Production Deployment
   ↓           ↓               ↓                    ↓              ↓
19 columns  Cleaned Data   86 Features        Best Model    Fantasy Optimization
```

## Key Technologies

- **Python**: Core language
- **Scikit-learn**: Machine learning framework
- **XGBoost**: Gradient boosting
- **Pandas/NumPy**: Data manipulation
- **Joblib**: Model serialization
- **JSON**: Component serialization

## Project Status

- **Data Pipeline**: Complete and tested
- **Feature Engineering**: 86+ features implemented
- **Model Training**: Ensemble methods working
- **Production API**: NHLModelPredictor ready
- **Optimization**: Fantasy team selection
- **Documentation**: Comprehensive coverage

## Contributing

When contributing to the documentation:

1. **Follow Structure**: Use the established module organization
2. **Include Examples**: Provide practical code examples
3. **Add Visuals**: Create diagrams for complex concepts
4. **Test Code**: Ensure all examples work
5. **Update Index**: Add new documents to this README

## Support

For documentation issues or questions:
1. Check the relevant module documentation
2. Review the [troubleshooting guide](troubleshooting.md)
3. Examine the [API reference](api-reference.md)
4. Look at practical examples in each module doc

---

**Last Updated**: 2025-09-21
**System Version**: 1.0
**Documentation Version**: 1.0