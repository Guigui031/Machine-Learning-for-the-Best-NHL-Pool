#!/usr/bin/env python3
"""
Machine Learning Pipeline Visualization
Generates detailed flowcharts showing the ML training and prediction pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Polygon
import numpy as np

def create_ml_pipeline_flowchart():
    """Create ML pipeline flowchart"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Define colors
    colors = {
        'data': '#E8F5E8',      # Light green
        'feature': '#FFF3E0',   # Light orange
        'model': '#E3F2FD',     # Light blue
        'validation': '#F3E5F5', # Light purple
        'output': '#FFEBEE'     # Light red
    }

    # Pipeline components
    components = {
        # Data Preparation
        'raw_data': {'pos': (2, 12), 'size': (3, 1), 'color': colors['data'],
                    'text': 'Raw Player Data\n• Multi-season statistics\n• Biographical information'},
        'data_filter': {'pos': (2, 10.5), 'size': (3, 1), 'color': colors['data'],
                       'text': 'Data Filtering\n• Min 3 seasons played\n• Min 100 total games'},

        # Feature Engineering
        'sliding_window': {'pos': (7, 12), 'size': (3, 1), 'color': colors['feature'],
                          'text': 'Sliding Window\n• Use seasons N-1, N\n• Predict season N+1'},
        'feature_extraction': {'pos': (7, 10.5), 'size': (3, 1), 'color': colors['feature'],
                              'text': 'Feature Extraction\n• Historical PPG\n• Player attributes\n• Team context'},
        'normalization': {'pos': (7, 9), 'size': (3, 1), 'color': colors['feature'],
                         'text': 'Data Normalization\n• Z-score scaling\n• Per-game statistics'},

        # Model Training Branch
        'train_test_split': {'pos': (2, 7.5), 'size': (3, 1), 'color': colors['validation'],
                            'text': 'Train/Test Split\n• Temporal split\n• 80/20 ratio'},

        # Individual Models
        'xgboost': {'pos': (0.5, 5.5), 'size': (2, 1.5), 'color': colors['model'],
                   'text': 'XGBoost\n• Gradient boosting\n• Tree-based\n• Handles interactions'},
        'svr': {'pos': (3, 5.5), 'size': (2, 1.5), 'color': colors['model'],
               'text': 'SVR\n• Support vectors\n• Kernel tricks\n• Robust to outliers'},
        'sgd': {'pos': (5.5, 5.5), 'size': (2, 1.5), 'color': colors['model'],
               'text': 'SGD Regressor\n• Stochastic gradient\n• Linear model\n• Fast training'},
        'logistic': {'pos': (8, 5.5), 'size': (2, 1.5), 'color': colors['model'],
                    'text': 'Logistic Reg.\n• Linear baseline\n• L1 regularization\n• Interpretable'},

        # Hyperparameter Tuning
        'hyperparam_xgb': {'pos': (0.5, 3.5), 'size': (2, 1), 'color': colors['validation'],
                          'text': 'XGB Tuning\n• learning_rate\n• n_estimators\n• max_depth'},
        'hyperparam_svr': {'pos': (3, 3.5), 'size': (2, 1), 'color': colors['validation'],
                          'text': 'SVR Tuning\n• C parameter\n• epsilon\n• kernel type'},
        'hyperparam_sgd': {'pos': (5.5, 3.5), 'size': (2, 1), 'color': colors['validation'],
                          'text': 'SGD Tuning\n• alpha\n• l1_ratio\n• max_iter'},
        'hyperparam_lr': {'pos': (8, 3.5), 'size': (2, 1), 'color': colors['validation'],
                         'text': 'LR Tuning\n• C parameter\n• penalty\n• solver'},

        # Cross-validation
        'cv_process': {'pos': (11.5, 5), 'size': (3, 2), 'color': colors['validation'],
                      'text': 'Cross-Validation\n• 5-fold stratified\n• RandomizedSearchCV\n• MSE scoring\n• Best model selection'},

        # Ensemble
        'ensemble': {'pos': (4, 1.5), 'size': (3, 1.5), 'color': colors['model'],
                    'text': 'Ensemble Model\n• VotingRegressor\n• Equal weights\n• Average predictions'},

        # Final Validation
        'final_validation': {'pos': (8.5, 1.5), 'size': (3, 1.5), 'color': colors['validation'],
                            'text': 'Final Validation\n• Test set evaluation\n• Performance metrics\n• Error analysis'},

        # Prediction Phase
        'new_season_data': {'pos': (12.5, 10), 'size': (2.5, 1), 'color': colors['data'],
                           'text': 'Current Season\n• Latest player data\n• Updated features'},
        'predictions': {'pos': (12.5, 7), 'size': (2.5, 1.5), 'color': colors['output'],
                       'text': 'PPG Predictions\n• All active players\n• Confidence intervals\n• Player rankings'}
    }

    # Draw components
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']

        # Create rounded rectangle
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)

        # Add text
        fontweight = 'bold' if 'Model' in comp['text'] or 'Ensemble' in comp['text'] else 'normal'
        ax.text(x + w/2, y + h/2, comp['text'],
                ha='center', va='center',
                fontsize=9, fontweight=fontweight)

    # Define connections
    connections = [
        # Data flow
        ('raw_data', 'data_filter'),
        ('data_filter', 'train_test_split'),
        ('raw_data', 'sliding_window'),
        ('sliding_window', 'feature_extraction'),
        ('feature_extraction', 'normalization'),

        # To individual models
        ('train_test_split', 'xgboost'),
        ('train_test_split', 'svr'),
        ('train_test_split', 'sgd'),
        ('train_test_split', 'logistic'),

        # Hyperparameter tuning
        ('xgboost', 'hyperparam_xgb'),
        ('svr', 'hyperparam_svr'),
        ('sgd', 'hyperparam_sgd'),
        ('logistic', 'hyperparam_lr'),

        # Cross-validation
        ('hyperparam_xgb', 'cv_process'),
        ('hyperparam_svr', 'cv_process'),
        ('hyperparam_sgd', 'cv_process'),
        ('hyperparam_lr', 'cv_process'),

        # To ensemble
        ('cv_process', 'ensemble'),
        ('ensemble', 'final_validation'),

        # Prediction flow
        ('new_season_data', 'predictions'),
        ('ensemble', 'predictions')
    ]

    # Draw connections
    for start, end in connections:
        start_comp = components[start]
        end_comp = components[end]

        # Calculate connection points
        start_x = start_comp['pos'][0] + start_comp['size'][0] / 2
        start_y = start_comp['pos'][1]
        end_x = end_comp['pos'][0] + end_comp['size'][0] / 2
        end_y = end_comp['pos'][1] + end_comp['size'][1]

        # Special routing for complex connections
        if start == 'train_test_split':
            start_x = start_comp['pos'][0] + start_comp['size'][0] / 2
            start_y = start_comp['pos'][1]
            if end in ['xgboost', 'svr', 'sgd', 'logistic']:
                end_y = end_comp['pos'][1] + end_comp['size'][1]

        elif 'hyperparam' in start and end == 'cv_process':
            start_x = start_comp['pos'][0] + start_comp['size'][0]
            start_y = start_comp['pos'][1] + start_comp['size'][1] / 2
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['size'][1] / 2

        elif start == 'ensemble' and end == 'predictions':
            start_x = start_comp['pos'][0] + start_comp['size'][0]
            start_y = start_comp['pos'][1] + start_comp['size'][1] / 2
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['size'][1] / 2

        arrow = ConnectionPatch((start_x, start_y), (end_x, end_y),
                               "data", "data",
                               arrowstyle="->",
                               shrinkA=5, shrinkB=5,
                               mutation_scale=15,
                               fc="darkblue",
                               ec="darkblue",
                               linewidth=1.5)
        ax.add_patch(arrow)

    # Add process annotations
    annotations = [
        {'text': 'Training Phase', 'pos': (1, 8.5), 'rotation': 90},
        {'text': 'Model Selection', 'pos': (5, 4.5), 'rotation': 0},
        {'text': 'Prediction Phase', 'pos': (13.5, 8.5), 'rotation': 90}
    ]

    for ann in annotations:
        ax.annotate(ann['text'], xy=ann['pos'], xytext=ann['pos'],
                   fontsize=12, fontweight='bold',
                   rotation=ann['rotation'],
                   bbox=dict(boxstyle="round,pad=0.5",
                            facecolor='lightgray',
                            alpha=0.7))

    # Add detailed process boxes
    detail_boxes = [
        {'pos': (0.5, 8.5), 'size': (10, 0.8), 'text': 'Feature Engineering Pipeline: Historical PPG → Player Age/Position → Team Performance → Normalized Features'},
        {'pos': (11, 2.5), 'size': (4, 2.5), 'text': 'Cross-Validation Process:\n1. Split data into 5 folds\n2. Train on 4 folds, validate on 1\n3. Repeat for all combinations\n4. Select best hyperparameters\n5. Retrain on full dataset'}
    ]

    for box in detail_boxes:
        x, y = box['pos']
        w, h = box['size']
        detail_box = FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightyellow',
                                   edgecolor='orange',
                                   linewidth=2,
                                   alpha=0.9)
        ax.add_patch(detail_box)
        ax.text(x + w/2, y + h/2, box['text'],
                ha='center', va='center',
                fontsize=10, style='italic')

    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Add title
    plt.title('Machine Learning Pipeline for NHL Player Performance Prediction',
              fontsize=18, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['data'], label='Data Processing'),
        mpatches.Patch(color=colors['feature'], label='Feature Engineering'),
        mpatches.Patch(color=colors['model'], label='Model Training'),
        mpatches.Patch(color=colors['validation'], label='Validation'),
        mpatches.Patch(color=colors['output'], label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    plt.savefig('ml_pipeline_flowchart.png', dpi=300, bbox_inches='tight')
    plt.savefig('ml_pipeline_flowchart.pdf', bbox_inches='tight')
    print("ML pipeline flowchart saved as ml_pipeline_flowchart.png/pdf")

    return fig

if __name__ == "__main__":
    create_ml_pipeline_flowchart()
    plt.show()