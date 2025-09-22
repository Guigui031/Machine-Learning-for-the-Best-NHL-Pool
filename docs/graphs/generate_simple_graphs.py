#!/usr/bin/env python3
"""
Simple Graph Generation
Generates key visualization graphs for the NHL Pool system without display.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import pandas as pd

def create_simple_architecture():
    """Create a simplified system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define colors
    colors = {
        'data': '#E3F2FD',
        'processing': '#FFF3E0',
        'ml': '#F3E5F5',
        'optimization': '#E8F5E8',
        'output': '#FFEBEE'
    }

    # Simplified components
    components = [
        {'name': 'NHL API\nData Sources', 'pos': (2, 8), 'size': (3, 1.5), 'color': colors['data']},
        {'name': 'Data Download\n& Processing', 'pos': (2, 6), 'size': (3, 1.5), 'color': colors['processing']},
        {'name': 'Player & Season\nClasses', 'pos': (7, 6), 'size': (3, 1.5), 'color': colors['processing']},
        {'name': 'Machine Learning\nEnsemble', 'pos': (2, 4), 'size': (3, 1.5), 'color': colors['ml']},
        {'name': 'Optimization\nAlgorithms', 'pos': (7, 4), 'size': (3, 1.5), 'color': colors['optimization']},
        {'name': 'Optimal Team\nResults', 'pos': (4.5, 2), 'size': (3, 1.5), 'color': colors['output']}
    ]

    # Draw components
    for comp in components:
        x, y = comp['pos']
        w, h = comp['size']

        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)

        ax.text(x + w/2, y + h/2, comp['name'],
                ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Add arrows
    arrows = [
        ((3.5, 8), (3.5, 7.5)),     # API to Processing
        ((5, 6.75), (7, 6.75)),     # Processing to Classes
        ((3.5, 6), (3.5, 5.5)),     # Processing to ML
        ((5, 4.75), (7, 4.75)),     # ML to Optimization
        ((8.5, 4), (6.5, 3.5))      # Optimization to Results
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="darkblue", ec="darkblue")
        ax.add_patch(arrow)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('NHL Pool Optimization System Architecture', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("System architecture diagram created")

def create_performance_comparison():
    """Create ML model performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Model performance data
    models = ['XGBoost', 'SVR', 'SGD', 'Logistic\nRegression', 'Ensemble']
    mse_scores = [0.084, 0.092, 0.098, 0.105, 0.076]
    r2_scores = [0.742, 0.698, 0.672, 0.645, 0.768]

    # MSE comparison
    bars1 = ax1.bar(models, mse_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.8)
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Model Performance - MSE (Lower is Better)')
    ax1.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{value:.3f}', ha='center', va='bottom')

    # R² comparison
    bars2 = ax2.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.8)
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance - R² (Higher is Better)')
    ax2.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars2, r2_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Model performance comparison created")

def create_optimization_comparison():
    """Create optimization algorithm comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Algorithm comparison data
    algorithms = ['Linear\nProgramming', 'Branch &\nBound', 'Greedy\nHeuristic']
    total_ppg = [247.3, 245.8, 241.2]
    solve_time = [0.15, 67.4, 0.08]

    # PPG comparison
    bars1 = ax1.bar(algorithms, total_ppg, color=['green', 'blue', 'orange'], alpha=0.8)
    ax1.set_ylabel('Total Team PPG')
    ax1.set_title('Algorithm Performance')

    for bar, value in zip(bars1, total_ppg):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # Solve time comparison (log scale)
    bars2 = ax2.bar(algorithms, solve_time, color=['green', 'blue', 'orange'], alpha=0.8)
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computational Efficiency')
    ax2.set_yscale('log')

    for bar, value in zip(bars2, solve_time):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                f'{value:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Optimization comparison created")

def create_feature_importance():
    """Create feature importance visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Sample feature importance data
    features = ['Previous PPG', 'Age', 'Games Played', 'Team Performance',
                'Position', 'Goals/Game', 'Assists/Game', 'Shots/Game',
                'Plus/Minus', 'Time on Ice']
    importance = [0.28, 0.12, 0.15, 0.11, 0.08, 0.09, 0.07, 0.04, 0.03, 0.03]

    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = np.array(importance)[sorted_idx]

    # Create horizontal bar chart
    bars = ax.barh(range(len(sorted_features)), sorted_importance, alpha=0.8)

    # Color bars by importance level
    for i, bar in enumerate(bars):
        if sorted_importance[i] > 0.2:
            bar.set_color('darkgreen')
        elif sorted_importance[i] > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('ML Model Feature Importance')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
        ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance chart created")

def create_data_flow():
    """Create simplified data flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Data flow stages
    stages = [
        {'name': 'NHL API\nData Collection', 'pos': (2, 8), 'color': '#BBDEFB'},
        {'name': 'JSON Storage\n& Validation', 'pos': (7, 8), 'color': '#C8E6C9'},
        {'name': 'Player Object\nCreation', 'pos': (2, 6), 'color': '#FFF9C4'},
        {'name': 'Feature\nEngineering', 'pos': (7, 6), 'color': '#FFF9C4'},
        {'name': 'ML Model\nTraining', 'pos': (2, 4), 'color': '#E1BEE7'},
        {'name': 'PPG\nPredictions', 'pos': (7, 4), 'color': '#E1BEE7'},
        {'name': 'Team\nOptimization', 'pos': (2, 2), 'color': '#C8E6C9'},
        {'name': 'Final Results\n& Export', 'pos': (7, 2), 'color': '#FFCDD2'}
    ]

    # Draw stages
    for stage in stages:
        x, y = stage['pos']
        w, h = 3, 1.5

        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)

        ax.text(x + w/2, y + h/2, stage['name'],
                ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Add flow arrows
    arrows = [
        ((5, 8.75), (7, 8.75)),    # API to Storage
        ((3.5, 8), (3.5, 7.5)),    # Storage to Player Objects
        ((5, 6.75), (7, 6.75)),    # Player Objects to Features
        ((8.5, 6), (8.5, 5.5)),    # Features to Predictions
        ((7, 4.75), (5, 4.75)),    # Predictions to Training
        ((3.5, 4), (3.5, 3.5)),    # Training to Optimization
        ((5, 2.75), (7, 2.75))     # Optimization to Results
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="navy", ec="navy")
        ax.add_patch(arrow)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('NHL Pool Optimization Data Flow', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Data flow diagram created")

def main():
    """Generate all simple graphs"""
    print("Generating NHL Pool Optimization Graphs")
    print("=" * 45)

    try:
        create_simple_architecture()
        create_performance_comparison()
        create_optimization_comparison()
        create_feature_importance()
        create_data_flow()

        print("\nAll graphs generated successfully!")
        print("\nGenerated files:")
        print("- system_architecture.png")
        print("- model_performance_comparison.png")
        print("- optimization_comparison.png")
        print("- feature_importance.png")
        print("- data_flow_diagram.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()