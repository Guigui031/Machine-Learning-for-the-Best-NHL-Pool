#!/usr/bin/env python3
"""
Performance Analysis Charts
Generates various performance analysis visualizations for the NHL Pool system.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_performance_charts():
    """Create comprehensive performance analysis charts"""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Generate sample data for demonstrations
    np.random.seed(42)

    # 1. Model Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    models = ['XGBoost', 'SVR', 'SGD', 'Logistic Reg', 'Ensemble']
    mse_scores = [0.084, 0.092, 0.098, 0.105, 0.076]  # Sample MSE scores
    r2_scores = [0.742, 0.698, 0.672, 0.645, 0.768]   # Sample R² scores

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mse_scores, width, label='MSE', alpha=0.8)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, r2_scores, width, label='R²', alpha=0.8, color='orange')

    ax1.set_xlabel('Models')
    ax1.set_ylabel('Mean Squared Error', color='blue')
    ax1_twin.set_ylabel('R² Score', color='orange')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax1_twin.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

    # 2. Prediction vs Actual PPG Scatter Plot
    ax2 = plt.subplot(2, 3, 2)
    n_players = 500
    actual_ppg = np.random.gamma(2, 0.4, n_players)  # Realistic PPG distribution
    prediction_error = np.random.normal(0, 0.1, n_players)
    predicted_ppg = actual_ppg + prediction_error

    ax2.scatter(actual_ppg, predicted_ppg, alpha=0.6, s=30)
    ax2.plot([0, 3], [0, 3], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual PPG')
    ax2.set_ylabel('Predicted PPG')
    ax2.set_title('Prediction Accuracy: Actual vs Predicted PPG')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate and display R²
    correlation_matrix = np.corrcoef(actual_ppg, predicted_ppg)
    r2 = correlation_matrix[0,1]**2
    ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    # 3. Cross-Validation Scores
    ax3 = plt.subplot(2, 3, 3)
    cv_folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    model_names = ['XGBoost', 'SVR', 'SGD', 'Ensemble']

    # Sample CV scores (negative MSE converted to positive for visualization)
    cv_data = {
        'XGBoost': [-0.078, -0.081, -0.089, -0.076, -0.084],
        'SVR': [-0.089, -0.095, -0.098, -0.087, -0.092],
        'SGD': [-0.094, -0.102, -0.105, -0.091, -0.098],
        'Ensemble': [-0.072, -0.075, -0.082, -0.071, -0.076]
    }

    for i, model in enumerate(model_names):
        scores = [-score for score in cv_data[model]]  # Convert to positive
        ax3.plot(cv_folds, scores, marker='o', linewidth=2, label=model)

    ax3.set_ylabel('MSE Score')
    ax3.set_title('Cross-Validation Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Feature Importance (for tree-based models)
    ax4 = plt.subplot(2, 3, 4)
    features = ['Previous PPG', 'Age', 'Games Played', 'Team Performance',
                'Position', 'Goals/Game', 'Assists/Game', 'Shots/Game',
                'Plus/Minus', 'Time on Ice']
    importance = np.array([0.25, 0.08, 0.12, 0.15, 0.06, 0.18, 0.16, 0.11, 0.04, 0.09])

    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    bars = ax4.barh(sorted_features, sorted_importance, alpha=0.8)
    ax4.set_xlabel('Feature Importance')
    ax4.set_title('Feature Importance (XGBoost)')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')

    # 5. Salary vs Performance Analysis
    ax5 = plt.subplot(2, 3, 5)
    n_players = 300
    # Generate realistic salary distribution (log-normal)
    salaries = np.random.lognormal(15, 0.8, n_players) / 1000  # In thousands
    # PPG tends to correlate with salary but with noise
    base_ppg = (salaries / 5000) ** 0.3 + np.random.normal(0, 0.2, n_players)
    base_ppg = np.clip(base_ppg, 0.1, 3.0)

    scatter = ax5.scatter(salaries/1000, base_ppg, alpha=0.6, s=30, c=base_ppg, cmap='viridis')
    ax5.set_xlabel('Salary (Millions $)')
    ax5.set_ylabel('Predicted PPG')
    ax5.set_title('Salary vs Performance Analysis')

    # Add trend line
    z = np.polyfit(salaries/1000, base_ppg, 1)
    p = np.poly1d(z)
    ax5.plot(sorted(salaries/1000), p(sorted(salaries/1000)), "r--", alpha=0.8, linewidth=2)

    plt.colorbar(scatter, ax=ax5, label='PPG')
    ax5.grid(True, alpha=0.3)

    # 6. Position Performance Distribution
    ax6 = plt.subplot(2, 3, 6)
    positions = ['Attackers', 'Defensemen', 'Goalies']

    # Sample data for different positions (different scoring systems)
    attacker_ppg = np.random.gamma(2.5, 0.35, 400)  # Higher PPG for attackers
    defensemen_ppg = np.random.gamma(2.0, 0.30, 200)  # Medium PPG for defensemen
    goalie_ppg = np.random.gamma(1.8, 0.40, 100)     # Variable PPG for goalies

    position_data = [attacker_ppg, defensemen_ppg, goalie_ppg]

    # Create violin plot
    parts = ax6.violinplot(position_data, positions=range(len(positions)), showmeans=True)

    for pc in parts['bodies']:
        pc.set_alpha(0.7)

    ax6.set_xticks(range(len(positions)))
    ax6.set_xticklabels(positions)
    ax6.set_ylabel('Predicted PPG')
    ax6.set_title('PPG Distribution by Position')
    ax6.grid(True, alpha=0.3)

    # Add mean values as text
    means = [np.mean(data) for data in position_data]
    for i, mean in enumerate(means):
        ax6.text(i, mean + 0.1, f'μ={mean:.2f}', ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('performance_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.savefig('performance_analysis_charts.pdf', bbox_inches='tight')
    print("Performance analysis charts saved as performance_analysis_charts.png/pdf")

    return fig

def create_optimization_results_chart():
    """Create optimization results comparison chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Algorithm Comparison
    algorithms = ['Linear\nProgramming', 'Branch &\nBound', 'Greedy\nHeuristic']
    total_ppg = [245.7, 243.2, 238.9]  # Sample optimal PPG values
    solve_time = [0.12, 45.3, 0.05]    # Sample solve times in seconds

    # PPG comparison
    bars1 = ax1.bar(algorithms, total_ppg, alpha=0.8, color=['green', 'blue', 'orange'])
    ax1.set_ylabel('Total Team PPG')
    ax1.set_title('Optimization Algorithm Performance')

    for bar, value in zip(bars1, total_ppg):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # Solve time comparison (log scale)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar([x + 0.3 for x in range(len(algorithms))], solve_time,
                        width=0.3, alpha=0.6, color='red', label='Solve Time')
    ax1_twin.set_ylabel('Solve Time (seconds)', color='red')
    ax1_twin.set_yscale('log')

    # 2. Salary Distribution
    positions = ['Attackers\n(12)', 'Defensemen\n(6)', 'Goalies\n(2)']
    salary_millions = [42.5, 28.3, 17.2]  # Sample salary allocation
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']

    wedges, texts, autotexts = ax2.pie(salary_millions, labels=positions, autopct='%1.1f%%',
                                      colors=colors_pie, startangle=90)
    ax2.set_title('Salary Cap Allocation\nTotal: $88M')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # 3. Team Composition
    ax3.axis('off')

    # Create a simple team roster visualization
    roster_data = {
        'Position': ['Attackers'] * 12 + ['Defensemen'] * 6 + ['Goalies'] * 2,
        'Player': [f'A{i+1}' for i in range(12)] + [f'D{i+1}' for i in range(6)] + ['G1', 'G2'],
        'PPG': np.concatenate([
            np.random.uniform(0.8, 1.4, 12),  # Attackers
            np.random.uniform(0.6, 1.1, 6),   # Defensemen
            np.random.uniform(0.7, 1.3, 2)    # Goalies
        ]),
        'Salary': np.concatenate([
            np.random.uniform(2, 8, 12),      # Attackers (millions)
            np.random.uniform(3, 7, 6),       # Defensemen
            np.random.uniform(4, 9, 2)        # Goalies
        ])
    }

    df = pd.DataFrame(roster_data)

    # Create a bubble chart
    colors_map = {'Attackers': 'red', 'Defensemen': 'blue', 'Goalies': 'green'}
    for position in df['Position'].unique():
        pos_data = df[df['Position'] == position]
        ax3.scatter(pos_data['Salary'], pos_data['PPG'],
                   s=pos_data['PPG']*100, alpha=0.6,
                   color=colors_map[position], label=position)

    ax3.set_xlabel('Salary (Millions $)')
    ax3.set_ylabel('Predicted PPG')
    ax3.set_title('Optimal Team Composition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Budget Utilization
    budget_categories = ['Players\nSelected', 'Remaining\nBudget']
    budget_values = [85.2, 2.8]  # Sample budget utilization
    colors_budget = ['lightcoral', 'lightgray']

    bars4 = ax4.bar(budget_categories, budget_values, color=colors_budget, alpha=0.8)
    ax4.set_ylabel('Budget (Millions $)')
    ax4.set_title('Budget Utilization')
    ax4.set_ylim(0, 90)

    # Add budget line
    ax4.axhline(y=88, color='red', linestyle='--', linewidth=2, label='Salary Cap')

    for bar, value in zip(bars4, budget_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')

    ax4.text(0.5, 85, f'Utilization: {budget_values[0]/88*100:.1f}%',
             ha='center', transform=ax4.transData,
             bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))

    ax4.legend()

    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('optimization_results.pdf', bbox_inches='tight')
    print("Optimization results chart saved as optimization_results.png/pdf")

    return fig

if __name__ == "__main__":
    # Create performance analysis charts
    fig1 = create_sample_performance_charts()

    # Create optimization results chart
    fig2 = create_optimization_results_chart()

    plt.show()