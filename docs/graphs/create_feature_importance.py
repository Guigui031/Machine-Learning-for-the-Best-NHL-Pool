#!/usr/bin/env python3
"""
Feature Importance and Analysis Visualization
Generates comprehensive feature analysis charts for the ML models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def create_feature_importance_analysis():
    """Create comprehensive feature importance analysis"""

    fig = plt.figure(figsize=(20, 16))

    # Sample feature importance data (would come from actual trained models)
    np.random.seed(42)

    features = [
        'Previous Season PPG', 'Age', 'Games Played (Prev)', 'Team Performance',
        'Position (Encoded)', 'Goals per Game', 'Assists per Game', 'Shots per Game',
        'Plus/Minus Rating', 'Time on Ice', 'Country (Encoded)', 'Height',
        'Weight', 'Penalty Minutes', '2-Season Trend', 'Team Strength'
    ]

    # Feature importance for different models
    model_importance = {
        'XGBoost': np.array([0.28, 0.12, 0.15, 0.11, 0.08, 0.09, 0.07, 0.04, 0.03, 0.02, 0.01]),
        'SVR': np.array([0.31, 0.18, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02]),
        'SGD': np.array([0.35, 0.15, 0.10, 0.08, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02]),
        'Logistic': np.array([0.33, 0.16, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02])
    }

    # Ensure we have enough features for visualization
    n_features_show = 11
    features_show = features[:n_features_show]

    # 1. Feature Importance Comparison Across Models
    ax1 = plt.subplot(2, 3, 1)

    x = np.arange(len(features_show))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (model, importance) in enumerate(model_importance.items()):
        bars = ax1.bar(x + i*width, importance, width, label=model, alpha=0.8, color=colors[i])

    ax1.set_xlabel('Features')
    ax1.set_ylabel('Feature Importance')
    ax1.set_title('Feature Importance by Model')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(features_show, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Average Feature Importance (Ensemble)
    ax2 = plt.subplot(2, 3, 2)

    # Calculate average importance across models
    avg_importance = np.mean([imp for imp in model_importance.values()], axis=0)
    std_importance = np.std([imp for imp in model_importance.values()], axis=0)

    # Sort features by average importance
    sorted_idx = np.argsort(avg_importance)[::-1]
    sorted_features = [features_show[i] for i in sorted_idx]
    sorted_avg = avg_importance[sorted_idx]
    sorted_std = std_importance[sorted_idx]

    bars = ax2.barh(range(len(sorted_features)), sorted_avg,
                    xerr=sorted_std, alpha=0.8, capsize=5)
    ax2.set_yticks(range(len(sorted_features)))
    ax2.set_yticklabels(sorted_features)
    ax2.set_xlabel('Average Feature Importance')
    ax2.set_title('Feature Ranking (Ensemble Average)')

    # Color bars by importance level
    for i, bar in enumerate(bars):
        if sorted_avg[i] > 0.2:
            bar.set_color('darkgreen')
        elif sorted_avg[i] > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_avg)):
        ax2.text(value + sorted_std[i] + 0.005, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')

    # 3. Feature Correlation Heatmap
    ax3 = plt.subplot(2, 3, 3)

    # Generate sample correlation matrix
    np.random.seed(123)
    n_features_corr = 8  # Show subset for clarity
    feature_subset = features_show[:n_features_corr]

    # Create realistic correlation matrix
    base_corr = np.random.randn(n_features_corr, n_features_corr)
    correlation_matrix = np.corrcoef(base_corr)

    # Make some realistic correlations
    correlation_matrix[0, 5] = 0.75  # PPG correlated with Goals per Game
    correlation_matrix[5, 0] = 0.75
    correlation_matrix[0, 6] = 0.65  # PPG correlated with Assists per Game
    correlation_matrix[6, 0] = 0.65
    correlation_matrix[1, 2] = -0.3   # Age slightly negatively correlated with Games Played
    correlation_matrix[2, 1] = -0.3

    im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(n_features_corr))
    ax3.set_yticks(range(n_features_corr))
    ax3.set_xticklabels([f.split()[0] for f in feature_subset], rotation=45, ha='right')
    ax3.set_yticklabels([f.split()[0] for f in feature_subset])

    # Add correlation values
    for i in range(n_features_corr):
        for j in range(n_features_corr):
            text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                           fontsize=8)

    ax3.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Feature Impact on Different Player Types
    ax4 = plt.subplot(2, 3, 4)

    player_types = ['Elite (>1.2 PPG)', 'Average (0.6-1.2)', 'Depth (<0.6 PPG)']
    feature_categories = ['Historical\nPerformance', 'Physical\nAttributes', 'Team\nContext', 'Situational\nStats']

    # Sample importance by player type
    importance_by_type = np.array([
        [0.45, 0.15, 0.25, 0.15],  # Elite players
        [0.35, 0.20, 0.30, 0.15],  # Average players
        [0.30, 0.25, 0.35, 0.10]   # Depth players
    ])

    x = np.arange(len(feature_categories))
    width = 0.25

    for i, player_type in enumerate(player_types):
        bars = ax4.bar(x + i*width, importance_by_type[i], width,
                      label=player_type, alpha=0.8)

    ax4.set_xlabel('Feature Categories')
    ax4.set_ylabel('Relative Importance')
    ax4.set_title('Feature Importance by Player Type')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(feature_categories)
    ax4.legend()

    # 5. SHAP-like Feature Impact Analysis
    ax5 = plt.subplot(2, 3, 5)

    # Simulate SHAP values (impact on predictions)
    n_samples = 100
    features_shap = ['Prev PPG', 'Age', 'Games', 'Team Perf', 'Position']

    # Generate sample SHAP values for different players
    shap_values = []
    for i in range(len(features_shap)):
        if i == 0:  # Previous PPG has highest impact
            values = np.random.normal(0.15, 0.08, n_samples)
        elif i == 1:  # Age
            values = np.random.normal(-0.02, 0.05, n_samples)
        else:  # Other features
            values = np.random.normal(0.0, 0.03, n_samples)
        shap_values.append(values)

    # Create violin plot
    parts = ax5.violinplot(shap_values, positions=range(len(features_shap)), showmeans=True)

    for pc in parts['bodies']:
        pc.set_alpha(0.7)

    ax5.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax5.set_xticks(range(len(features_shap)))
    ax5.set_xticklabels(features_shap)
    ax5.set_ylabel('SHAP Value (Impact on Prediction)')
    ax5.set_title('Feature Impact Distribution\n(SHAP-like Analysis)')
    ax5.grid(True, alpha=0.3)

    # 6. Feature Selection History
    ax6 = plt.subplot(2, 3, 6)

    # Simulate feature selection process
    selection_rounds = np.arange(1, 11)
    n_features_selected = [16, 14, 12, 11, 10, 9, 8, 7, 6, 5]
    model_performance = [0.742, 0.748, 0.751, 0.753, 0.754, 0.752, 0.748, 0.742, 0.735, 0.720]

    ax6.plot(n_features_selected, model_performance, 'bo-', linewidth=2, markersize=8)

    # Mark optimal point
    optimal_idx = np.argmax(model_performance)
    ax6.plot(n_features_selected[optimal_idx], model_performance[optimal_idx],
             'ro', markersize=12, label=f'Optimal ({n_features_selected[optimal_idx]} features)')

    ax6.set_xlabel('Number of Features')
    ax6.set_ylabel('Model Performance (R²)')
    ax6.set_title('Feature Selection Performance')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Add annotations
    ax6.annotate(f'Best: {n_features_selected[optimal_idx]} features\nR² = {model_performance[optimal_idx]:.3f}',
                xy=(n_features_selected[optimal_idx], model_performance[optimal_idx]),
                xytext=(n_features_selected[optimal_idx]-2, model_performance[optimal_idx]-0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('feature_importance_analysis.pdf', bbox_inches='tight')
    print("Feature importance analysis saved as feature_importance_analysis.png/pdf")

    return fig

def create_data_exploration_charts():
    """Create data exploration and distribution charts"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Generate sample NHL data for visualization
    np.random.seed(42)

    # 1. PPG Distribution by Position
    positions = ['Attackers', 'Defensemen', 'Goalies']
    n_players = [400, 200, 80]

    # Different PPG distributions by position (using different scoring systems)
    attacker_ppg = np.random.gamma(2.2, 0.35, n_players[0])
    defensemen_ppg = np.random.gamma(1.8, 0.4, n_players[1])
    goalie_ppg = np.random.gamma(1.6, 0.45, n_players[2])

    position_data = [attacker_ppg, defensemen_ppg, goalie_ppg]

    # Create box plot with overlay violin plot
    bp = ax1.boxplot(position_data, positions=range(len(positions)), patch_artist=True,
                     boxprops=dict(alpha=0.7))
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xticks(range(len(positions)))
    ax1.set_xticklabels(positions)
    ax1.set_ylabel('Points Per Game (PPG)')
    ax1.set_title('PPG Distribution by Position')
    ax1.grid(True, alpha=0.3)

    # Add mean lines
    for i, data in enumerate(position_data):
        mean_val = np.mean(data)
        ax1.plot(i+1, mean_val, 'ro', markersize=8)
        ax1.text(i+1, mean_val + 0.1, f'μ={mean_val:.2f}', ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 2. Age vs Performance Analysis
    ages = np.random.normal(26, 4, 600)
    ages = np.clip(ages, 18, 40)

    # PPG tends to peak around 25-29 years old
    age_effect = -(ages - 27)**2 / 50 + 1.0
    base_ppg = 0.5 + age_effect + np.random.normal(0, 0.2, len(ages))
    base_ppg = np.clip(base_ppg, 0.1, 3.0)

    scatter = ax2.scatter(ages, base_ppg, alpha=0.6, s=30, c=ages, cmap='viridis')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('PPG')
    ax2.set_title('Age vs Performance Relationship')

    # Add trend line
    z = np.polyfit(ages, base_ppg, 2)  # Quadratic fit
    p = np.poly1d(z)
    age_range = np.linspace(18, 40, 100)
    ax2.plot(age_range, p(age_range), "r-", alpha=0.8, linewidth=3, label='Trend')
    ax2.legend()

    plt.colorbar(scatter, ax=ax2, label='Age')
    ax2.grid(True, alpha=0.3)

    # Mark peak performance age
    peak_age = age_range[np.argmax(p(age_range))]
    peak_ppg = np.max(p(age_range))
    ax2.plot(peak_age, peak_ppg, 'ro', markersize=10)
    ax2.annotate(f'Peak: {peak_age:.1f} years', xy=(peak_age, peak_ppg),
                xytext=(peak_age+3, peak_ppg+0.2),
                arrowprops=dict(arrowstyle='->', color='red'))

    # 3. Team Performance Impact
    team_performance = np.random.uniform(0.4, 0.7, 600)  # Team points percentage
    team_effect = (team_performance - 0.55) * 2  # Team effect on individual PPG
    individual_ppg = np.random.gamma(2, 0.4, 600) + team_effect
    individual_ppg = np.clip(individual_ppg, 0.1, 4.0)

    ax3.scatter(team_performance * 100, individual_ppg, alpha=0.6, s=30)
    ax3.set_xlabel('Team Performance (Points %)')
    ax3.set_ylabel('Individual PPG')
    ax3.set_title('Team Performance Impact on Individual Stats')

    # Add trend line
    z = np.polyfit(team_performance * 100, individual_ppg, 1)
    p = np.poly1d(z)
    team_range = np.linspace(40, 70, 100)
    ax3.plot(team_range, p(team_range), "r-", alpha=0.8, linewidth=3)

    ax3.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr_coef = np.corrcoef(team_performance, individual_ppg)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))

    # 4. Seasonal Consistency Analysis
    players = np.arange(1, 101)  # 100 sample players
    season1_ppg = np.random.gamma(2, 0.4, 100)
    season2_ppg = season1_ppg + np.random.normal(0, 0.15, 100)  # Some year-to-year variation
    season2_ppg = np.clip(season2_ppg, 0.1, 4.0)

    ax4.scatter(season1_ppg, season2_ppg, alpha=0.7, s=40)
    ax4.plot([0, 4], [0, 4], 'r--', linewidth=2, label='Perfect Consistency')
    ax4.set_xlabel('Season 1 PPG')
    ax4.set_ylabel('Season 2 PPG')
    ax4.set_title('Year-to-Year Performance Consistency')

    # Add trend line
    z = np.polyfit(season1_ppg, season2_ppg, 1)
    p = np.poly1d(z)
    ax4.plot(np.linspace(0, 4, 100), p(np.linspace(0, 4, 100)),
             "b-", alpha=0.8, linewidth=2, label='Actual Trend')

    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Calculate and display consistency metrics
    consistency_corr = np.corrcoef(season1_ppg, season2_ppg)[0, 1]
    mean_abs_diff = np.mean(np.abs(season1_ppg - season2_ppg))

    ax4.text(0.05, 0.95, f'Consistency (r): {consistency_corr:.3f}\nAvg Change: ±{mean_abs_diff:.2f}',
             transform=ax4.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('data_exploration_charts.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_exploration_charts.pdf', bbox_inches='tight')
    print("Data exploration charts saved as data_exploration_charts.png/pdf")

    return fig

if __name__ == "__main__":
    # Create feature importance analysis
    fig1 = create_feature_importance_analysis()

    # Create data exploration charts
    fig2 = create_data_exploration_charts()

    plt.show()