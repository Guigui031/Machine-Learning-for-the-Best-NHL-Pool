#!/usr/bin/env python3
"""
Optimization Algorithm Comparison Visualization
Generates detailed comparisons between different optimization approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def create_optimization_comparison():
    """Create comprehensive optimization algorithm comparison"""

    fig = plt.figure(figsize=(20, 16))

    # Sample data for algorithm comparison
    np.random.seed(42)

    # Algorithm performance data
    algorithms = ['Linear Programming\n(PuLP)', 'Branch & Bound\n(Custom)', 'Greedy Heuristic\n(PPG/Salary)', 'Random Selection\n(Baseline)']
    performance_data = {
        'Total PPG': [247.3, 245.8, 241.2, 198.5],
        'Solve Time (s)': [0.15, 67.4, 0.08, 0.01],
        'Memory Usage (MB)': [45, 120, 25, 10],
        'Optimality Gap (%)': [0.0, 0.6, 2.9, 19.8]
    }

    # 1. Algorithm Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(algorithms))
    bars = ax1.bar(x_pos, performance_data['Total PPG'],
                   color=['green', 'blue', 'orange', 'red'], alpha=0.8)

    ax1.set_ylabel('Total Team PPG')
    ax1.set_title('Optimization Algorithm Performance\n(Higher is Better)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')

    # Add value labels
    for bar, value in zip(bars, performance_data['Total PPG']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # Add optimality indicator
    ax1.axhline(y=performance_data['Total PPG'][0], color='green',
                linestyle='--', alpha=0.7, label='Optimal Solution')
    ax1.legend()

    # 2. Computational Efficiency (Log Scale)
    ax2 = plt.subplot(2, 3, 2)

    # Create dual axis for time and memory
    bars_time = ax2.bar([x - 0.2 for x in x_pos], performance_data['Solve Time (s)'],
                        width=0.4, label='Solve Time (s)', alpha=0.8)

    ax2_twin = ax2.twinx()
    bars_mem = ax2_twin.bar([x + 0.2 for x in x_pos], performance_data['Memory Usage (MB)'],
                           width=0.4, label='Memory (MB)', alpha=0.8, color='orange')

    ax2.set_ylabel('Solve Time (seconds)', color='blue')
    ax2_twin.set_ylabel('Memory Usage (MB)', color='orange')
    ax2.set_title('Computational Efficiency')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_yscale('log')

    # Add value labels
    for i, (bar_time, bar_mem) in enumerate(zip(bars_time, bars_mem)):
        ax2.text(bar_time.get_x() + bar_time.get_width()/2, bar_time.get_height() * 1.5,
                f'{performance_data["Solve Time (s)"][i]:.2f}s',
                ha='center', va='bottom', fontsize=8, rotation=90)
        ax2_twin.text(bar_mem.get_x() + bar_mem.get_width()/2, bar_mem.get_height() + 5,
                     f'{performance_data["Memory Usage (MB)"][i]}MB',
                     ha='center', va='bottom', fontsize=8)

    # 3. Optimality Gap Analysis
    ax3 = plt.subplot(2, 3, 3)
    colors = ['green', 'lightgreen', 'yellow', 'red']
    bars = ax3.bar(x_pos, performance_data['Optimality Gap (%)'], color=colors, alpha=0.8)

    ax3.set_ylabel('Optimality Gap (%)')
    ax3.set_title('Solution Quality\n(Lower is Better)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')

    for bar, value in zip(bars, performance_data['Optimality Gap (%)']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add quality threshold lines
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Acceptable (5%)')
    ax3.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Excellent (1%)')
    ax3.legend()

    # 4. Branch and Bound Progress Visualization
    ax4 = plt.subplot(2, 3, 4)

    # Simulate branch and bound progress
    iterations = np.arange(0, 100, 1)
    upper_bound = 250 * np.exp(-iterations/30) + 245.8
    lower_bound = 200 + (245.8 - 200) * (1 - np.exp(-iterations/20))

    ax4.plot(iterations, upper_bound, 'r-', linewidth=2, label='Upper Bound', alpha=0.8)
    ax4.plot(iterations, lower_bound, 'b-', linewidth=2, label='Lower Bound', alpha=0.8)
    ax4.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, color='gray', label='Gap')

    # Mark convergence
    convergence_point = 75
    ax4.axvline(x=convergence_point, color='green', linestyle='--',
                label=f'Convergence (iter {convergence_point})')

    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Objective Value (PPG)')
    ax4.set_title('Branch & Bound Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Salary Cap Utilization Comparison
    ax5 = plt.subplot(2, 3, 5)

    salary_utilization = [96.8, 96.2, 94.1, 78.3]  # Percentage of salary cap used
    efficiency = [p/s*100 for p, s in zip(performance_data['Total PPG'], salary_utilization)]

    # Create scatter plot with size indicating efficiency
    scatter = ax5.scatter(salary_utilization, performance_data['Total PPG'],
                         s=[e*3 for e in efficiency], alpha=0.7,
                         c=['green', 'blue', 'orange', 'red'])

    # Add algorithm labels
    for i, alg in enumerate(['LP', 'B&B', 'Greedy', 'Random']):
        ax5.annotate(alg, (salary_utilization[i], performance_data['Total PPG'][i]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax5.set_xlabel('Salary Cap Utilization (%)')
    ax5.set_ylabel('Total Team PPG')
    ax5.set_title('Efficiency: Performance vs Budget Usage')
    ax5.grid(True, alpha=0.3)

    # Add efficiency legend
    ax5.text(0.02, 0.98, 'Bubble size = PPG per $ efficiency',
             transform=ax5.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 6. Algorithm Decision Matrix
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Create decision matrix
    criteria = ['Optimality', 'Speed', 'Memory', 'Scalability', 'Interpretability']
    scores = {
        'Linear Programming': [5, 4, 4, 5, 3],
        'Branch & Bound': [4, 2, 3, 3, 5],
        'Greedy Heuristic': [3, 5, 5, 4, 4],
        'Random Baseline': [1, 5, 5, 5, 2]
    }

    # Create heatmap data
    heatmap_data = np.array([scores[alg] for alg in scores.keys()])

    # Create custom heatmap
    im = ax6.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

    # Set ticks and labels
    ax6.set_xticks(np.arange(len(criteria)))
    ax6.set_yticks(np.arange(len(scores)))
    ax6.set_xticklabels(criteria, rotation=45, ha='right')
    ax6.set_yticklabels(scores.keys())

    # Add text annotations
    for i in range(len(scores)):
        for j in range(len(criteria)):
            text = ax6.text(j, i, heatmap_data[i, j],
                           ha="center", va="center", color="black", fontweight='bold')

    ax6.set_title('Algorithm Comparison Matrix\n(1=Poor, 5=Excellent)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label('Score (1-5)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('optimization_comparison.pdf', bbox_inches='tight')
    print("Optimization comparison chart saved as optimization_comparison.png/pdf")

    return fig

def create_constraint_analysis():
    """Create constraint analysis visualization"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Budget Constraint Analysis
    budgets = np.linspace(60, 100, 20)  # Budget range in millions
    optimal_ppg = []

    # Simulate optimal PPG for different budgets
    base_ppg = 180
    for budget in budgets:
        if budget < 70:
            ppg = base_ppg + (budget - 60) * 3
        elif budget < 88:
            ppg = base_ppg + 30 + (budget - 70) * 1.5
        else:
            ppg = base_ppg + 57 + (budget - 88) * 0.5
        optimal_ppg.append(ppg)

    ax1.plot(budgets, optimal_ppg, 'b-', linewidth=3, label='Optimal PPG')
    ax1.axvline(x=88, color='red', linestyle='--', linewidth=2, label='Salary Cap ($88M)')
    ax1.fill_between(budgets, optimal_ppg, alpha=0.3)

    ax1.set_xlabel('Budget (Millions $)')
    ax1.set_ylabel('Optimal Team PPG')
    ax1.set_title('Budget Impact on Team Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark current operating point
    current_budget = 88
    current_ppg = optimal_ppg[np.argmin(np.abs(budgets - current_budget))]
    ax1.plot(current_budget, current_ppg, 'ro', markersize=10, label='Current')

    # 2. Position Constraint Impact
    positions = ['Attackers', 'Defensemen', 'Goalies']
    current_limits = [12, 6, 2]
    min_limits = [10, 4, 2]
    max_limits = [14, 8, 2]

    # PPG impact of changing position limits
    ppg_impact_min = [242.1, 244.5, 247.3]  # PPG if we use minimum players
    ppg_impact_current = [247.3, 247.3, 247.3]  # Current optimal
    ppg_impact_max = [249.1, 246.8, 247.3]  # PPG if we use maximum players

    x = np.arange(len(positions))
    width = 0.25

    bars1 = ax2.bar(x - width, ppg_impact_min, width, label='Minimum Limits', alpha=0.8)
    bars2 = ax2.bar(x, ppg_impact_current, width, label='Current Limits', alpha=0.8)
    bars3 = ax2.bar(x + width, ppg_impact_max, width, label='Maximum Limits', alpha=0.8)

    ax2.set_xlabel('Position')
    ax2.set_ylabel('Team PPG')
    ax2.set_title('Position Limit Impact on Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()

    # Add limit labels
    for i, (pos, curr, min_lim, max_lim) in enumerate(zip(positions, current_limits, min_limits, max_limits)):
        ax2.text(i, 240, f'{min_lim}-{curr}-{max_lim}', ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # 3. Feasible Region Visualization (2D projection)
    # Simulate attackers vs defensemen trade-off with fixed goalies
    attackers = np.arange(8, 16)
    max_defensemen = []

    for att in attackers:
        # Remaining budget after attackers and 2 goalies (simplified)
        remaining_spots = 20 - att - 2
        max_def = min(remaining_spots, 8)  # Can't exceed reasonable defensemen limit
        max_defensemen.append(max_def)

    ax3.fill_between(attackers, 0, max_defensemen, alpha=0.3, color='lightblue', label='Feasible Region')
    ax3.plot(attackers, max_defensemen, 'b-', linewidth=2, label='Budget Constraint')

    # Mark current solution
    ax3.plot(12, 6, 'ro', markersize=10, label='Current Optimal (12A, 6D)')

    ax3.set_xlabel('Number of Attackers')
    ax3.set_ylabel('Number of Defensemen')
    ax3.set_title('Feasible Region (Attackers vs Defensemen)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(7, 16)
    ax3.set_ylim(0, 10)

    # 4. Constraint Sensitivity Analysis
    constraints = ['Budget\n($88M)', 'Attackers\n(≤12)', 'Defensemen\n(≤6)', 'Goalies\n(≤2)']
    shadow_prices = [0.85, 0.45, 0.12, 0.03]  # PPG improvement per unit constraint relaxation

    bars = ax4.bar(constraints, shadow_prices, color=['red', 'blue', 'green', 'orange'], alpha=0.8)

    ax4.set_ylabel('Shadow Price (PPG per unit)')
    ax4.set_title('Constraint Sensitivity Analysis\n(Which constraints limit performance most)')

    for bar, value in zip(bars, shadow_prices):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Add interpretation
    ax4.text(0.02, 0.98, 'Higher values indicate more\nlimiting constraints',
             transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('constraint_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('constraint_analysis.pdf', bbox_inches='tight')
    print("Constraint analysis chart saved as constraint_analysis.png/pdf")

    return fig

if __name__ == "__main__":
    # Create optimization comparison
    fig1 = create_optimization_comparison()

    # Create constraint analysis
    fig2 = create_constraint_analysis()

    plt.show()