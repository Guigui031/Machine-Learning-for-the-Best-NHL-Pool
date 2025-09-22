#!/usr/bin/env python3
"""
Data Flow Visualization
Generates data flow diagrams showing how data moves through the NHL Pool system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np

def create_data_flow_diagram():
    """Create comprehensive data flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    # Define colors
    colors = {
        'input': '#BBDEFB',     # Light blue
        'process': '#FFF9C4',   # Light yellow
        'storage': '#C8E6C9',   # Light green
        'ml': '#E1BEE7',        # Light purple
        'output': '#FFCDD2'     # Light red
    }

    # Data flow components with detailed information
    components = {
        # Input Sources
        'nhl_api': {'pos': (1, 10), 'size': (2, 1.5), 'color': colors['input'],
                   'text': 'NHL API\n• api-web.nhle.com\n• Player stats\n• Team rosters'},
        'salary_source': {'pos': (4, 10), 'size': (2, 1.5), 'color': colors['input'],
                         'text': 'Salary Data\n• External TSV\n• Contract info\n• Cap hits'},

        # Raw Data Processing
        'download_players': {'pos': (0.5, 8), 'size': (2, 1), 'color': colors['process'],
                            'text': 'download_players_points()'},
        'download_teams': {'pos': (3, 8), 'size': (2, 1), 'color': colors['process'],
                          'text': 'download_teams_roster()'},
        'download_standings': {'pos': (5.5, 8), 'size': (2, 1), 'color': colors['process'],
                              'text': 'download_season_standing()'},

        # Raw Storage
        'json_storage': {'pos': (8, 8.5), 'size': (3, 2), 'color': colors['storage'],
                        'text': 'JSON Files\n• {season}_players_points.json\n• teams/{abbrev}.json\n• {season}_standings.json'},

        # Data Processing
        'load_player': {'pos': (1, 6), 'size': (2.5, 1), 'color': colors['process'],
                       'text': 'load_player()\n• Biographical data\n• Multi-season stats'},
        'process_skaters': {'pos': (4, 6), 'size': (2.5, 1), 'color': colors['process'],
                           'text': 'process_data_skaters()\n• Normalize by games\n• Rate statistics'},

        # Structured Data
        'player_objects': {'pos': (8, 6), 'size': (2, 1.5), 'color': colors['storage'],
                          'text': 'Player Objects\n• Multi-season data\n• Normalized stats\n• Salary info'},
        'training_data': {'pos': (11, 6), 'size': (2, 1.5), 'color': colors['storage'],
                         'text': 'Training Dataset\n• Feature matrix\n• Target values\n• Sliding window'},

        # Feature Engineering
        'feature_extraction': {'pos': (2, 4), 'size': (3, 1), 'color': colors['process'],
                              'text': 'Feature Engineering\n• Historical performance\n• Player attributes\n• Team context'},

        # Machine Learning Pipeline
        'hyperparameter_tuning': {'pos': (6, 4), 'size': (2.5, 1), 'color': colors['ml'],
                                 'text': 'Hyperparameter Tuning\n• RandomizedSearchCV\n• 5-fold CV'},
        'ensemble_training': {'pos': (9, 4), 'size': (2.5, 1), 'color': colors['ml'],
                             'text': 'Ensemble Training\n• XGBoost, SVR, SGD\n• VotingRegressor'},

        # Predictions
        'ml_predictions': {'pos': (12, 4), 'size': (2, 1.5), 'color': colors['ml'],
                          'text': 'ML Predictions\n• PPG forecasts\n• Player rankings'},

        # Optimization
        'constraint_setup': {'pos': (3, 2), 'size': (2.5, 1), 'color': colors['process'],
                            'text': 'Constraint Setup\n• Salary budget\n• Position limits'},
        'lp_solver': {'pos': (6, 2), 'size': (2, 1), 'color': colors['process'],
                     'text': 'LP Solver\n• PuLP library\n• Optimal solution'},
        'branch_bound': {'pos': (8.5, 2), 'size': (2, 1), 'color': colors['process'],
                        'text': 'Branch & Bound\n• Custom algorithm\n• Heuristic solution'},

        # Final Output
        'optimal_team': {'pos': (11.5, 2), 'size': (2.5, 1.5), 'color': colors['output'],
                        'text': 'Optimal Team\n• 20 players\n• Role distribution\n• Total PPG'},
        'results_export': {'pos': (15, 2), 'size': (2, 1.5), 'color': colors['output'],
                          'text': 'Results Export\n• CSV files\n• Analysis\n• Visualizations'}
    }

    # Draw components
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']

        # Create rounded rectangle
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.05",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)

        # Add text
        ax.text(x + w/2, y + h/2, comp['text'],
                ha='center', va='center',
                fontsize=8, fontweight='normal')

    # Define data flow arrows with labels
    flows = [
        # Data collection flows
        {'from': 'nhl_api', 'to': 'download_players', 'label': 'API calls'},
        {'from': 'nhl_api', 'to': 'download_teams', 'label': 'Team data'},
        {'from': 'nhl_api', 'to': 'download_standings', 'label': 'Standings'},
        {'from': 'salary_source', 'to': 'load_player', 'label': 'Salary lookup'},

        # Storage flows
        {'from': 'download_players', 'to': 'json_storage', 'label': 'JSON'},
        {'from': 'download_teams', 'to': 'json_storage', 'label': 'JSON'},
        {'from': 'download_standings', 'to': 'json_storage', 'label': 'JSON'},

        # Processing flows
        {'from': 'json_storage', 'to': 'load_player', 'label': 'Raw data'},
        {'from': 'load_player', 'to': 'process_skaters', 'label': 'Player data'},
        {'from': 'process_skaters', 'to': 'player_objects', 'label': 'Processed'},
        {'from': 'player_objects', 'to': 'training_data', 'label': 'Features'},

        # ML pipeline flows
        {'from': 'training_data', 'to': 'feature_extraction', 'label': 'Raw features'},
        {'from': 'feature_extraction', 'to': 'hyperparameter_tuning', 'label': 'Engineered'},
        {'from': 'hyperparameter_tuning', 'to': 'ensemble_training', 'label': 'Best params'},
        {'from': 'ensemble_training', 'to': 'ml_predictions', 'label': 'Trained model'},

        # Optimization flows
        {'from': 'ml_predictions', 'to': 'constraint_setup', 'label': 'PPG predictions'},
        {'from': 'constraint_setup', 'to': 'lp_solver', 'label': 'Problem setup'},
        {'from': 'constraint_setup', 'to': 'branch_bound', 'label': 'Problem setup'},
        {'from': 'lp_solver', 'to': 'optimal_team', 'label': 'LP solution'},
        {'from': 'branch_bound', 'to': 'optimal_team', 'label': 'B&B solution'},
        {'from': 'optimal_team', 'to': 'results_export', 'label': 'Team data'},
    ]

    # Draw arrows
    for flow in flows:
        from_comp = components[flow['from']]
        to_comp = components[flow['to']]

        # Calculate connection points
        from_center_x = from_comp['pos'][0] + from_comp['size'][0] / 2
        from_center_y = from_comp['pos'][1] + from_comp['size'][1] / 2
        to_center_x = to_comp['pos'][0] + to_comp['size'][0] / 2
        to_center_y = to_comp['pos'][1] + to_comp['size'][1] / 2

        # Determine connection points (closest edges)
        if from_center_x < to_center_x:  # Arrow goes right
            start_x = from_comp['pos'][0] + from_comp['size'][0]
            end_x = to_comp['pos'][0]
        else:  # Arrow goes left
            start_x = from_comp['pos'][0]
            end_x = to_comp['pos'][0] + to_comp['size'][0]

        if from_center_y > to_center_y:  # Arrow goes down
            start_y = from_comp['pos'][1]
            end_y = to_comp['pos'][1] + to_comp['size'][1]
        else:  # Arrow goes up
            start_y = from_comp['pos'][1] + from_comp['size'][1]
            end_y = to_comp['pos'][1]

        # Special cases for better routing
        if abs(from_center_y - to_center_y) < 0.5:  # Horizontal flow
            start_y = from_center_y
            end_y = to_center_y

        arrow = ConnectionPatch((start_x, start_y), (end_x, end_y),
                               "data", "data",
                               arrowstyle="->",
                               shrinkA=2, shrinkB=2,
                               mutation_scale=15,
                               fc="navy",
                               ec="navy",
                               linewidth=1.5,
                               alpha=0.7)
        ax.add_patch(arrow)

        # Add flow label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y, flow['label'],
                ha='center', va='center',
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2",
                         facecolor='white',
                         alpha=0.8,
                         edgecolor='none'),
                rotation=0 if abs(start_x - end_x) > abs(start_y - end_y) else 90)

    # Add process stage labels
    stage_labels = [
        {'text': 'DATA COLLECTION', 'pos': (0, 11.5), 'color': colors['input']},
        {'text': 'DATA PROCESSING', 'pos': (0, 7), 'color': colors['process']},
        {'text': 'MACHINE LEARNING', 'pos': (0, 5), 'color': colors['ml']},
        {'text': 'OPTIMIZATION', 'pos': (0, 3), 'color': colors['process']},
        {'text': 'RESULTS', 'pos': (0, 0.5), 'color': colors['output']}
    ]

    for label in stage_labels:
        ax.text(label['pos'][0], label['pos'][1], label['text'],
                ha='left', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4",
                         facecolor=label['color'],
                         alpha=0.8))

    # Set axis properties
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Add title
    plt.title('NHL Pool Optimization Data Flow Diagram',
              fontsize=20, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Data Sources'),
        mpatches.Patch(color=colors['process'], label='Processing'),
        mpatches.Patch(color=colors['storage'], label='Data Storage'),
        mpatches.Patch(color=colors['ml'], label='Machine Learning'),
        mpatches.Patch(color=colors['output'], label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.98, 0.95), fontsize=10)

    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_flow_diagram.pdf', bbox_inches='tight')
    print("Data flow diagram saved as data_flow_diagram.png/pdf")

    return fig

if __name__ == "__main__":
    create_data_flow_diagram()
    plt.show()