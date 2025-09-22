#!/usr/bin/env python3
"""
System Architecture Visualization
Generates a comprehensive system architecture diagram for the NHL Pool Optimization system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_architecture():
    """Create system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Define colors
    colors = {
        'data': '#E3F2FD',      # Light blue
        'processing': '#FFF3E0', # Light orange
        'ml': '#F3E5F5',        # Light purple
        'optimization': '#E8F5E8', # Light green
        'output': '#FFEBEE'     # Light red
    }

    # Define component positions and sizes
    components = {
        # Data Layer
        'nhl_api': {'pos': (2, 9), 'size': (2.5, 1), 'color': colors['data'], 'text': 'NHL API\n• Player Stats\n• Team Rosters\n• Standings'},
        'salary_data': {'pos': (5.5, 9), 'size': (2, 1), 'color': colors['data'], 'text': 'Salary Data\n• Player Salaries\n• Contract Info'},

        # Data Download Layer
        'data_download': {'pos': (1, 7), 'size': (7, 1.2), 'color': colors['processing'], 'text': 'Data Download Module (data_download.py)\n• API Integration • Caching • Error Handling • Batch Downloads'},

        # Data Storage
        'json_files': {'pos': (9, 8.5), 'size': (3, 2), 'color': colors['data'], 'text': 'JSON Data Files\n• Season Data\n• Team Rosters\n• Player Profiles'},

        # Data Processing Layer
        'data_processing': {'pos': (1, 5), 'size': (7, 1.2), 'color': colors['processing'], 'text': 'Data Processing Module (process_data.py)\n• Data Validation • Normalization • Feature Engineering'},

        # Core Classes
        'player_class': {'pos': (9, 5.5), 'size': (1.4, 1.5), 'color': colors['processing'], 'text': 'Player\nClass'},
        'season_class': {'pos': (10.6, 5.5), 'size': (1.4, 1.5), 'color': colors['processing'], 'text': 'Season\nClass'},

        # Feature Engineering
        'features': {'pos': (3, 3), 'size': (3, 1), 'color': colors['ml'], 'text': 'Feature Engineering\n• Historical Performance\n• Player Attributes'},

        # Machine Learning Layer
        'ensemble_ml': {'pos': (1, 1), 'size': (7, 1.5), 'color': colors['ml'], 'text': 'Ensemble Learning Module (ensemble_learning.py)\n• XGBoost • SVR • SGD • Logistic Regression\n• Hyperparameter Tuning • Cross-Validation'},

        # Optimization Layer
        'optimization': {'pos': (9, 1), 'size': (3, 1.5), 'color': colors['optimization'], 'text': 'Optimization Module\n(pool_classifier.py)\n• Linear Programming\n• Branch & Bound'},

        # Output
        'results': {'pos': (13, 4), 'size': (2.5, 3), 'color': colors['output'], 'text': 'Results\n• Optimal Team\n• Performance\n  Metrics\n• CSV Export\n• Analysis'}
    }

    # Draw components
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']

        # Create fancy box
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)

        # Add text
        ax.text(x + w/2, y + h/2, comp['text'],
                ha='center', va='center',
                fontsize=9, fontweight='bold' if 'Module' in comp['text'] else 'normal',
                wrap=True)

    # Define arrows (connections)
    arrows = [
        # Data flow
        ('nhl_api', 'data_download'),
        ('salary_data', 'data_download'),
        ('data_download', 'json_files'),
        ('json_files', 'data_processing'),
        ('data_processing', 'player_class'),
        ('data_processing', 'season_class'),
        ('player_class', 'features'),
        ('season_class', 'features'),
        ('features', 'ensemble_ml'),
        ('ensemble_ml', 'optimization'),
        ('optimization', 'results'),
    ]

    # Draw arrows
    for start, end in arrows:
        start_comp = components[start]
        end_comp = components[end]

        # Calculate connection points
        start_x = start_comp['pos'][0] + start_comp['size'][0] / 2
        start_y = start_comp['pos'][1]
        end_x = end_comp['pos'][0] + end_comp['size'][0] / 2
        end_y = end_comp['pos'][1] + end_comp['size'][1]

        # Special cases for horizontal connections
        if start == 'data_processing' and end in ['player_class', 'season_class']:
            start_x = start_comp['pos'][0] + start_comp['size'][0]
            start_y = start_comp['pos'][1] + start_comp['size'][1] / 2
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['size'][1] / 2
        elif start == 'ensemble_ml' and end == 'optimization':
            start_x = start_comp['pos'][0] + start_comp['size'][0]
            start_y = start_comp['pos'][1] + start_comp['size'][1] / 2
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['size'][1] / 2

        arrow = ConnectionPatch((start_x, start_y), (end_x, end_y),
                               "data", "data",
                               arrowstyle="->",
                               shrinkA=5, shrinkB=5,
                               mutation_scale=20,
                               fc="darkblue",
                               ec="darkblue",
                               linewidth=2)
        ax.add_patch(arrow)

    # Add layer labels
    layer_labels = [
        {'text': 'Data Sources', 'pos': (0.5, 9.5), 'color': colors['data']},
        {'text': 'Data Collection', 'pos': (0.5, 7.6), 'color': colors['processing']},
        {'text': 'Data Processing', 'pos': (0.5, 5.6), 'color': colors['processing']},
        {'text': 'Machine Learning', 'pos': (0.5, 2.2), 'color': colors['ml']},
        {'text': 'Optimization', 'pos': (8.5, 2.2), 'color': colors['optimization']},
    ]

    for label in layer_labels:
        ax.text(label['pos'][0], label['pos'][1], label['text'],
                ha='left', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor=label['color'],
                         alpha=0.7))

    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title
    plt.title('NHL Pool Optimization System Architecture',
              fontsize=18, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['data'], label='Data Layer'),
        mpatches.Patch(color=colors['processing'], label='Processing Layer'),
        mpatches.Patch(color=colors['ml'], label='Machine Learning'),
        mpatches.Patch(color=colors['optimization'], label='Optimization'),
        mpatches.Patch(color=colors['output'], label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('system_architecture.pdf', bbox_inches='tight')
    print("System architecture diagram saved as system_architecture.png/pdf")

    return fig

if __name__ == "__main__":
    create_system_architecture()
    plt.show()