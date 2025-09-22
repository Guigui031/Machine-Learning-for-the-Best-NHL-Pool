#!/usr/bin/env python3
"""
Master Graph Generation Script
Generates all visualization graphs for the NHL Pool Optimization system.
"""

import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def generate_all_graphs():
    """Generate all graphs for the NHL Pool system documentation"""

    print("🏒 NHL Pool Optimization - Graph Generation")
    print("=" * 50)

    # Set matplotlib backend for non-interactive use
    plt.switch_backend('Agg')

    try:
        # 1. System Architecture
        print("📊 Generating system architecture diagram...")
        from create_system_architecture import create_system_architecture
        fig1 = create_system_architecture()
        plt.close(fig1)
        print("✅ System architecture diagram complete")

        # 2. Data Flow
        print("📊 Generating data flow visualization...")
        from create_data_flow import create_data_flow_diagram
        fig2 = create_data_flow_diagram()
        plt.close(fig2)
        print("✅ Data flow diagram complete")

        # 3. ML Pipeline
        print("📊 Generating ML pipeline flowchart...")
        from create_ml_pipeline import create_ml_pipeline_flowchart
        fig3 = create_ml_pipeline_flowchart()
        plt.close(fig3)
        print("✅ ML pipeline flowchart complete")

        # 4. Performance Analysis
        print("📊 Generating performance analysis charts...")
        from create_performance_analysis import create_sample_performance_charts, create_optimization_results_chart
        fig4 = create_sample_performance_charts()
        plt.close(fig4)
        fig5 = create_optimization_results_chart()
        plt.close(fig5)
        print("✅ Performance analysis charts complete")

        # 5. Optimization Comparison
        print("📊 Generating optimization comparison charts...")
        from create_optimization_comparison import create_optimization_comparison, create_constraint_analysis
        fig6 = create_optimization_comparison()
        plt.close(fig6)
        fig7 = create_constraint_analysis()
        plt.close(fig7)
        print("✅ Optimization comparison charts complete")

        # 6. Feature Importance
        print("📊 Generating feature importance analysis...")
        from create_feature_importance import create_feature_importance_analysis, create_data_exploration_charts
        fig8 = create_feature_importance_analysis()
        plt.close(fig8)
        fig9 = create_data_exploration_charts()
        plt.close(fig9)
        print("✅ Feature importance analysis complete")

        print("\n🎉 All graphs generated successfully!")
        print("\nGenerated files:")
        print("- system_architecture.png/pdf")
        print("- data_flow_diagram.png/pdf")
        print("- ml_pipeline_flowchart.png/pdf")
        print("- performance_analysis_charts.png/pdf")
        print("- optimization_results.png/pdf")
        print("- optimization_comparison.png/pdf")
        print("- constraint_analysis.png/pdf")
        print("- feature_importance_analysis.png/pdf")
        print("- data_exploration_charts.png/pdf")

        print(f"\n📁 All files saved to: {current_dir}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install matplotlib numpy pandas seaborn")
        return False

    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
        return False

    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['matplotlib', 'numpy', 'pandas', 'seaborn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False

    return True

if __name__ == "__main__":
    print("Checking dependencies...")
    if check_dependencies():
        print("✅ All dependencies available")
        generate_all_graphs()
    else:
        print("❌ Please install missing dependencies first")
        sys.exit(1)