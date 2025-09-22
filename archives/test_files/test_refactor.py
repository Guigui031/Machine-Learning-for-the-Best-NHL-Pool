#!/usr/bin/env python3
"""
Quick test to verify refactored codebase still works
"""

def test_core_imports():
    """Test that core modules can be imported"""
    print("Testing core imports...")

    try:
        # Core data models
        import player
        import team
        print("- Core models: OK")

        # Data pipeline
        import data_download
        import process_data
        import data_validation
        import config
        print("- Data pipeline: OK")

        # Optimization
        import pool_classifier
        print("- Optimization: OK")

        # ML (might fail if dependencies not installed)
        try:
            import ensemble_learning
            print("- ML ensemble: OK")
        except ImportError as e:
            print(f"- ML ensemble: SKIP (missing dependency: {e})")

        try:
            import model_predictor
            print("- Model predictor: OK")
        except ImportError as e:
            print(f"- Model predictor: SKIP (missing dependency: {e})")

        print("\nCore imports test: PASSED")
        return True

    except ImportError as e:
        print(f"Core imports test: FAILED - {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core classes"""
    print("\nTesting basic functionality...")

    try:
        # Test Player class
        from player import Player
        player = Player("test_id")
        player.set_name("Test", "Player")
        player.set_position("C")
        assert player.name == "Test Player"
        assert player.role == "A"  # Center -> Attacker
        print("- Player class: OK")

        # Test Team class
        from team import Team
        team = Team("Test Team", "TT", "20232024", 0.650)
        assert team.team_name == "Test Team"
        print("- Team class: OK")

        print("Basic functionality test: PASSED")
        return True

    except Exception as e:
        print(f"Basic functionality test: FAILED - {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("NHL Pool Optimization - Refactor Test")
    print("=" * 50)

    import_success = test_core_imports()
    func_success = test_basic_functionality()

    print("\n" + "=" * 50)
    if import_success and func_success:
        print("OVERALL: All tests PASSED - Refactor successful!")
    else:
        print("OVERALL: Some tests FAILED - Check issues above")
    print("=" * 50)

if __name__ == "__main__":
    main()