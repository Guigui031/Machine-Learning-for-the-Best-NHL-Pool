#!/usr/bin/env python3
"""
Simple test for the new NHL data collection structure
Run with: conda activate ift3150 && python test_simple.py
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test if all modules import correctly"""
    print("Testing imports...")
    
    try:
        from src.config.settings import API_CONFIG, DATA_CONFIG
        print("Config imports working")
        
        from src.utils.api import APIClient
        print("API client imports working")
        
        from src.data.collectors.games import GameCollector
        print("Game collector imports working")
        
        from src.data.models.game import GameData, PlayerGameStats
        print("Data models imports working")
        
        return True
        
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def test_config():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    try:
        from src.config.settings import API_CONFIG, DATA_CONFIG
        
        print(f"API Base URL: {API_CONFIG.base_url}")
        print(f"Data Path: {DATA_CONFIG.base_path}")
        print(f"Seasons: {DATA_CONFIG.seasons}")
        print(f"Position limits: {DATA_CONFIG.positions}")
        print("Configuration looks good")
        return True
        
    except Exception as e:
        print(f"Config test failed: {e}")
        return False

def test_api_client():
    """Test basic API client functionality"""
    print("\nTesting API client...")
    
    try:
        from src.utils.api import APIClient
        
        with APIClient() as client:
            print(f"API client created successfully")
            print(f"Base URL: {client.base_url}")
            return True
            
    except Exception as e:
        print(f"API client test failed: {e}")
        return False

def test_data_models():
    """Test data model creation"""
    print("\nTesting data models...")
    
    try:
        from src.data.models.game import PlayerGameStats, GameData
        from datetime import datetime
        
        # Test PlayerGameStats
        player_stats = PlayerGameStats(
            player_id="8478402",
            player_name="Connor McDavid", 
            position="C",
            team="EDM",
            goals=2,
            assists=1,
            shots=6,
            ice_time=1200  # 20 minutes in seconds
        )
        
        print(f"Created player stats: {player_stats.player_name}")
        print(f"  Points: {player_stats.points}")
        print(f"  Ice time: {player_stats.ice_time_minutes:.1f} min")
        
        # Test GameData  
        game_data = GameData(
            game_id="2023020001",
            date=datetime.now(),
            season="20232024",
            game_type="regular",
            home_team="TOR",
            away_team="EDM", 
            home_score=3,
            away_score=4
        )
        
        print(f"Created game data: {game_data.away_team} @ {game_data.home_team}")
        print(f"  Winner: {game_data.winning_team}")
        
        return True
        
    except Exception as e:
        print(f"Data model test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("Testing NHL Data Collection Structure")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config, 
        test_api_client,
        test_data_models
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All basic tests passed! Ready for API testing.")
        return True
    else:
        print("Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)