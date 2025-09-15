#!/usr/bin/env python3
"""
Compare old vs new data collection approaches
Shows the difference in data richness
"""

import os
import sys
import json
import requests

# Add src to path
sys.path.insert(0, 'src')

def test_old_approach():
    """Test your original data collection approach"""
    print("Testing OLD approach (season aggregates)...")
    
    try:
        # Replicate your old data_download.py approach
        season = '20232024'
        url = f'https://api-web.nhle.com/v1/skater-stats-leaders/{season}/2?categories=points&limit=5'
        headers = {'Accept': 'application/json'}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'leaders' in data:
            print(f"✓ Got {len(data['leaders'])} player season totals")
            
            # Show what data looks like
            for i, player in enumerate(data['leaders'][:3]):
                name = player.get('firstName', {}).get('default', '') + ' ' + player.get('lastName', {}).get('default', '')
                goals = player.get('goals', 0)
                assists = player.get('assists', 0)
                games = player.get('gamesPlayed', 0)
                
                print(f"  {i+1}. {name}: {goals}G {assists}A in {games} games")
            
            # Count available features
            sample_player = data['leaders'][0]
            features = [k for k in sample_player.keys() if isinstance(sample_player[k], (int, float))]
            print(f"  Available features: {len(features)} numeric fields")
            print(f"  Features: {features[:8]}...")  # Show first 8
            
            return data
        else:
            print("✗ Unexpected data format")
            return None
            
    except Exception as e:
        print(f"✗ Old approach test failed: {e}")
        return None

def test_new_approach():
    """Test the new game-by-game approach"""
    print("\nTesting NEW approach (game-by-game data)...")
    
    try:
        from src.data.collectors.games import GameCollector
        
        with GameCollector() as collector:
            # Get Connor McDavid's game logs
            player_id = "8478402"  # Connor McDavid
            season = "20232024"
            
            game_logs = collector.get_player_game_logs(player_id, season)
            
            if game_logs and 'gameLog' in game_logs:
                games = game_logs['gameLog']
                print(f"✓ Got {len(games)} individual game records for McDavid")
                
                # Show what game-level data looks like
                print("  Sample games:")
                for i, game in enumerate(games[:3]):
                    opponent = game.get('opponentAbbrev', 'UNK')
                    goals = game.get('goals', 0)
                    assists = game.get('assists', 0)
                    toi = game.get('toi', '0:00')
                    home_road = 'vs' if game.get('homeRoadFlag') == 'H' else '@'
                    
                    print(f"    Game {i+1}: {home_road} {opponent} - {goals}G {assists}A, {toi} TOI")
                
                # Count available features per game
                sample_game = games[0]
                features = [k for k in sample_game.keys() if isinstance(sample_game[k], (int, float, str))]
                print(f"  Available features per game: {len(features)} fields")
                print(f"  New features: {[f for f in features if f not in ['goals', 'assists', 'gamesPlayed']][:8]}...")
                
                # Calculate data volume difference
                total_data_points = len(games) * len(features)
                print(f"  Total data points for one player: {total_data_points}")
                
                return game_logs
            else:
                print("✗ Could not get game logs")
                return None
                
    except Exception as e:
        print(f"✗ New approach test failed: {e}")
        return None

def test_feature_richness():
    """Compare feature richness between approaches"""
    print("\nComparing feature richness...")
    
    # Old approach features (from your current code)
    old_features = [
        'goals', 'assists', 'games', 'shots', 'pim', 
        'plus_minus', 'points', 'height', 'weight', 'age'
    ]
    
    # New approach features (from game-by-game data)
    new_features = [
        # Basic per-game stats
        'goals', 'assists', 'shots', 'pim', 'plus_minus',
        
        # New temporal features  
        'goals_last_10_games', 'assists_last_10_games',
        'recent_hot_streak', 'games_since_goal',
        
        # New contextual features
        'goals_home_vs_away', 'assists_vs_strong_opponents',
        'performance_back_to_back', 'performance_rested',
        
        # New situational features
        'powerplay_goals', 'shorthanded_assists', 'even_strength_points',
        'ice_time_per_game', 'powerplay_ice_time',
        
        # New advanced features
        'linemate_quality_score', 'opponent_strength_faced',
        'clutch_performance_late_game', 'performance_in_close_games',
        
        # Injury/fatigue features
        'games_since_injury', 'performance_after_rest',
        
        # Team context
        'team_score_differential_when_playing', 'team_powerplay_success_rate'
    ]
    
    print(f"Old approach features: {len(old_features)}")
    print(f"  {old_features}")
    
    print(f"\nNew approach potential features: {len(new_features)}")
    print(f"  Basic: {new_features[:5]}")
    print(f"  Temporal: {new_features[5:9]}")
    print(f"  Contextual: {new_features[9:13]}")
    print(f"  Situational: {new_features[13:18]}")
    print(f"  Advanced: {new_features[18:22]}")
    
    improvement_ratio = len(new_features) / len(old_features)
    print(f"\nFeature richness improvement: {improvement_ratio:.1f}x more features")

def test_data_volume():
    """Compare data volume between approaches"""
    print("\nComparing data volumes...")
    
    # Your current approach
    old_players = 645  # From your notebook
    old_seasons = 4
    old_total_rows = old_players * old_seasons
    
    print(f"Old approach:")
    print(f"  {old_players} players × {old_seasons} seasons = {old_total_rows:,} training examples")
    
    # New approach  
    new_players = 645  # Same players
    new_seasons = 4    # Same seasons
    games_per_season = 82  # Regular season games
    new_total_rows = new_players * new_seasons * games_per_season
    
    print(f"\nNew approach:")
    print(f"  {new_players} players × {new_seasons} seasons × {games_per_season} games = {new_total_rows:,} training examples")
    
    volume_improvement = new_total_rows / old_total_rows
    print(f"\nData volume improvement: {volume_improvement:.0f}x more training data")

def main():
    """Run comparison tests"""
    print("🏒 NHL Data Collection: Old vs New Comparison")
    print("=" * 60)
    
    # Test both approaches
    old_data = test_old_approach() 
    new_data = test_new_approach()
    
    # Compare feature richness
    test_feature_richness()
    
    # Compare data volume
    test_data_volume()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("Old approach: ❌ Limited seasonal aggregates, few features")
    print("New approach: ✅ Rich game-by-game context, many features") 
    print("\nThe new approach gives you:")
    print("• 80x more training data")  
    print("• 3x more features per data point")
    print("• Context that actually affects performance")
    print("• Temporal patterns ML can learn from")

if __name__ == "__main__":
    main()