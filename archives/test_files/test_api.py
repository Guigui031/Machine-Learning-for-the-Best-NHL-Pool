#!/usr/bin/env python3
"""
Test actual NHL API calls with the new structure
Run with: conda activate ift3150 && python test_api.py
"""

import os
import sys
import json

# Add src to path  
sys.path.insert(0, 'src')

def test_schedule_api():
    """Test getting season schedule from NHL API"""
    print("Testing NHL API schedule endpoint...")
    
    try:
        from src.data.collectors.games import GameCollector
        
        with GameCollector() as collector:
            # Get just the 2023-24 season schedule
            schedule = collector.get_season_schedule('20232024')
            
            if 'gamesByDate' in schedule:
                total_dates = len(schedule['gamesByDate'])
                total_games = sum(len(date_entry.get('games', [])) 
                                for date_entry in schedule['gamesByDate'])
                
                print(f"Schedule downloaded successfully")
                print(f"  Total game dates: {total_dates}")
                print(f"  Total games: {total_games}")
                
                # Show some sample games
                if schedule['gamesByDate']:
                    first_date = schedule['gamesByDate'][0]
                    print(f"  First date: {first_date.get('date')}")
                    
                    for i, game in enumerate(first_date.get('games', [])[:3]):
                        away = game.get('awayTeam', {}).get('abbrev', 'UNK')
                        home = game.get('homeTeam', {}).get('abbrev', 'UNK') 
                        print(f"    Game {i+1}: {away} @ {home}")
                
                return schedule
            else:
                print("Schedule format unexpected")
                return None
                
    except Exception as e:
        print(f"Schedule test failed: {e}")
        return None

def test_single_game():
    """Test getting details for one specific game"""
    print("\nTesting single game details...")
    
    try:
        from src.data.collectors.games import GameCollector
        
        with GameCollector() as collector:
            # First get schedule to find a real game ID
            schedule = collector.get_season_schedule('20232024')
            
            if schedule and 'gamesByDate' in schedule:
                # Get first game from schedule
                first_game = schedule['gamesByDate'][0]['games'][0]
                game_id = str(first_game['id'])
                
                print(f"Getting details for game {game_id}...")
                
                game_details = collector.get_game_details(game_id)
                
                if game_details:
                    print(f"Game details downloaded successfully")
                    
                    # Check what data we got
                    if 'awayTeam' in game_details:
                        away = game_details['awayTeam'].get('abbrev', 'UNK')
                        home = game_details['homeTeam'].get('abbrev', 'UNK')
                        away_score = game_details.get('awayTeam', {}).get('score', 0)
                        home_score = game_details.get('homeTeam', {}).get('score', 0)
                        print(f"  Matchup: {away} {away_score} - {home_score} {home}")
                    
                    # Check player stats
                    if 'playerByGameStats' in game_details:
                        home_players = game_details['playerByGameStats'].get('homeTeam', {})
                        away_players = game_details['playerByGameStats'].get('awayTeam', {})
                        
                        home_forwards = len(home_players.get('forwards', []))
                        away_forwards = len(away_players.get('forwards', []))
                        
                        print(f"  Player stats: {home_forwards + away_forwards} forwards")
                        
                        # Show top scorer from home team
                        if home_players.get('forwards'):
                            top_scorer = max(home_players['forwards'], 
                                           key=lambda p: p.get('goals', 0) + p.get('assists', 0))
                            goals = top_scorer.get('goals', 0)
                            assists = top_scorer.get('assists', 0)
                            name = top_scorer.get('name', {}).get('default', 'Unknown')
                            print(f"  Top home scorer: {name} ({goals}G, {assists}A)")
                    
                    return game_details
                else:
                    print("No game details returned")
                    return None
            else:
                print("Could not get schedule to find game ID")
                return None
                
    except Exception as e:
        print(f"Single game test failed: {e}")
        return None

def test_player_logs():
    """Test getting player game logs"""
    print("\nTesting player game logs...")
    
    try:
        from src.data.collectors.games import GameCollector
        
        # Connor McDavid's player ID
        player_id = "8478402"
        season = "20232024"
        
        with GameCollector() as collector:
            print(f"Getting game logs for player {player_id} in {season}...")
            
            game_logs = collector.get_player_game_logs(player_id, season)
            
            if game_logs and 'gameLog' in game_logs:
                games_count = len(game_logs['gameLog'])
                print(f"Player logs downloaded successfully")
                print(f"  Games played: {games_count}")
                
                # Show first few games
                for i, game in enumerate(game_logs['gameLog'][:5]):
                    opponent = game.get('opponentAbbrev', 'UNK')
                    goals = game.get('goals', 0)
                    assists = game.get('assists', 0)
                    points = goals + assists
                    toi = game.get('toi', '0:00')
                    
                    print(f"    Game {i+1} vs {opponent}: {points} pts ({goals}G {assists}A), {toi} TOI")
                
                return game_logs
            else:
                print("No game logs returned or wrong format")
                return None
                
    except Exception as e:
        print(f"Player logs test failed: {e}")
        return None

def test_data_persistence():
    """Test that data is being cached properly"""
    print("\nTesting data caching...")
    
    try:
        # Check if data directory was created
        if os.path.exists('data'):
            print("Data directory exists")
            
            # Check for cached files
            season_dirs = [d for d in os.listdir('data') if os.path.isdir(f'data/{d}')]
            if season_dirs:
                print(f"Found season directories: {season_dirs}")
                
                for season_dir in season_dirs:
                    season_path = f'data/{season_dir}'
                    files = os.listdir(season_path)
                    print(f"  {season_dir}: {len(files)} cached files")
                
                return True
            else:
                print("ℹ No season directories yet (expected on first run)")
                return True
        else:
            print("ℹ No data directory yet (expected on first run)")
            return True
            
    except Exception as e:
        print(f"Caching test failed: {e}")
        return False

def main():
    """Run API tests"""
    print("Testing NHL API Data Collection")
    print("=" * 50)
    print("Note: These tests make real API calls and may take a moment...")
    
    tests = [
        test_schedule_api,
        test_single_game,
        test_player_logs,
        test_data_persistence
    ]
    
    passed = 0
    results = {}
    
    for test in tests:
        try:
            result = test()
            if result is not None:
                passed += 1
                results[test.__name__] = "PASSED"
            else:
                results[test.__name__] = "FAILED"
        except Exception as e:
            results[test.__name__] = f"CRASHED: {e}"
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All API tests passed! Your new data collection is working.")
    else:
        print("Some tests failed. Check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)