#!/usr/bin/env python3
"""
Test script for the new data collection structure
"""

import logging
from src.data.collectors.games import GameCollector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_schedule_collection():
    """Test collecting season schedule"""
    with GameCollector() as collector:
        logger.info("Testing schedule collection...")
        schedule = collector.get_season_schedule('20232024')
        
        if 'gamesByDate' in schedule:
            total_games = sum(len(date_entry.get('games', [])) 
                            for date_entry in schedule['gamesByDate'])
            logger.info(f"Found {total_games} games in 2023-24 season")
            
            # Show first few games
            first_date = schedule['gamesByDate'][0]
            logger.info(f"First game date: {first_date.get('date')}")
            for game in first_date.get('games', [])[:3]:
                logger.info(f"Game {game['id']}: {game.get('awayTeam', {}).get('abbrev')} @ {game.get('homeTeam', {}).get('abbrev')}")
        
        return schedule

def test_game_details():
    """Test collecting individual game details"""
    with GameCollector() as collector:
        logger.info("Testing game details collection...")
        
        # Get a sample game ID from schedule first
        schedule = collector.get_season_schedule('20232024')
        if schedule.get('gamesByDate'):
            first_game = schedule['gamesByDate'][0]['games'][0]
            game_id = str(first_game['id'])
            
            logger.info(f"Getting details for game {game_id}")
            game_details = collector.get_game_details(game_id)
            
            if 'playerByGameStats' in game_details:
                home_players = len(game_details['playerByGameStats'].get('homeTeam', {}).get('forwards', []))
                away_players = len(game_details['playerByGameStats'].get('awayTeam', {}).get('forwards', []))
                logger.info(f"Found stats for {home_players + away_players} players")
            
            return game_details
        
        return None

def test_player_game_logs():
    """Test collecting player game logs"""
    with GameCollector() as collector:
        logger.info("Testing player game logs...")
        
        # Connor McDavid's ID: 8478402
        player_id = "8478402"
        season = "20232024"
        
        game_logs = collector.get_player_game_logs(player_id, season)
        
        if 'gameLog' in game_logs:
            games_played = len(game_logs['gameLog'])
            logger.info(f"Found {games_played} games for player {player_id}")
            
            # Show first few games
            for game in game_logs['gameLog'][:3]:
                logger.info(f"Game vs {game.get('opponentAbbrev')}: {game.get('goals')}G {game.get('assists')}A")
        
        return game_logs

def main():
    """Run all tests"""
    logger.info("Starting tests of new data collection structure...")
    
    try:
        # Test 1: Schedule collection
        schedule = test_schedule_collection()
        
        # Test 2: Game details  
        game_details = test_game_details()
        
        # Test 3: Player game logs
        player_logs = test_player_game_logs()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()