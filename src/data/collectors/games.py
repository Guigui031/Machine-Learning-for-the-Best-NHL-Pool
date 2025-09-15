import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from src.data.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

class GameCollector(BaseCollector):
    """Collector for game-level data from NHL API"""
    
    def get_season_schedule(self, season: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get all games for a season"""
        endpoint = f"schedule/{season}"
        filepath = f"{self.data_config.get_season_path(season)}/schedule.json"
        
        return self.get_or_fetch(endpoint, filepath, force_refresh)
    
    def get_game_details(self, game_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get detailed game information including box score"""
        endpoint = f"gamecenter/{game_id}/boxscore"
        season = self._extract_season_from_game_id(game_id)
        filepath = f"{self.data_config.get_games_path(season)}/{game_id}_boxscore.json"
        
        return self.get_or_fetch(endpoint, filepath, force_refresh)
    
    def get_game_play_by_play(self, game_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get play-by-play data for a game"""
        endpoint = f"gamecenter/{game_id}/play-by-play"
        season = self._extract_season_from_game_id(game_id)
        filepath = f"{self.data_config.get_games_path(season)}/{game_id}_pbp.json"
        
        return self.get_or_fetch(endpoint, filepath, force_refresh)
    
    def get_season_games_batch(self, season: str, max_games: Optional[int] = None, 
                              force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get game details for all games in a season"""
        schedule = self.get_season_schedule(season, force_refresh)
        
        game_ids = []
        if 'gamesByDate' in schedule:
            for date_entry in schedule['gamesByDate']:
                for game in date_entry.get('games', []):
                    game_ids.append(str(game['id']))
        
        if max_games:
            game_ids = game_ids[:max_games]
        
        logger.info(f"Collecting {len(game_ids)} games for season {season}")
        
        games_data = []
        for i, game_id in enumerate(game_ids):
            try:
                game_data = self.get_game_details(game_id, force_refresh)
                games_data.append({
                    'game_id': game_id,
                    'data': game_data
                })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(game_ids)} games")
                    
            except Exception as e:
                logger.error(f"Failed to get game {game_id}: {e}")
                continue
        
        return games_data
    
    def get_player_game_logs(self, player_id: str, season: str, 
                           force_refresh: bool = False) -> Dict[str, Any]:
        """Get game-by-game stats for a player in a season"""
        endpoint = f"player/{player_id}/game-log/{season}/2"
        filepath = f"{self.data_config.get_players_path(season)}/{player_id}_gamelog.json"
        
        return self.get_or_fetch(endpoint, filepath, force_refresh)
    
    def _extract_season_from_game_id(self, game_id: str) -> str:
        """Extract season from game ID (e.g., '2023020001' -> '20232024')"""
        try:
            # Game ID format: SSSSTTNNNN where SSSS is season start year
            season_start = game_id[:4]
            season_end = str(int(season_start) + 1)
            return f"{season_start}{season_end}"
        except (ValueError, IndexError):
            logger.warning(f"Could not extract season from game ID {game_id}")
            return "20232024"  # Default fallback
    
    def collect_all_season_data(self, season: str, max_games: Optional[int] = None,
                              force_refresh: bool = False) -> Dict[str, Any]:
        """Collect comprehensive season data"""
        logger.info(f"Starting comprehensive data collection for season {season}")
        
        return {
            'schedule': self.get_season_schedule(season, force_refresh),
            'games': self.get_season_games_batch(season, max_games, force_refresh)
        }
    
    def collect(self, season: str, **kwargs) -> Dict[str, Any]:
        """Main collection method"""
        return self.collect_all_season_data(season, **kwargs)