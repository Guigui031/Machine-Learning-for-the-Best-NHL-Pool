from dataclasses import dataclass
from typing import List, Dict
import os

@dataclass
class APIConfig:
    base_url: str = "https://api-web.nhle.com/v1"
    headers: Dict[str, str] = None
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {'Accept': 'application/json'}

@dataclass
class DataConfig:
    base_path: str = "data"
    seasons: List[str] = None
    positions: Dict[str, int] = None
    salary_cap: int = 88_000_000
    
    def __post_init__(self):
        if self.seasons is None:
            self.seasons = ['20202021', '20212022', '20222223', '20232024']
        if self.positions is None:
            self.positions = {"A": 12, "D": 6, "G": 2}
    
    def get_season_path(self, season: str) -> str:
        return os.path.join(self.base_path, season)
    
    def get_games_path(self, season: str) -> str:
        return os.path.join(self.get_season_path(season), "games")
    
    def get_players_path(self, season: str) -> str:
        return os.path.join(self.get_season_path(season), "players")

# Global configuration instances
API_CONFIG = APIConfig()
DATA_CONFIG = DataConfig()