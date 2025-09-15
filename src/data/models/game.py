from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class GameContext:
    """Context information for a game"""
    is_home: bool
    is_back_to_back: bool
    rest_days: int
    opponent_team: str
    opponent_strength: float  # Team rating/strength
    score_differential: int   # Team score - opponent score at game end
    game_situation: str      # 'regulation', 'overtime', 'shootout'

@dataclass 
class PlayerGameStats:
    """Individual player performance in a single game"""
    player_id: str
    player_name: str
    position: str
    team: str
    
    # Basic stats
    goals: int = 0
    assists: int = 0
    shots: int = 0
    hits: int = 0
    blocks: int = 0
    pim: int = 0  # Penalty minutes
    plus_minus: int = 0
    
    # Ice time (in seconds)
    ice_time: int = 0
    powerplay_time: int = 0
    shorthanded_time: int = 0
    
    # Advanced stats
    faceoff_wins: int = 0
    faceoff_attempts: int = 0
    takeaways: int = 0
    giveaways: int = 0
    
    # Contextual data
    line_number: Optional[int] = None  # 1st, 2nd, 3rd, 4th line
    linemates: List[str] = field(default_factory=list)
    
    @property
    def points(self) -> int:
        return self.goals + self.assists
    
    @property
    def faceoff_percentage(self) -> float:
        return self.faceoff_wins / max(1, self.faceoff_attempts)
    
    @property
    def ice_time_minutes(self) -> float:
        return self.ice_time / 60.0

@dataclass
class GameData:
    """Complete game information"""
    game_id: str
    date: datetime
    season: str
    game_type: str  # 'regular', 'playoff'
    
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    
    # Game flow
    period_scores: List[Dict[str, int]] = field(default_factory=list)
    game_length: str = "regulation"  # 'regulation', 'overtime', 'shootout'
    
    # Player performances
    player_stats: List[PlayerGameStats] = field(default_factory=list)
    
    # Game context
    context: Optional[GameContext] = None
    
    def get_player_stats(self, player_id: str) -> Optional[PlayerGameStats]:
        """Get stats for specific player"""
        for stats in self.player_stats:
            if stats.player_id == player_id:
                return stats
        return None
    
    def get_team_players(self, team: str) -> List[PlayerGameStats]:
        """Get all player stats for a team"""
        return [stats for stats in self.player_stats if stats.team == team]
    
    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score
    
    @property
    def winning_team(self) -> str:
        return self.home_team if self.home_score > self.away_score else self.away_team