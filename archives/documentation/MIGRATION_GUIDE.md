# Migration Guide: Enhanced Data Collection

## What's New

### 🔥 **Major Improvements**
1. **Game-by-game data** instead of season aggregates
2. **Modular architecture** with proper error handling  
3. **Rich contextual features** for ML
4. **Automatic caching** with smart refresh
5. **Proper data models** with type hints

### 📊 **New Data Available**
- **Game context**: Home/away, rest days, opponent strength
- **Situational stats**: Power play, penalty kill, even strength
- **Advanced metrics**: Faceoffs, hits, blocks, takeaways
- **Line combinations**: Teammate information
- **Game flow**: Period-by-period breakdown

## Usage Examples

### 1. **Collect Season Schedule** 
```python
from src.data.collectors.games import GameCollector

with GameCollector() as collector:
    # Get all games for 2023-24 season
    schedule = collector.get_season_schedule('20232024')
    print(f"Found {len(schedule['gamesByDate'])} game dates")
```

### 2. **Get Individual Game Details**
```python
with GameCollector() as collector:
    # Get detailed box score for specific game
    game_data = collector.get_game_details('2023020001')
    
    # Access player stats
    home_forwards = game_data['playerByGameStats']['homeTeam']['forwards']
    for player in home_forwards:
        print(f"{player['name']}: {player['goals']}G {player['assists']}A")
```

### 3. **Player Game Logs** 
```python
with GameCollector() as collector:
    # Connor McDavid's season game-by-game
    logs = collector.get_player_game_logs('8478402', '20232024')
    
    for game in logs['gameLog']:
        print(f"vs {game['opponentAbbrev']}: {game['points']} pts, {game['toi']} TOI")
```

### 4. **Bulk Season Collection**
```python
with GameCollector() as collector:
    # Collect first 100 games of season (for testing)
    season_data = collector.get_season_games_batch('20232024', max_games=100)
    print(f"Collected {len(season_data)} games")
```

## **Migration Steps**

### **Step 1: Replace your old data_download.py calls**

**Old way:**
```python
import data_download
data_download.download_players_points('20232024')
```

**New way:**
```python
from src.data.collectors.games import GameCollector

with GameCollector() as collector:
    season_data = collector.collect('20232024', max_games=100)  # Start small
```

### **Step 2: Update your data processing**

**Old features** (basic stats):
```python
features = ['goals', 'assists', 'games', 'shots', 'pim']
```

**New features** (rich context):
```python
# From game-by-game data, you can now create:
features = [
    'goals_per_game',
    'goals_vs_strong_opponents',  # Split by opponent quality
    'goals_home_vs_away',         # Home/away splits  
    'goals_rested_vs_tired',      # Based on rest days
    'powerplay_goals_per_opportunity',
    'linemate_quality_score',     # Weighted by ice time together
    'recent_form_10_games',       # Rolling averages
    'clutch_performance',         # Late-game situations
]
```

### **Step 3: Feature Engineering Examples**

```python
def create_rich_features(player_games):
    """Extract ML-ready features from game logs"""
    features = {}
    
    # Basic per-game averages
    features['ppg'] = sum(g['goals'] + g['assists'] for g in player_games) / len(player_games)
    
    # Context splits
    home_games = [g for g in player_games if g['homeRoadFlag'] == 'H']
    features['home_ppg'] = sum(g['goals'] + g['assists'] for g in home_games) / max(1, len(home_games))
    
    # Recent form (last 10 games)
    recent = player_games[-10:]
    features['recent_ppg'] = sum(g['goals'] + g['assists'] for g in recent) / len(recent)
    
    # Usage patterns
    features['avg_ice_time'] = sum(parse_toi(g['toi']) for g in player_games) / len(player_games)
    features['powerplay_usage'] = sum(1 for g in player_games if g.get('ppToi', 0) > 0) / len(player_games)
    
    return features

def parse_toi(toi_string):
    """Convert 'MM:SS' to minutes"""
    if ':' in toi_string:
        mins, secs = toi_string.split(':')
        return int(mins) + int(secs) / 60
    return 0
```

## **Key Benefits**

### **For Machine Learning:**
- **25x more data**: Game-level instead of season-level
- **Better features**: Context that actually affects performance  
- **Temporal patterns**: Can model streaks, slumps, trends
- **Situational awareness**: Home/away, opponent quality, rest

### **For Code Quality:**
- **Proper error handling**: Won't crash on API failures
- **Smart caching**: Faster development, respects rate limits
- **Type safety**: Catch errors at development time
- **Modular design**: Easy to extend and test

### **Quick Start:**
1. Install: No new dependencies needed
2. Test: Run a small collection first (`max_games=10`)  
3. Migrate: Update one feature at a time
4. Scale: Once working, collect full seasons

The new structure gives you the **rich data ML needs** while being **much more maintainable** than your current approach.