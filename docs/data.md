# Data Pipeline Documentation

## Overview

The data pipeline handles NHL data acquisition, processing, and modeling. It provides a robust foundation for the machine learning system by ensuring clean, validated data flows through the entire pipeline.

## Module Structure

```
Data Pipeline
├── data_download.py     # NHL API integration
├── process_data.py      # Data cleaning & processing
├── player.py           # Player data models
└── team.py             # Team data models
```

## Data Sources

### NHL API Integration
The system integrates with the official NHL API to gather comprehensive player and team statistics.

**Endpoints Used:**
- `api-web.nhle.com/v1/roster/{team}/{season}` - Team rosters
- `api-web.nhle.com/v1/standings/{season}` - Season standings
- `api-web.nhle.com/v1/player-stats/{season}` - Player statistics

## Data Models

### Player Class (`player.py`)

```python
class Player:
    """Represents an NHL player with biographical and statistical data"""

    def __init__(self, name: str, age: int, position: str, salary: float = 88_000_000):
        self.name = name
        self.age = age
        self.position = position  # A (Attacker), D (Defenseman), G (Goalie)
        self.salary = salary
        self.height = None
        self.weight = None
        self.country = None
        self.seasons = []  # List of Season objects
```

**Key Attributes:**
- `name`: Player's full name
- `age`: Current age (used for age curve analysis)
- `position`: Role-based classification (A/D/G)
- `salary`: Fantasy salary (defaults to max budget)
- `seasons`: Historical performance data

**Methods:**
```python
def add_season(self, season: Season) -> None:
    """Add a season of performance data"""

def get_recent_performance(self, n_seasons: int = 2) -> Dict:
    """Get recent performance summary"""

def calculate_career_trajectory(self) -> float:
    """Calculate performance trend over time"""
```

### Season Class (`player.py`)

```python
class Season:
    """Represents a single season of player performance"""

    def __init__(self, year: str, team: str):
        self.year = year         # e.g., "2023-24"
        self.team = team         # Team abbreviation
        self.goals = 0
        self.assists = 0
        self.games = 0
        self.points = 0
        self.plus_minus = 0
        self.pim = 0            # Penalty minutes
        self.shots = 0
        self.time_on_ice = 0    # Average per game
```

**Calculated Properties:**
```python
@property
def points_per_game(self) -> float:
    """Calculate points per game (PPG)"""
    return self.points / self.games if self.games > 0 else 0.0

@property
def shooting_percentage(self) -> float:
    """Calculate shooting percentage"""
    return (self.goals / self.shots * 100) if self.shots > 0 else 0.0
```

### Team Class (`team.py`)

```python
class Team:
    """Represents an NHL team with performance metrics"""

    def __init__(self, name: str, abbreviation: str, season: str):
        self.name = name                    # Full team name
        self.abbreviation = abbreviation    # e.g., "TOR", "MTL"
        self.season = season               # Season identifier
        self.points = 0                    # Season points
        self.wins = 0
        self.losses = 0
        self.overtime_losses = 0
        self.points_percentage = 0.0       # Team performance metric
```

## Data Download (`data_download.py`)

### NHLDataFetcher Class

```python
class NHLDataFetcher:
    """Handles NHL API interactions with rate limiting and error handling"""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit  # Seconds between requests
        self.session = requests.Session()
        self.last_request_time = 0
```

**Key Methods:**

#### 1. Player Statistics
```python
def get_player_stats(self, season: str) -> Dict:
    """
    Fetch comprehensive player statistics for a season

    Args:
        season (str): Season in format "20232024"

    Returns:
        Dict: Player statistics organized by player ID

    Example:
        stats = fetcher.get_player_stats("20232024")
        # Returns: {player_id: {name, position, stats...}}
    """
```

#### 2. Team Rosters
```python
def get_team_rosters(self, season: str) -> Dict:
    """
    Get team compositions and player affiliations

    Args:
        season (str): Target season

    Returns:
        Dict: Team rosters with player assignments
    """
```

#### 3. Season Standings
```python
def get_standings(self, season: str) -> Dict:
    """
    Retrieve season standings and team performance

    Args:
        season (str): Target season

    Returns:
        Dict: Team standings with win/loss records
    """
```

#### 4. Rate Limiting
```python
def _handle_rate_limit(self) -> None:
    """Ensure API rate limiting compliance"""
    elapsed = time.time() - self.last_request_time
    if elapsed < self.rate_limit:
        time.sleep(self.rate_limit - elapsed)
    self.last_request_time = time.time()
```

### Usage Example

```python
from data_download import NHLDataFetcher

# Initialize fetcher with 1-second rate limit
fetcher = NHLDataFetcher(rate_limit=1.0)

# Download complete season data
season = "20232024"
player_stats = fetcher.get_player_stats(season)
team_rosters = fetcher.get_team_rosters(season)
standings = fetcher.get_standings(season)

# Process and save data
save_season_data(player_stats, team_rosters, standings, season)
```

## Data Processing (`process_data.py`)

### DataProcessor Class

The DataProcessor handles cleaning, validation, and normalization of raw NHL data.

```python
class DataProcessor:
    """Comprehensive data processing for NHL statistics"""

    def __init__(self, min_games: int = 10, min_seasons: int = 2):
        self.min_games = min_games      # Minimum games for inclusion
        self.min_seasons = min_seasons  # Minimum seasons for training
```

### Processing Pipeline

#### 1. Data Cleaning
```python
def clean_player_data(self, raw_data: Dict) -> pd.DataFrame:
    """
    Remove invalid entries and standardize formats

    Steps:
    - Remove players with insufficient games
    - Standardize team names and positions
    - Handle missing biographical data
    - Validate statistical consistency
    """
```

#### 2. Data Normalization
```python
def normalize_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize statistics for fair comparison

    Normalization:
    - Per-game rates (goals/game, assists/game)
    - Percentage metrics (shooting %, save %)
    - Time-based metrics (TOI per game)
    - Position adjustments
    """
```

#### 3. Training Data Creation
```python
def create_training_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create consecutive season pairs for ML training

    Strategy:
    - Use seasons N-1 and N to predict season N+1
    - Ensure chronological order
    - Handle player team changes
    - Filter for data quality

    Returns:
        DataFrame with features from 2 seasons, target from 3rd
    """
```

### Data Quality Validation

```python
def validate_data_quality(self, df: pd.DataFrame) -> bool:
    """
    Ensure data meets quality standards

    Checks:
    - No negative statistics
    - Games played <= 82 (regular season max)
    - Reasonable statistical ranges
    - Required columns present
    - Sufficient sample size
    """
```

## Data Storage Structure

### File Organization
```
data/
├── {season}/                           # Season-specific data
│   ├── {season}_players_points.json    # Player statistics
│   ├── {season}_standings.json         # Team standings
│   └── teams/                          # Team-specific data
│       ├── TOR.json                   # Toronto Maple Leafs
│       ├── MTL.json                   # Montreal Canadiens
│       └── ...
└── processed/                          # Cleaned data
    ├── training_data.csv              # ML-ready dataset
    ├── player_database.csv            # Complete player database
    └── validation_data.csv            # Hold-out validation set
```

### Data Format Examples

#### Player Statistics JSON
```json
{
  "player_id": "8477956",
  "name": "Connor McDavid",
  "position": "C",
  "role": "A",
  "age": 26,
  "team": "EDM",
  "season": "20232024",
  "stats": {
    "goals": 64,
    "assists": 89,
    "points": 153,
    "games": 82,
    "plus_minus": 28,
    "pim": 18,
    "shots": 353,
    "time_on_ice": 1533
  }
}
```

#### Training Data CSV Structure
```
name,age,position,role,team_1,team_2,
goals_1,assists_1,games_1,plus_minus_1,
goals_2,assists_2,games_2,plus_minus_2,
target_points,target_ppg
```

## Data Quality Metrics

### Validation Criteria

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **Minimum Games** | 10 games/season | Ensure meaningful sample size |
| **Maximum Games** | 82 games/season | Validate regular season data |
| **Position Consistency** | Must be A/D/G | Ensure valid position classification |
| **Statistical Ranges** | Reasonable bounds | Detect data entry errors |
| **Team Validation** | Valid NHL teams | Ensure legitimate team affiliations |

### Data Coverage

**Current Dataset:**
- **Seasons**: 2020-21 through 2023-24 (4 seasons)
- **Players**: ~800 unique players
- **Player-Seasons**: ~2,500 records
- **Training Pairs**: ~1,200 pairs
- **Teams**: All 32 NHL teams

**Data Completeness:**
- **Biographical**: 95% complete (age, position, country)
- **Basic Stats**: 100% complete (goals, assists, games)
- **Advanced Stats**: 90% complete (TOI, shots, plus/minus)
- **Team Context**: 100% complete (team affiliations)

## Performance Considerations

### API Rate Limiting
```python
# Recommended settings
RATE_LIMIT = 1.0  # 1 second between requests
MAX_RETRIES = 3   # Retry failed requests
TIMEOUT = 30      # Request timeout in seconds
```

### Memory Management
```python
# For large datasets
CHUNK_SIZE = 1000    # Process in chunks
MAX_MEMORY = "2GB"   # Memory limit
CACHE_SIZE = 100     # LRU cache size
```

### Error Handling
```python
def robust_data_fetch(self, season: str, max_retries: int = 3) -> Dict:
    """Fetch data with automatic retry logic"""
    for attempt in range(max_retries):
        try:
            return self._fetch_data(season)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Common Usage Patterns

### 1. Download New Season Data
```python
# Download and process new season
season = "20242025"
fetcher = NHLDataFetcher()
processor = DataProcessor()

# Fetch raw data
raw_data = fetcher.get_player_stats(season)

# Clean and process
clean_data = processor.clean_player_data(raw_data)

# Create training pairs
training_data = processor.create_training_pairs(clean_data)

# Save for ML pipeline
training_data.to_csv(f'data/processed/{season}_training.csv')
```

### 2. Build Complete Dataset
```python
# Combine multiple seasons
seasons = ["20202021", "20212022", "20222023", "20232024"]
all_data = []

for season in seasons:
    season_data = load_season_data(season)
    all_data.append(season_data)

# Combine and create training pairs
combined_df = pd.concat(all_data, ignore_index=True)
training_pairs = processor.create_training_pairs(combined_df)
```

### 3. Data Validation
```python
# Validate data quality
validation_results = processor.validate_data_quality(training_data)

if not validation_results['is_valid']:
    print("Data quality issues:")
    for issue in validation_results['issues']:
        print(f"  - {issue}")
else:
    print("Data quality validation passed")
```

## Troubleshooting

### Common Issues

#### 1. API Rate Limiting
**Symptom**: 429 HTTP errors
**Solution**: Increase rate limit delay
```python
fetcher = NHLDataFetcher(rate_limit=2.0)  # 2 second delay
```

#### 2. Missing Player Data
**Symptom**: KeyError on player statistics
**Solution**: Handle missing data gracefully
```python
def safe_get_stat(player_data: Dict, stat: str, default=0):
    return player_data.get('stats', {}).get(stat, default)
```

#### 3. Team Name Inconsistencies
**Symptom**: Team name variations
**Solution**: Use standardization mapping
```python
TEAM_MAPPING = {
    "Montréal Canadiens": "Montreal Canadiens",
    "St Louis Blues": "St. Louis Blues",
    # ... more mappings
}
```

### Performance Issues

#### 1. Large Dataset Processing
```python
# Use chunked processing
def process_large_dataset(df: pd.DataFrame, chunk_size: int = 1000):
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
```

#### 2. Memory Optimization
```python
# Optimize data types
df = df.astype({
    'goals': 'int16',
    'assists': 'int16',
    'games': 'int8',
    'team': 'category'
})
```

## Next Steps

1. **Real-time Updates**: Implement live season data fetching
2. **Advanced Metrics**: Add expected goals, Corsi, Fenwick
3. **Historical Data**: Extend to more historical seasons
4. **Data Validation**: Enhanced automated quality checks
5. **Performance**: Optimize for larger datasets

---

**Related Documentation:**
- [Feature Engineering](features.md) - How data flows to feature creation
- [Models](models.md) - How processed data trains models
- [API Reference](api-reference.md) - Complete method documentation