# Data Module Documentation

The data module handles all aspects of NHL data collection, processing, and loading. It consists of two main components that work together to provide clean, validated data for machine learning models.

## Module Components

### 1. Data Download (`data_download.py`)

This module manages the collection of NHL data from the official NHL API. It implements robust data fetching with proper error handling, caching, and validation.

#### Core Functions

**`download_players_points(season, force_refresh=False)`**
- **Purpose**: Downloads player scoring statistics for a specific season
- **API Endpoint**: `/skater-stats-leaders/{season}/2?categories=points&limit=-1`
- **Returns**: Boolean indicating success/failure
- **Key Features**:
  - Automatic caching (skips download if file exists)
  - Exponential backoff on failures
  - JSON validation before saving
  - Comprehensive logging

**`download_season_standing(season, force_refresh=False)`**
- **Purpose**: Downloads team standings for end-of-season analysis
- **API Endpoint**: `/standings/{end_year}-05-01`
- **Returns**: Boolean indicating success/failure
- **Data Structure**: Team standings with points percentages

**`download_teams_roster(season, team, force_refresh=False)`**
- **Purpose**: Downloads complete team roster and player statistics
- **API Endpoint**: `/club-stats/{team}/{season}/2`
- **Returns**: Boolean indicating success/failure
- **Validation**: Checks for "Not Found" responses and valid data structure

**`download_player(player_id, season, force_refresh=False)`**
- **Purpose**: Downloads individual player biographical and career data
- **API Endpoint**: `/player/{player_id}/landing`
- **Returns**: Boolean indicating success/failure
- **Data Includes**: Career stats, biographical info, season-by-season totals

#### Batch Operations

**`download_multiple_players(player_ids, season, force_refresh=False)`**
- Downloads multiple players with progress tracking
- Returns dictionary mapping player_id to success status

**`download_all_teams_stats(season, teams=None, force_refresh=False)`**
- Downloads all team rosters for a season
- Auto-detects teams if not provided

**`download_season_data(season, force_refresh=False)`**
- Complete season data download (players, standings, teams)
- Orchestrates all download functions

#### Error Handling & Reliability

- **Retry Logic**: Up to 3 attempts with exponential backoff
- **Timeout Handling**: Configurable request timeouts
- **Rate Limiting**: Built-in delays between requests
- **Data Validation**: JSON structure validation before saving
- **Logging**: Comprehensive logging at INFO/ERROR levels

### 2. Data Processing (`process_data.py`)

This module transforms raw NHL API data into clean, normalized datasets suitable for machine learning. It handles data loading, validation, normalization, and feature engineering.

#### Core Functions

**`get_json(season=None)`**
- **Purpose**: Loads and combines player and goalie data for a season
- **Returns**: Combined data dictionary or None
- **Features**:
  - Handles multiple file naming conventions
  - Graceful error handling for missing files
  - UTF-8 encoding support

**`get_season_standings(season)`**
- **Purpose**: Loads season standings with automatic download
- **Returns**: Standings data dictionary
- **Auto-download**: Downloads data if not locally available

**`get_season_teams(season)`**
- **Purpose**: Extracts unique team abbreviations from player data
- **Returns**: List of team abbreviations
- **Use Case**: Useful for batch operations on all teams

#### Player Data Loading

**`load_player(player_id, seasons, season=None)`**
- **Purpose**: Creates complete Player object with multi-season data
- **Parameters**:
  - `player_id`: NHL player ID
  - `seasons`: List of seasons to load (e.g., ['20232024', '20222023'])
  - `season`: Season to store player file under
- **Returns**: Player object or None if loading fails
- **Features**:
  - Validates player data structure
  - Sets biographical information (name, age, position, salary)
  - Loads season-by-season statistics
  - Handles missing data with defaults

#### Data Processing & Normalization

**`process_data_skaters(df)`**
- **Purpose**: Normalizes skater statistics for fair comparison
- **Normalization Applied**:
  - Points per game = (goals + assists) / games_played
  - Rate statistics = stat / games_played
  - Games played normalized by 82-game season
  - Time on ice converted from MM:SS to minutes per game
- **Returns**: Processed DataFrame with normalized statistics

**`get_player_salary(name, season=None)`**
- **Purpose**: Retrieves player salary from external salary data
- **Parameters**: Player name in 'Last, First' format
- **Returns**: Salary in dollars or default (88M) if not found
- **Validation**: Checks for realistic salary ranges

#### Data Validation & Cleaning

**`clean_pl_team_data(pl_team_data)`**
- **Purpose**: Standardizes player team data with default values
- **Default Fields**: goals, assists, wins, shutouts, gamesPlayed, points, pim, shots, plusMinus
- **Features**:
  - Handles missing or null values
  - Type conversion and validation
  - NaN detection and replacement

#### Utility Functions

**`get_year_data_skaters(player, year, index)`**
- Extracts specific season data from Player object
- Returns dictionary with all relevant statistics
- Handles missing seasons with zero values

**`normalize_data(data, standardization)`**
- Applies z-score normalization to specified columns
- Uses pre-computed mean and standard deviation

## Data Flow Architecture

```
NHL API → download_players_points() → {season}_players_points.json
         ↓
load_player() → Player object → process_data_skaters() → Normalized DataFrame
         ↓
Machine Learning Pipeline
```

## File Structure

The module expects and creates the following directory structure:

```
data/
├── {season}/                          # e.g., 20232024/
│   ├── {season}_players_points.json   # Player scoring data
│   ├── {season}_standings.json        # Team standings
│   ├── teams/                         # Team-specific data
│   │   └── {team_abbrev}.json         # e.g., TOR.json
│   └── players/                       # Individual player data
│       └── {player_id}.json
└── salary_data_{season}.tsv           # External salary data
```

## Configuration Integration

The data module integrates with a configuration system (`config`) that defines:
- API endpoints and headers
- File paths and naming conventions
- Default values (salary, timeouts, retries)
- Current season information

## Error Handling Strategy

1. **Network Errors**: Exponential backoff with configurable retry limits
2. **Data Validation**: JSON structure validation before processing
3. **Missing Data**: Graceful degradation with default values
4. **File I/O**: Comprehensive exception handling with logging
5. **Type Conversion**: Safe conversion with fallback values

## Performance Considerations

- **Caching**: Automatic file-based caching to avoid redundant API calls
- **Batch Operations**: Efficient bulk downloading with progress tracking
- **Memory Management**: Streaming JSON processing for large datasets
- **Rate Limiting**: Built-in delays to respect API rate limits

## Usage Examples

```python
# Download complete season data
success = download_season_data('20232024')

# Load specific player with multiple seasons
player = load_player('8478402', ['20232024', '20222023'])

# Process skater data for ML
normalized_df = process_data_skaters(raw_skater_df)

# Get team rosters
teams = get_season_teams('20232024')
for team in teams:
    player_ids = get_all_player_ids('20232024', team)
```

This data module provides a robust foundation for the NHL pool optimization system, ensuring reliable data collection and consistent preprocessing for downstream machine learning applications.