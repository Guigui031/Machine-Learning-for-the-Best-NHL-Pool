# Core Classes Documentation

The core classes (`player.py` and `team.py`) define the fundamental data structures for the NHL pool optimization system. These classes encapsulate player and team information, providing methods for calculating performance metrics and managing multi-season data.

## Player Class (`player.py`)

The `Player` class is the central data structure representing an NHL player with comprehensive biographical information, salary data, and multi-season performance statistics.

### Class Attributes

#### Biographical Information
- **`id`**: Unique NHL player identifier (string)
- **`first_name`**: Player's first name
- **`last_name`**: Player's last name
- **`name`**: Full name (first + last)
- **`salary_name`**: Name in "Last, First" format for salary lookups
- **`birth_date`**: Birth date string (YYYY-MM-DD format)
- **`age`**: Current age (calculated from birth_date)
- **`height`**: Height in centimeters
- **`weight`**: Weight in kilograms
- **`country`**: Birth country

#### Position and Role
- **`position`**: NHL position code ('C', 'L', 'R', 'D', 'G')
- **`role`**: Simplified role category ('A', 'D', 'G')
  - 'A' (Attacker): Centers, Left Wings, Right Wings
  - 'D' (Defenseman): Defensemen
  - 'G' (Goalie): Goalies

#### Financial and Performance Data
- **`salary`**: Player salary in dollars (default: 88,000,000)
- **`team`**: Current team
- **`predict_points`**: ML-predicted PPG for upcoming season (default: 0)
- **`points`**: Current season points
- **`seasons`**: Dictionary of Season objects keyed by season ID

### Core Methods

#### `set_name(first_name, last_name)`
**Purpose**: Sets player name information
**Side Effects**:
- Updates `name` as concatenated full name
- Sets `salary_name` in "Last, First" format for external data lookups
- Stores individual name components

#### `set_age(birth_date)`
**Purpose**: Calculates and sets current age from birth date
**Algorithm**:
```python
today = datetime.date.today()
born = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
```
**Features**: Accounts for whether birthday has occurred this year

#### `set_position(position_code)`
**Purpose**: Maps NHL position codes to simplified role categories
**Mapping Logic**:
- Centers (C), Left Wings (L), Right Wings (R) → 'A' (Attacker)
- Defensemen (D) → 'D' (Defenseman)
- Goalies (G) → 'G' (Goalie)

#### `set_season_points(season)`
**Purpose**: Creates new Season object for specified season
**Usage**: Must be called before setting season-specific statistics

#### `get_season_points(season)`
**Purpose**: Retrieves total fantasy points for a specific season
**Returns**: Position-adjusted points using role-specific scoring system

#### `get_ratio_season_points(season)`
**Purpose**: Calculates points per game for a specific season
**Returns**: Fantasy points divided by games played
**Use Case**: Normalized performance metric for comparison across players

### Property Setters
- `set_salary(salary)`: Updates salary information
- `set_country(country)`: Sets birth country
- `set_weight(weight)`: Sets weight in kilograms
- `set_height(height)`: Sets height in centimeters
- `set_team(team)`: Updates current team

## Season Class (`player.py`)

The `Season` class represents a player's performance data for a single NHL season.

### Class Attributes

#### Season Identification
- **`season_id`**: Season identifier (e.g., '20232024')
- **`team`**: Team during this season

#### Performance Statistics
- **`n_games_played`**: Games played in season
- **`n_goals`**: Goals scored
- **`n_assists`**: Assists recorded
- **`n_pim`**: Penalty minutes
- **`n_shots`**: Shots on goal
- **`n_plusMinus`**: Plus/minus rating
- **`n_time`**: Average time on ice

#### Goalie-Specific Statistics
- **`n_wins`**: Wins (goalies only)
- **`n_shutouts`**: Shutouts (goalies only)

### Core Methods

#### Statistics Setters
Each statistic has a dedicated setter method:
- `set_n_goals(n_goals)`
- `set_n_assists(n_assists)`
- `set_n_pim(n_pim)`
- `set_n_shots(n_shots)`
- `set_n_time(n_time)`
- `set_n_plusMinus(n_plusMinus)`
- `set_n_wins(n_wins)`
- `set_n_shutouts(n_shutouts)`
- `set_n_games_played(n_games_played)`
- `set_team(team)`

#### Performance Calculation Methods

#### `get_marqueur_points(role)`
**Purpose**: Calculates fantasy points using position-specific scoring system
**Scoring Rules**:
- **Attackers (A)**: `2 × goals + 1 × assists`
- **Defensemen (D)**: `3 × goals + 1 × assists`
- **Goalies (G)**: `3 × wins + 5 × shutouts + 3 × goals + 1 × assists`

**Rationale**:
- Defensemen get bonus for goals (harder to score from defense)
- Goalies have completely different scoring system emphasizing wins/shutouts
- System balances scoring across positions for fair comparison

#### `get_ratio_marqueur_points(role)`
**Purpose**: Calculates points per game using position-specific scoring
**Formula**: `get_marqueur_points(role) / n_games_played`
**Returns**: Normalized performance metric for season comparison

#### `get_points()`
**Purpose**: Gets traditional NHL points (goals + assists)
**Returns**: `n_goals + n_assists`
**Use Case**: Standard NHL statistic regardless of position

## Team Class (`team.py`)

The `Team` class is a simple data structure for team metadata and performance information.

### Class Attributes
- **`team_name`**: Full team name
- **`team_id`**: Unique team identifier
- **`season`**: Season identifier
- **`points_percentage`**: Team's points percentage for the season

### Constructor
```python
def __init__(self, team_name, team_id, season, points_percentage):
    self.team_name = team_name
    self.team_id = team_id
    self.season = season
    self.points_percentage = points_percentage
```

## Usage Patterns

### Creating and Populating a Player
```python
# Create player object
player = Player('8478402')  # McDavid's NHL ID

# Set biographical info
player.set_name('Connor', 'McDavid')
player.set_age('1997-01-13')
player.set_position('C')  # Center -> 'A' role
player.set_salary(12500000)

# Add season data
player.set_season_points('20232024')
player.seasons['20232024'].set_n_goals(64)
player.seasons['20232024'].set_n_assists(89)
player.seasons['20232024'].set_n_games_played(76)

# Calculate performance
total_points = player.get_season_points('20232024')  # 217 (64*2 + 89*1)
ppg = player.get_ratio_season_points('20232024')     # 2.855
```

### Multi-Season Analysis
```python
seasons = ['20212022', '20222023', '20232024']
for season in seasons:
    player.set_season_points(season)
    # ... populate season data ...

# Compare performance across seasons
for season in seasons:
    ppg = player.get_ratio_season_points(season)
    print(f"Season {season}: {ppg:.3f} PPG")
```

### Position-Specific Scoring Examples
```python
# Defenseman example
defenseman = Player('8477500')  # Hypothetical D
defenseman.set_position('D')
defenseman.set_season_points('20232024')
defenseman.seasons['20232024'].set_n_goals(15)      # Goals worth 3 points
defenseman.seasons['20232024'].set_n_assists(45)    # Assists worth 1 point
# Total: 15*3 + 45*1 = 90 fantasy points

# Goalie example
goalie = Player('8477400')  # Hypothetical G
goalie.set_position('G')
goalie.set_season_points('20232024')
goalie.seasons['20232024'].set_n_wins(35)          # Wins worth 3 points
goalie.seasons['20232024'].set_n_shutouts(5)       # Shutouts worth 5 points
# Total: 35*3 + 5*5 = 130 fantasy points
```

## Data Validation and Error Handling

### Default Values
The classes handle missing data gracefully:
- Numeric fields default to None or 0
- String fields default to None
- Methods check for None values before calculations

### Type Safety
- Season IDs are handled as strings to preserve leading zeros
- Numeric calculations use appropriate data types
- Date parsing includes error handling for invalid formats

### Consistency Checks
- Position codes are validated during role assignment
- Season objects must be created before adding statistics
- Fantasy point calculations require valid games played (> 0)

## Integration with Data Pipeline

### Loading from NHL API Data
The classes integrate with the data processing module:
1. `load_player()` function creates Player objects
2. NHL API data populates biographical information
3. Season statistics are extracted from team roster data
4. Salary information comes from external salary databases

### Machine Learning Integration
- `predict_points` attribute stores ML model predictions
- Historical season data provides features for training
- PPG calculations create target variables for regression

### Optimization Integration
- `salary` and `role` attributes used for constraints
- `predict_points` used in objective function
- Player objects passed directly to optimization algorithms

## Performance Considerations

### Memory Efficiency
- Season data stored only when needed
- Lazy loading of optional attributes
- Minimal object overhead for large player datasets

### Calculation Efficiency
- Position-specific scoring uses simple arithmetic
- No complex data structures for basic operations
- Direct attribute access for performance-critical paths

## Future Enhancements

### Additional Statistics
- Advanced metrics (Corsi, Fenwick, expected goals)
- Situational statistics (power play, penalty kill)
- Teammate and opponent quality adjustments

### Enhanced Validation
- Range checking for statistical values
- Consistency validation across seasons
- Data quality scoring and flagging

### Performance Optimization
- Caching of calculated values
- Vectorized operations for batch processing
- Database integration for persistent storage

These core classes provide a solid foundation for the NHL pool optimization system, balancing simplicity with the flexibility needed for complex analysis and optimization tasks.