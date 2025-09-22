# Complete Workflow Documentation

This document provides a comprehensive overview of the NHL Pool Optimization system workflow, from data collection to final team selection. The workflow integrates multiple modules to create an end-to-end machine learning pipeline for fantasy hockey team optimization.

## System Architecture Overview

```
NHL API → Data Collection → Data Processing → Feature Engineering → ML Training → Team Optimization → Results
```

## Phase 1: Data Collection

### 1.1 Season Data Download
**Objective**: Collect comprehensive NHL data for multiple seasons

**Process**:
1. **Player Statistics Download**
   ```python
   for season in ['20202021', '20212022', '20222023', '20232024']:
       download_players_points(season)
   ```
   - Downloads top scoring players for each season
   - Creates `{season}_players_points.json` files
   - Includes goals, assists, team information

2. **Team Roster Download**
   ```python
   teams = get_season_teams(season)
   for team in teams:
       download_teams_roster(season, team)
   ```
   - Downloads complete team rosters
   - Creates individual team files in `teams/` directory
   - Includes detailed player statistics per team

3. **Season Standings Download**
   ```python
   download_season_standing(season)
   ```
   - Downloads team performance data
   - Creates `{season}_standings.json` files
   - Used for team strength analysis

**Output Structure**:
```
data/
├── 20202021/
│   ├── 20202021_players_points.json
│   ├── 20202021_standings.json
│   └── teams/
│       ├── TOR.json
│       ├── MTL.json
│       └── ...
├── 20212022/
└── ...
```

### 1.2 Individual Player Data
**Process**:
```python
player_ids = get_all_player_ids(season, team)
for player_id in player_ids:
    download_player(player_id, season)
```
- Downloads biographical and career data
- Creates individual player files
- Includes multi-season statistics

## Phase 2: Data Processing and Validation

### 2.1 Player Object Creation
**Objective**: Transform raw JSON data into structured Player objects

**Process**:
```python
def load_player(player_id, seasons):
    player = Player(player_id)

    # Load biographical data
    player_data = load_player_json(player_id)
    player.set_name(first_name, last_name)
    player.set_age(birth_date)
    player.set_position(position)

    # Load multi-season statistics
    for season in seasons:
        player.set_season_points(season)
        # Populate season statistics...

    return player
```

**Key Features**:
- Validates data integrity
- Handles missing data with defaults
- Creates multi-season player profiles
- Integrates salary information

### 2.2 Data Normalization
**Objective**: Create fair comparison metrics across players and seasons

**Normalization Applied**:
```python
def process_data_skaters(df):
    # Points per game normalization
    df['points_per_game'] = (df['goals'] + df['assists']) / df['games_played']

    # Rate statistics
    df['shots_per_game'] = df['shots'] / df['games_played']
    df['pim_per_game'] = df['pim'] / df['games_played']

    # Season normalization (82-game season)
    df['games_played_normalized'] = df['games_played'] / 82

    return df
```

**Benefits**:
- Accounts for different season lengths (COVID-shortened seasons)
- Enables fair comparison across players
- Handles time-on-ice format conversion

## Phase 3: Feature Engineering

### 3.1 Training Dataset Creation
**Objective**: Create training examples for machine learning models

**Methodology**: Sliding Window Approach
- Use seasons N-1 and N to predict season N+1
- Example: Use 2021-22 and 2022-23 to predict 2023-24

```python
def create_training_data(players, target_season):
    training_examples = []

    for player in players:
        if has_required_seasons(player, target_season):
            features = extract_features(player, seasons_before_target)
            target = player.get_ratio_season_points(target_season)
            training_examples.append((features, target))

    return training_examples
```

### 3.2 Feature Extraction
**Features Used**:
1. **Historical Performance**:
   - PPG in previous two seasons
   - Goals, assists, shots per game
   - Plus/minus rating
   - Time on ice

2. **Player Attributes**:
   - Age (normalized)
   - Height, weight (normalized)
   - Position/role
   - Country of origin

3. **Team Context**:
   - Team performance (points percentage)
   - Team offensive/defensive rankings

4. **Derived Features**:
   - Performance trends (improvement/decline)
   - Consistency measures (variance across seasons)
   - Age-adjusted performance

## Phase 4: Machine Learning Training

### 4.1 Model Ensemble Training
**Objective**: Train ensemble of regression models to predict PPG

**Process**:
```python
def train_prediction_models(X_train, y_train):
    model_names = ['XGBoost', 'SVR', 'SGD']

    # Train individual models with hyperparameter tuning
    tuned_models = []
    for model_name in model_names:
        pipeline, param_grid = create_pipeline_and_params(model_name)
        best_model = tune_model(pipeline, param_grid, X_train, y_train)
        tuned_models.append((model_name, best_model))

    # Create ensemble
    ensemble = VotingRegressor(estimators=tuned_models)
    ensemble.fit(X_train, y_train)

    return ensemble
```

### 4.2 Model Validation
**Cross-Validation Strategy**:
- 5-fold stratified cross-validation
- Time series validation (train on earlier seasons, test on later)
- Performance metrics: MSE, MAE, R²

**Validation Process**:
```python
scores = cross_val_score(ensemble, X_train, y_train,
                        cv=5, scoring='neg_mean_squared_error')
print(f"CV MSE: {-scores.mean():.4f} (±{scores.std():.4f})")
```

### 4.3 Prediction Generation
**For Current Season**:
```python
def generate_predictions(players, trained_model, current_season):
    for player in players:
        if has_sufficient_history(player):
            features = extract_current_features(player)
            predicted_ppg = trained_model.predict([features])[0]
            player.predict_points = predicted_ppg
        else:
            player.predict_points = calculate_fallback_prediction(player)

    return players
```

## Phase 5: Team Optimization

### 5.1 Problem Setup
**Optimization Problem**:
- **Objective**: Maximize total predicted PPG
- **Constraints**:
  - Budget ≤ $88M
  - ≤12 Attackers, ≤6 Defensemen, ≤2 Goalies
  - Binary selection (each player in or out)

### 5.2 Linear Programming Solution
```python
def optimize_team(players):
    prob = LpProblem("Maximize_PPG", LpMaximize)

    # Decision variables
    x = {i: LpVariable(f"x_{i}", cat="Binary")
         for i in range(len(players))}

    # Objective function
    prob += lpSum(players[i].predict_points * x[i]
                  for i in range(len(players)))

    # Constraints
    prob += lpSum(players[i].salary * x[i]
                  for i in range(len(players))) <= 88000000

    prob += lpSum(x[i] for i in range(len(players))
                  if players[i].role == "A") <= 12
    # ... other position constraints

    prob.solve()

    return extract_solution(prob, players, x)
```

### 5.3 Alternative: Branch and Bound
For comparison and educational purposes:
```python
def branch_and_bound_optimization(players):
    players.sort(key=lambda p: p.predict_points / p.salary, reverse=True)

    best_team, best_ppg = team_optimization_branch_and_bound(
        players,
        budget=88000000,
        max_roles={"A": 12, "D": 6, "G": 2}
    )

    return best_team, best_ppg
```

## Phase 6: Results Analysis and Output

### 6.1 Solution Validation
```python
def validate_solution(selected_team):
    total_salary = sum(player.salary for player in selected_team)
    role_counts = count_by_role(selected_team)
    total_ppg = sum(player.predict_points for player in selected_team)

    assert total_salary <= 88000000
    assert role_counts['A'] <= 12
    assert role_counts['D'] <= 6
    assert role_counts['G'] <= 2

    return {
        'total_ppg': total_ppg,
        'total_salary': total_salary,
        'salary_utilization': total_salary / 88000000,
        'role_distribution': role_counts
    }
```

### 6.2 Output Generation
**CSV Export**:
```python
results_df = pd.DataFrame([
    {
        'role': player.role,
        'name': player.name,
        'position': player.position,
        'team': player.team,
        'salary': player.salary,
        'predicted_ppg': player.predict_points
    }
    for player in optimal_team
])

results_df.to_csv('optimal_team_2024.csv', index=False)
```

**Summary Statistics**:
```python
def generate_summary(optimal_team, solution_stats):
    summary = {
        'total_players': len(optimal_team),
        'total_predicted_ppg': solution_stats['total_ppg'],
        'total_salary': solution_stats['total_salary'],
        'salary_utilization_pct': solution_stats['salary_utilization'] * 100,
        'avg_ppg_per_player': solution_stats['total_ppg'] / len(optimal_team),
        'avg_salary_per_player': solution_stats['total_salary'] / len(optimal_team),
        'role_distribution': solution_stats['role_distribution']
    }

    return summary
```

## Complete Workflow Execution

### Main Pipeline
```python
def run_complete_pipeline():
    # Phase 1: Data Collection
    seasons = ['20202021', '20212022', '20222023', '20232024']
    for season in seasons:
        download_season_data(season)

    # Phase 2: Data Processing
    all_players = []
    for season in seasons[:-1]:  # Exclude current season for training
        season_players = load_season_players(season)
        all_players.extend(season_players)

    # Phase 3: Feature Engineering
    X_train, y_train = create_training_dataset(all_players, target_season='20232024')

    # Phase 4: ML Training
    ensemble_model = train_ensemble(X_train, y_train,
                                   ['XGBoost', 'SVR', 'SGD'])

    # Phase 5: Prediction and Optimization
    current_players = load_season_players('20232024')
    predicted_players = generate_predictions(current_players, ensemble_model, '20232024')

    optimal_team, total_ppg, total_salary = solve_problem(predicted_players)

    # Phase 6: Results
    solution_stats = validate_solution(optimal_team)
    save_results(optimal_team, solution_stats)

    return optimal_team, solution_stats
```

## Error Handling and Monitoring

### Data Quality Checks
```python
def validate_data_quality(players):
    issues = []

    for player in players:
        if player.salary is None or player.salary <= 0:
            issues.append(f"Invalid salary for {player.name}")

        if not player.seasons:
            issues.append(f"No season data for {player.name}")

        if player.predict_points is None:
            issues.append(f"No prediction for {player.name}")

    return issues
```

### Pipeline Monitoring
- Log execution time for each phase
- Track data quality metrics
- Monitor prediction accuracy over time
- Validate optimization solution feasibility

## Performance Optimization

### Computational Efficiency
1. **Parallel Processing**: Download data concurrently
2. **Caching**: Avoid redundant API calls
3. **Memory Management**: Process data in batches
4. **Algorithm Selection**: Use LP for optimal solutions

### Data Efficiency
1. **Incremental Updates**: Only download new/changed data
2. **Compression**: Store data efficiently
3. **Indexing**: Fast player and season lookups

## Future Enhancements

### Advanced Features
1. **Real-time Updates**: Injury reports, trades, lineup changes
2. **Risk Management**: Prediction uncertainty, player volatility
3. **Multi-objective Optimization**: Balance PPG, risk, team chemistry
4. **Interactive Interface**: Web-based team selection tool

### Technical Improvements
1. **Model Deployment**: Production ML pipeline
2. **A/B Testing**: Compare optimization strategies
3. **Feature Store**: Centralized feature management
4. **MLOps**: Automated retraining and validation

This workflow provides a robust, end-to-end system for NHL fantasy team optimization, combining data engineering, machine learning, and mathematical optimization to solve a complex real-world problem.