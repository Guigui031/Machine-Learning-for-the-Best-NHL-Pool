# Optimizer Module Documentation

The optimizer module (`pool_classifier.py`) implements team selection algorithms for NHL fantasy pools. It uses mathematical optimization techniques to select the best possible team within salary and position constraints, maximizing predicted team performance.

## Module Overview

The optimizer solves a complex combinatorial optimization problem: selecting 20 players (12 attackers, 6 defensemen, 2 goalies) from all available NHL players while staying within an $88M salary budget and maximizing total predicted Points Per Game (PPG).

## Problem Formulation

### Objective Function
**Maximize**: Total predicted PPG of selected team
```
Maximize: Σ(player_i.predict_points * x_i)
```
Where `x_i` is 1 if player i is selected, 0 otherwise

### Constraints
1. **Budget Constraint**: Total salary ≤ $88,000,000
2. **Position Constraints**:
   - Attackers (A): ≤ 12 players
   - Defensemen (D): ≤ 6 players
   - Goalies (G): ≤ 2 players
3. **Binary Variables**: Each player either selected (1) or not (0)

## Algorithm Implementations

### 1. Linear Programming Solution (`solve_problem`)

**Method**: Uses PuLP library for exact linear programming solution

**Implementation Details**:
```python
prob = LpProblem("Maximize_PPG", LpMaximize)
x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(len(players))}
```

**Advantages**:
- **Optimal Solution**: Guaranteed to find the global optimum
- **Fast Execution**: Efficient for this problem size (~800 players)
- **Proven Algorithms**: Uses commercial-grade LP solvers

**Process**:
1. Create binary decision variables for each player
2. Define objective function (maximize total PPG)
3. Add salary and position constraints
4. Solve using LP solver
5. Extract selected players and solution metrics

**Returns**:
- `selected_players`: List of player names
- `total_ppg`: Total predicted points per game
- `total_salary`: Total salary cost

### 2. Branch and Bound Algorithm

The module implements two versions of branch-and-bound for comparison and educational purposes.

#### Simple Branch and Bound (`branch_and_bound`)

**Method**: Basic recursive search with pruning

**Key Functions**:
- `calcul_borne_sup(solution, joueurs_restants)`: Calculates upper bound
- `est_faisable(solution)`: Checks constraint feasibility
- **Pruning Strategy**: Abandon branches where upper bound ≤ current best

**Process**:
1. Use stack-based search (avoids recursion limits)
2. For each player: try including and excluding
3. Prune branches that cannot improve current best
4. Check feasibility only at leaf nodes

#### Advanced Branch and Bound (`team_optimization_branch_and_bound`)

**Method**: Optimized branch-and-bound with intelligent bounds

**Key Optimizations**:
1. **Preprocessing**: Sort players by PPG/salary ratio
2. **Early Pruning**: Check constraints at each node
3. **Tight Bounds**: Consider remaining budget and position limits
4. **Backtracking**: Efficient state management

**Bounding Function**:
```python
def bound(index, current_ppg, current_salary, role_counts):
    # Assume we can greedily pick remaining best players
    # within budget and position constraints
```

**Advantages**:
- More efficient pruning than simple version
- Realistic upper bounds considering all constraints
- Better performance on larger problem instances

### 3. Simplified Legacy Pipeline (`create_team`)

**Purpose**: Demonstrates complete workflow from data loading to team selection

**Process**:
1. **Load Data**: `load_players()` - Get all player objects
2. **Create Dataset**: Extract historical performance matrix
3. **Train Predictions**: Simple linear regression per player
4. **Optimize**: Apply branch-and-bound algorithm
5. **Output**: Save results to CSV file

**Prediction Model**:
- Uses simple linear regression on historical seasons
- Predicts future performance from seasons 1-4 to season 5
- Basic but interpretable approach

## Data Structures and Utilities

### Global Constants
```python
budget = 88000000      # Salary cap
n_goalies = 2         # Maximum goalies
n_atk = 12           # Maximum attackers
n_def = 6            # Maximum defensemen
```

### Supporting Functions

#### `load_players()`
**Purpose**: Loads all available players from data files
**Process**:
1. Get JSON data using `get_json()`
2. Extract all player IDs
3. Load complete Player objects
**Returns**: List of Player objects

#### `create_dataset(players)`
**Purpose**: Creates feature matrix for machine learning
**Format**: N×4 matrix (players × seasons)
**Seasons**: ['20202021', '20212022', '20222023', '20232024']
**Features**: PPG ratios for each historical season

#### `regression_on_player(ratios_points_par_match)`
**Purpose**: Predicts next season performance using linear regression
**Input**: Array of historical PPG ratios
**Model**: Linear trend from seasons 1-4 to predict season 5
**Returns**: Predicted PPG for upcoming season

## Constraint Handling

### Position Role Mapping
Player positions are mapped to three role categories:
- **Attackers (A)**: Centers (C), Left Wings (L), Right Wings (R)
- **Defensemen (D)**: Defensemen (D)
- **Goalies (G)**: Goalies (G)

### Feasibility Checking
```python
def est_faisable(solution):
    total_salaire = sum(joueur.salary for joueur in solution)
    nb_gardiens = len([j for j in solution if j.role == "G"])
    nb_defenseurs = len([j for j in solution if j.role == "D"])
    nb_attaquants = len([j for j in solution if j.role == "A"])

    return (total_salaire <= budget and
            0 <= nb_defenseurs <= n_def and
            0 <= nb_attaquants <= n_atk)
```

## Performance Optimization Strategies

### Preprocessing
1. **Player Sorting**: Sort by PPG/salary ratio for better bounds
2. **Data Validation**: Ensure all players have necessary attributes
3. **Memory Management**: Efficient data structures for large datasets

### Algorithmic Optimizations
1. **Branch Ordering**: Explore promising branches first
2. **Constraint Propagation**: Early constraint checking
3. **Bound Tightening**: Realistic upper bounds considering all constraints
4. **State Management**: Efficient backtracking in recursive algorithms

### Computational Complexity
- **LP Solution**: Polynomial time (practical for this problem size)
- **Branch and Bound**: Exponential worst-case, but efficient with good pruning
- **Problem Size**: ~800 players, manageable for both approaches

## Integration with ML Pipeline

### Input Requirements
The optimizer expects Player objects with:
- `predict_points`: ML-generated PPG prediction
- `salary`: Player salary (dollars)
- `role`: Position category ('A', 'D', 'G')
- `name`: Player identification

### ML Model Integration
```python
# After training ML models:
for player in players:
    features = extract_features(player)
    player.predict_points = trained_model.predict(features)

# Then optimize:
optimal_team, total_ppg = solve_problem(players)
```

### Output Format
Results can be exported as:
- **CSV Files**: Player lists with role, name, salary, team
- **DataFrame**: Structured data for further analysis
- **Summary Stats**: Total PPG, salary utilization, position breakdown

## Error Handling and Validation

### Data Validation
- **Salary Verification**: Ensure all players have valid salary data
- **Position Mapping**: Verify all positions map to valid roles
- **Prediction Validation**: Check for reasonable PPG predictions

### Algorithm Robustness
- **Infeasible Solutions**: Handle cases where no valid team exists
- **Numerical Precision**: Handle floating-point arithmetic carefully
- **Memory Limits**: Manage memory for large branch-and-bound trees

## Comparison of Approaches

### Linear Programming
**Pros**:
- Guaranteed optimal solution
- Fast execution time
- Well-tested solvers
- Handles large problem instances

**Cons**:
- Requires external solver (PuLP)
- Less educational value
- Limited customization

### Branch and Bound
**Pros**:
- Educational value (shows algorithm internals)
- Customizable pruning strategies
- No external dependencies
- Transparent decision process

**Cons**:
- Potentially slower on large instances
- Implementation complexity
- May not find optimal in reasonable time

## Usage Examples

### Basic Optimization
```python
# Load and prepare players
players = load_players()

# Add ML predictions
for player in players:
    player.predict_points = ml_model.predict(player.features)

# Solve using Linear Programming
team, ppg, salary = solve_problem(players)
print(f"Optimal team PPG: {ppg}, Total salary: ${salary:,}")
```

### Branch and Bound Comparison
```python
# Compare both algorithms
lp_team, lp_ppg, lp_salary = solve_problem(players)
bb_team, bb_ppg = team_optimization_branch_and_bound(players)

print(f"LP Solution: {lp_ppg} PPG")
print(f"B&B Solution: {bb_ppg} PPG")
```

### Complete Pipeline
```python
# Full workflow from scratch
def complete_optimization():
    players = load_players()
    dataset = create_dataset(players)

    # Simple prediction model
    for i, player in enumerate(players):
        player.predict_points = regression_on_player(dataset[i,:])

    # Optimize team
    optimal_team, total_ppg = branch_and_bound(players)

    # Save results
    save_team_results(optimal_team, total_ppg)

    return optimal_team, total_ppg
```

## Future Enhancements

### Algorithm Improvements
1. **Metaheuristics**: Genetic algorithms, simulated annealing
2. **Constraint Programming**: More flexible constraint handling
3. **Multi-Objective**: Balance PPG, salary, risk, team chemistry
4. **Dynamic Programming**: For specific problem structures

### Problem Extensions
1. **Trade Constraints**: Handle existing team composition
2. **Risk Management**: Incorporate prediction uncertainty
3. **Lineup Optimization**: Daily/weekly lineup changes
4. **Multi-Season**: Long-term team building strategies

### Performance Scaling
1. **Parallel Processing**: Distributed branch-and-bound
2. **Approximation Algorithms**: Fast near-optimal solutions
3. **Incremental Updates**: Handle player additions/removals
4. **GPU Acceleration**: Parallel constraint checking

This optimizer module provides both optimal and heuristic solutions to the NHL fantasy team selection problem, balancing mathematical rigor with practical performance considerations.