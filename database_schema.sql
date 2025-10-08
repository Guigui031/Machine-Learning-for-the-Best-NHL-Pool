CREATE TABLE goalie_seasons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        season_id TEXT,
        games_played INTEGER,
        games_started INTEGER,
        wins INTEGER,
        losses INTEGER,
        ties INTEGER,
        overtime_losses INTEGER,
        shutouts INTEGER,
        goals_against INTEGER,
        goals_against_avg REAL,
        shots_against INTEGER,
        saves INTEGER,
        save_pct REAL,
        time_on_ice REAL,
        penalty_minutes INTEGER,
        -- Additional goalie metrics
        goals INTEGER,  -- Rare but some goalies score
        assists INTEGER,
        points INTEGER,
        -- Derived metrics
        wins_per_game REAL,
        shutout_pct REAL,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (season_id) REFERENCES seasons(season_id),
        UNIQUE(player_id, season_id)
    );
CREATE TABLE player_teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        team_id INTEGER,
        season_id TEXT,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (season_id) REFERENCES seasons(season_id),
        UNIQUE(player_id, team_id, season_id)
    );
CREATE TABLE players (
        player_id INTEGER PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        full_name TEXT,
        position TEXT,
        role TEXT CHECK(role IN ('A', 'D', 'G')),
        birth_date TEXT,
        age INTEGER,
        height INTEGER,
        weight INTEGER,
        country TEXT,
        salary INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
CREATE TABLE seasons (
        season_id TEXT PRIMARY KEY,
        start_year INTEGER,
        end_year INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
CREATE TABLE skater_seasons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        season_id TEXT,
        games_played INTEGER,
        goals INTEGER,
        assists INTEGER,
        points INTEGER,
        plus_minus INTEGER,
        penalty_minutes INTEGER,
        shots INTEGER,
        shooting_pct REAL,
        time_on_ice_per_game REAL,
        powerplay_goals INTEGER,
        powerplay_points INTEGER,
        shorthanded_goals INTEGER,
        shorthanded_points INTEGER,
        game_winning_goals INTEGER,
        overtime_goals INTEGER,
        faceoff_pct REAL,
        avg_shifts_per_game REAL,
        -- Derived metrics
        ppg REAL,  -- Points per game
        goals_per_game REAL,
        assists_per_game REAL,
        shots_per_game REAL,
        marqueur_points REAL,
        marqueur_ppg REAL,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (season_id) REFERENCES seasons(season_id),
        UNIQUE(player_id, season_id)
    );
CREATE TABLE team_standings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id INTEGER,
        season_id TEXT,
        games_played INTEGER,
        wins INTEGER,
        losses INTEGER,
        overtime_losses INTEGER,
        points INTEGER,
        points_pct REAL,
        goals_for INTEGER,
        goals_against INTEGER,
        goal_differential INTEGER,
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (season_id) REFERENCES seasons(season_id),
        UNIQUE(team_id, season_id)
    );
CREATE TABLE teams (
        team_id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_abbrev TEXT UNIQUE NOT NULL,
        team_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
CREATE INDEX idx_skater_seasons_player ON skater_seasons(player_id);
CREATE INDEX idx_skater_seasons_season ON skater_seasons(season_id);
CREATE INDEX idx_goalie_seasons_player ON goalie_seasons(player_id);
CREATE INDEX idx_goalie_seasons_season ON goalie_seasons(season_id);
CREATE INDEX idx_player_teams_player ON player_teams(player_id);
CREATE INDEX idx_player_teams_team ON player_teams(team_id);
CREATE INDEX idx_player_teams_season ON player_teams(season_id);
