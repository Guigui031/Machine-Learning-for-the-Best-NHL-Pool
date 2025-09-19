"""
Configuration management for NHL pool optimization project.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for NHL API settings."""
    base_url: str = "https://api-web.nhle.com/v1"
    request_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {'Accept': 'application/json'}


@dataclass
class DataConfig:
    """Configuration for data paths and settings."""
    base_data_dir: str = "data"
    current_season: str = "20232024"
    training_seasons: List[str] = None
    salary_file_name: str = "players_salary.tsv"
    default_salary: int = 88000000

    def __post_init__(self):
        if self.training_seasons is None:
            self.training_seasons = ["20202021", "20212022", "20222023", "20232024"]

    @property
    def data_path(self) -> Path:
        """Get base data directory as Path object."""
        return Path(self.base_data_dir)

    def get_season_path(self, season: str) -> Path:
        """Get path for specific season data."""
        return self.data_path / season

    def get_players_points_path(self, season: str) -> Path:
        """Get path for player points data file."""
        return self.get_season_path(season) / f"{season}_players_points.json"

    def get_standings_path(self, season: str) -> Path:
        """Get path for standings data file."""
        return self.get_season_path(season) / f"{season}_standings.json"

    def get_team_path(self, season: str, team: str) -> Path:
        """Get path for team data file."""
        return self.get_season_path(season) / "teams" / f"{team}.json"

    def get_player_path(self, season: str, player_id: str) -> Path:
        """Get path for individual player data file."""
        return self.get_season_path(season) / "players" / f"{player_id}.json"

    def get_salary_path(self, season: str) -> Path:
        """Get path for salary data file."""
        return self.get_season_path(season) / self.salary_file_name


@dataclass
class PoolConfig:
    """Configuration for pool optimization settings."""
    salary_cap: int = 88000000
    max_attackers: int = 12
    max_defensemen: int = 6
    max_goalies: int = 2
    min_games_threshold: int = 100
    min_seasons_threshold: int = 3

    @property
    def total_players(self) -> int:
        """Total number of players in the pool."""
        return self.max_attackers + self.max_defensemen + self.max_goalies

    def validate_position_limits(self) -> bool:
        """Validate that position limits are reasonable."""
        return (self.max_attackers > 0 and
                self.max_defensemen > 0 and
                self.max_goalies > 0 and
                self.total_players <= 25)  # Reasonable upper limit


@dataclass
class MLConfig:
    """Configuration for machine learning settings."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring_metric: str = "neg_mean_squared_error"
    ensemble_voting: str = "soft"
    n_jobs: int = -1

    def validate(self) -> bool:
        """Validate ML configuration parameters."""
        return (0 < self.test_size < 1 and
                self.cv_folds > 1 and
                self.ensemble_voting in ["hard", "soft"])


class ProjectConfig:
    """Main configuration class that combines all configuration sections."""

    def __init__(self, config_file: Optional[str] = None):
        self.api = APIConfig()
        self.data = DataConfig()
        self.pool = PoolConfig()
        self.ml = MLConfig()

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file {config_file} not found, using defaults")
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Update configurations from file
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)

            if 'data' in config_data:
                for key, value in config_data['data'].items():
                    if hasattr(self.data, key):
                        setattr(self.data, key, value)

            if 'pool' in config_data:
                for key, value in config_data['pool'].items():
                    if hasattr(self.pool, key):
                        setattr(self.pool, key, value)

            if 'ml' in config_data:
                for key, value in config_data['ml'].items():
                    if hasattr(self.ml, key):
                        setattr(self.ml, key, value)

            logger.info(f"Configuration loaded from {config_file}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")

    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'api': {
                    'base_url': self.api.base_url,
                    'request_delay': self.api.request_delay,
                    'max_retries': self.api.max_retries,
                    'timeout': self.api.timeout,
                    'headers': self.api.headers
                },
                'data': {
                    'base_data_dir': self.data.base_data_dir,
                    'current_season': self.data.current_season,
                    'training_seasons': self.data.training_seasons,
                    'salary_file_name': self.data.salary_file_name,
                    'default_salary': self.data.default_salary
                },
                'pool': {
                    'salary_cap': self.pool.salary_cap,
                    'max_attackers': self.pool.max_attackers,
                    'max_defensemen': self.pool.max_defensemen,
                    'max_goalies': self.pool.max_goalies,
                    'min_games_threshold': self.pool.min_games_threshold,
                    'min_seasons_threshold': self.pool.min_seasons_threshold
                },
                'ml': {
                    'test_size': self.ml.test_size,
                    'random_state': self.ml.random_state,
                    'cv_folds': self.ml.cv_folds,
                    'scoring_metric': self.ml.scoring_metric,
                    'ensemble_voting': self.ml.ensemble_voting,
                    'n_jobs': self.ml.n_jobs
                }
            }

            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Error saving config file {config_file}: {e}")

    def validate(self) -> bool:
        """Validate all configuration sections."""
        validation_results = []

        # Validate pool configuration
        pool_valid = self.pool.validate_position_limits()
        if not pool_valid:
            logger.error("Invalid pool configuration: position limits")
        validation_results.append(pool_valid)

        # Validate ML configuration
        ml_valid = self.ml.validate()
        if not ml_valid:
            logger.error("Invalid ML configuration")
        validation_results.append(ml_valid)

        # Check data paths exist or can be created
        try:
            self.data.data_path.mkdir(parents=True, exist_ok=True)
            data_valid = True
        except Exception as e:
            logger.error(f"Cannot create data directory: {e}")
            data_valid = False
        validation_results.append(data_valid)

        return all(validation_results)

    def setup_logging(self, log_level: str = "INFO") -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('nhl_pool.log')
            ]
        )

    @classmethod
    def from_environment(cls) -> 'ProjectConfig':
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if they exist
        if 'NHL_DATA_DIR' in os.environ:
            config.data.base_data_dir = os.environ['NHL_DATA_DIR']

        if 'NHL_CURRENT_SEASON' in os.environ:
            config.data.current_season = os.environ['NHL_CURRENT_SEASON']

        if 'NHL_SALARY_CAP' in os.environ:
            try:
                config.pool.salary_cap = int(os.environ['NHL_SALARY_CAP'])
            except ValueError:
                logger.warning(f"Invalid NHL_SALARY_CAP value: {os.environ['NHL_SALARY_CAP']}")

        return config


# Global configuration instance
config = ProjectConfig()

# Load from config file if it exists
default_config_file = "config.json"
if Path(default_config_file).exists():
    config = ProjectConfig(default_config_file)