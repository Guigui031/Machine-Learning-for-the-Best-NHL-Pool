"""
Comprehensive data pipeline for NHL pool optimization project.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from config import config
from data_download import download_season_data, download_multiple_players
from process_data import (
    get_season_teams, get_all_player_ids, load_player,
    process_data_skaters, get_year_data_skaters
)
from data_validation import DataValidator
from player import Player

logger = logging.getLogger(__name__)


class NHLDataPipeline:
    """Main data pipeline class for NHL data processing."""

    def __init__(self, seasons: Optional[List[str]] = None):
        """Initialize the data pipeline.

        Args:
            seasons: List of seasons to process. If None, uses config.data.training_seasons
        """
        self.seasons = seasons or config.data.training_seasons
        self.players_cache = {}
        self.teams_cache = {}

    def download_all_data(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Download all required data for the specified seasons.

        Args:
            force_refresh: If True, re-download even if files exist

        Returns:
            Dict mapping season to success status
        """
        logger.info(f"Starting data download for seasons: {self.seasons}")
        results = {}

        for season in self.seasons:
            logger.info(f"Downloading data for season {season}")
            success = download_season_data(season, force_refresh)
            results[season] = success

            if not success:
                logger.error(f"Failed to download data for season {season}")

        successful_seasons = [s for s, success in results.items() if success]
        logger.info(f"Successfully downloaded data for {len(successful_seasons)}/{len(self.seasons)} seasons")

        return results

    def get_all_players_for_seasons(self, min_games: int = None) -> List[Player]:
        """Get all players with data across multiple seasons.

        Args:
            min_games: Minimum total games threshold. If None, uses config.pool.min_games_threshold

        Returns:
            List of Player objects with multi-season data
        """
        if min_games is None:
            min_games = config.pool.min_games_threshold

        all_players = {}

        for season in self.seasons:
            logger.info(f"Processing players for season {season}")

            # Get teams for this season
            teams = get_season_teams(season)
            if not teams:
                logger.warning(f"No teams found for season {season}")
                continue

            # Get all player IDs for this season
            season_players = []
            for team in teams:
                player_ids = get_all_player_ids(season, team)
                season_players.extend(player_ids)

            logger.info(f"Found {len(season_players)} players for season {season}")

            # Load player data
            for player_id in season_players:
                try:
                    if player_id not in all_players:
                        player = load_player(player_id, self.seasons)
                        if player and self._validate_player(player, min_games):
                            all_players[player_id] = player
                except Exception as e:
                    logger.error(f"Error loading player {player_id}: {e}")

        logger.info(f"Loaded {len(all_players)} players with sufficient data")
        return list(all_players.values())

    def _validate_player(self, player: Player, min_games: int) -> bool:
        """Validate if player meets minimum criteria.

        Args:
            player: Player object to validate
            min_games: Minimum total games threshold

        Returns:
            bool: True if player meets criteria
        """
        if not player.seasons:
            return False

        # Count total games and seasons
        total_games = sum(
            season.n_games_played or 0
            for season in player.seasons.values()
            if season.n_games_played is not None
        )

        num_seasons = len([
            season for season in player.seasons.values()
            if season.n_games_played and season.n_games_played > 0
        ])

        return (total_games >= min_games and
                num_seasons >= config.pool.min_seasons_threshold)

    def create_training_dataset(self, players: List[Player]) -> pd.DataFrame:
        """Create training dataset from player data.

        Args:
            players: List of Player objects

        Returns:
            DataFrame ready for machine learning
        """
        logger.info(f"Creating training dataset from {len(players)} players")

        training_data = []

        for player in players:
            if not player.seasons or len(player.seasons) < 3:
                continue

            # Create training examples using consecutive season pairs
            sorted_seasons = sorted(player.seasons.keys())

            for i in range(len(sorted_seasons) - 2):
                season1 = sorted_seasons[i]
                season2 = sorted_seasons[i + 1]
                target_season = sorted_seasons[i + 2]

                # Get data for each season
                data1 = get_year_data_skaters(player, season1, '1')
                data2 = get_year_data_skaters(player, season2, '2')

                # Skip if any season has insufficient data
                if (data1['games_1'] < 10 or data2['games_2'] < 10 or
                    player.seasons[target_season].n_games_played < 10):
                    continue

                # Combine player info and season data
                row_data = {
                    'player_id': player.id,
                    'name': player.name,
                    'position': player.position,
                    'role': player.role,
                    'age': player.age,
                    'height': player.height,
                    'weight': player.weight,
                    'country': player.country,
                    'salary': player.salary,
                    'target_points': player.get_ratio_season_points(target_season)
                }

                # Add season data
                row_data.update(data1)
                row_data.update(data2)

                training_data.append(row_data)

        if not training_data:
            logger.error("No training data created")
            return pd.DataFrame()

        df = pd.DataFrame(training_data)

        # Process and normalize the data
        df = process_data_skaters(df)

        # Validate dataset quality
        numeric_columns = ['age', 'height', 'weight', 'salary', 'target_points']
        is_valid, warnings = DataValidator.validate_dataframe_quality(df, numeric_columns)

        if warnings:
            logger.warning(f"Dataset quality issues: {warnings}")

        logger.info(f"Created training dataset with {len(df)} examples")
        return df

    def prepare_current_season_data(self, season: str = None) -> pd.DataFrame:
        """Prepare current season data for prediction.

        Args:
            season: Season to prepare data for. If None, uses config.data.current_season

        Returns:
            DataFrame ready for prediction
        """
        if season is None:
            season = config.data.current_season

        logger.info(f"Preparing current season data for {season}")

        # Get teams and players for current season
        teams = get_season_teams(season)
        if not teams:
            logger.error(f"No teams found for season {season}")
            return pd.DataFrame()

        all_players = []
        for team in teams:
            player_ids = get_all_player_ids(season, team)

            for player_id in player_ids:
                try:
                    # Load player with historical data
                    player = load_player(player_id, self.seasons)
                    if player and len(player.seasons) >= 2:
                        all_players.append(player)
                except Exception as e:
                    logger.error(f"Error loading player {player_id} for prediction: {e}")

        if not all_players:
            logger.error("No players loaded for current season prediction")
            return pd.DataFrame()

        # Create prediction dataset using last two seasons
        prediction_data = []

        for player in all_players:
            sorted_seasons = sorted(player.seasons.keys())

            if len(sorted_seasons) >= 2:
                # Use last two available seasons for prediction
                season1 = sorted_seasons[-2]
                season2 = sorted_seasons[-1]

                data1 = get_year_data_skaters(player, season1, '1')
                data2 = get_year_data_skaters(player, season2, '2')

                # Skip players with insufficient recent data
                if data1['games_1'] < 10 or data2['games_2'] < 10:
                    continue

                row_data = {
                    'player_id': player.id,
                    'name': player.name,
                    'position': player.position,
                    'role': player.role,
                    'age': player.age,
                    'height': player.height,
                    'weight': player.weight,
                    'country': player.country,
                    'salary': player.salary
                }

                row_data.update(data1)
                row_data.update(data2)
                prediction_data.append(row_data)

        if not prediction_data:
            logger.error("No prediction data created")
            return pd.DataFrame()

        df = pd.DataFrame(prediction_data)

        # Process the data
        df = process_data_skaters(df)

        logger.info(f"Prepared prediction dataset with {len(df)} players")
        return df

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity across all seasons.

        Returns:
            Dict with validation results
        """
        logger.info("Validating data integrity")

        results = {
            'seasons_checked': len(self.seasons),
            'valid_seasons': [],
            'invalid_seasons': [],
            'total_teams': 0,
            'total_players': 0,
            'data_quality_issues': []
        }

        for season in self.seasons:
            try:
                # Check if basic files exist
                players_path = config.data.get_players_points_path(season)
                standings_path = config.data.get_standings_path(season)

                if not players_path.exists():
                    results['invalid_seasons'].append(f"{season}: Missing players data")
                    continue

                if not standings_path.exists():
                    results['data_quality_issues'].append(f"{season}: Missing standings data")

                # Check teams
                teams = get_season_teams(season)
                results['total_teams'] += len(teams)

                # Sample player data quality
                sample_team = teams[0] if teams else None
                if sample_team:
                    player_ids = get_all_player_ids(season, sample_team)
                    results['total_players'] += len(player_ids)

                results['valid_seasons'].append(season)

            except Exception as e:
                results['invalid_seasons'].append(f"{season}: {e}")

        logger.info(f"Validation complete: {len(results['valid_seasons'])}/{len(self.seasons)} seasons valid")
        return results

    def export_summary_stats(self, output_path: str = "data_summary.json") -> None:
        """Export summary statistics about the dataset.

        Args:
            output_path: Path to save summary statistics
        """
        logger.info("Generating dataset summary statistics")

        integrity_results = self.validate_data_integrity()

        summary = {
            'config': {
                'seasons': self.seasons,
                'min_games_threshold': config.pool.min_games_threshold,
                'min_seasons_threshold': config.pool.min_seasons_threshold,
                'salary_cap': config.pool.salary_cap
            },
            'data_integrity': integrity_results,
            'pipeline_status': 'ready' if len(integrity_results['valid_seasons']) > 0 else 'incomplete'
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2)

        logger.info(f"Summary statistics saved to {output_path}")


def main():
    """Example usage of the data pipeline."""
    # Initialize pipeline
    pipeline = NHLDataPipeline()

    # Download all required data
    download_results = pipeline.download_all_data(force_refresh=False)

    # Validate data integrity
    validation_results = pipeline.validate_data_integrity()

    # Export summary
    pipeline.export_summary_stats()

    logger.info("Data pipeline setup complete")


if __name__ == '__main__':
    main()