"""
Data validation utilities for NHL data processing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates NHL data for consistency and quality."""

    @staticmethod
    def validate_player_data(player_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate player data structure and required fields.

        Args:
            player_data: Player data dictionary from NHL API

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Required fields
        required_fields = ['playerId', 'firstName', 'lastName', 'position']
        for field in required_fields:
            if field not in player_data:
                errors.append(f"Missing required field: {field}")

        # Validate player ID
        if 'playerId' in player_data:
            try:
                player_id = int(player_data['playerId'])
                if player_id <= 0:
                    errors.append("Player ID must be positive")
            except (ValueError, TypeError):
                errors.append("Player ID must be a valid integer")

        # Validate position
        if 'position' in player_data:
            valid_positions = ['C', 'L', 'R', 'D', 'G']
            if player_data['position'] not in valid_positions:
                errors.append(f"Invalid position: {player_data['position']}")

        # Validate names
        for name_field in ['firstName', 'lastName']:
            if name_field in player_data:
                name_data = player_data[name_field]
                if isinstance(name_data, dict):
                    if 'default' not in name_data or not name_data['default'].strip():
                        errors.append(f"Empty or missing {name_field} default value")
                elif not isinstance(name_data, str) or not name_data.strip():
                    errors.append(f"Invalid {name_field} format")

        return len(errors) == 0, errors

    @staticmethod
    def validate_season_stats(stats: Dict[str, Any], season: str) -> Tuple[bool, List[str]]:
        """Validate season statistics data.

        Args:
            stats: Season statistics dictionary
            season: Season identifier

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check season format
        if not season.isdigit() or len(season) != 8:
            errors.append(f"Invalid season format: {season}")

        # Required numeric fields
        numeric_fields = ['gamesPlayed', 'goals', 'assists']
        for field in numeric_fields:
            if field in stats:
                try:
                    value = int(stats[field])
                    if value < 0:
                        errors.append(f"{field} cannot be negative")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid integer")

        # Validate games played is reasonable
        if 'gamesPlayed' in stats:
            games = stats['gamesPlayed']
            if isinstance(games, (int, float)) and (games > 82 or games < 0):
                errors.append(f"Games played ({games}) is unrealistic for NHL season")

        # Validate goals/assists relationship
        if all(field in stats for field in ['goals', 'assists', 'points']):
            try:
                goals = int(stats['goals'])
                assists = int(stats['assists'])
                points = int(stats['points'])
                if points != goals + assists:
                    errors.append(f"Points ({points}) != goals ({goals}) + assists ({assists})")
            except (ValueError, TypeError):
                pass  # Will be caught by individual field validation

        return len(errors) == 0, errors

    @staticmethod
    def validate_team_data(team_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate team data structure.

        Args:
            team_data: Team data dictionary from NHL API

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for required sections
        required_sections = ['skaters']
        for section in required_sections:
            if section not in team_data:
                errors.append(f"Missing required section: {section}")

        # Validate skaters data
        if 'skaters' in team_data:
            skaters = team_data['skaters']
            if not isinstance(skaters, list):
                errors.append("Skaters must be a list")
            elif len(skaters) == 0:
                errors.append("Team has no skaters")
            else:
                for i, skater in enumerate(skaters):
                    if not isinstance(skater, dict):
                        errors.append(f"Skater {i} is not a dictionary")
                        continue

                    if 'playerId' not in skater:
                        errors.append(f"Skater {i} missing playerId")

        return len(errors) == 0, errors

    @staticmethod
    def validate_dataframe_quality(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate DataFrame data quality.

        Args:
            df: Pandas DataFrame to validate
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            warnings.append(f"Missing required columns: {missing_cols}")

        # Check for empty DataFrame
        if len(df) == 0:
            warnings.append("DataFrame is empty")
            return False, warnings

        # Check for excessive missing values
        for col in df.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.5:
                    warnings.append(f"Column {col} has {missing_pct:.1%} missing values")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate rows")

        # Check for unrealistic values in common columns
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 18) | (df['age'] > 45)]['age'].count()
            if invalid_ages > 0:
                warnings.append(f"Found {invalid_ages} players with unrealistic ages")

        if 'games_played' in df.columns:
            invalid_games = df[(df['games_played'] < 0) | (df['games_played'] > 82)]['games_played'].count()
            if invalid_games > 0:
                warnings.append(f"Found {invalid_games} entries with invalid games played")

        return len(warnings) == 0, warnings

    @staticmethod
    def clean_numeric_data(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Clean and standardize numeric data in DataFrame.

        Args:
            df: Input DataFrame
            numeric_columns: List of columns to treat as numeric

        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()

        for col in numeric_columns:
            if col in df_cleaned.columns:
                # Convert to numeric, coercing errors to NaN
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

                # Replace infinite values with NaN
                df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)

                # Log cleaning results
                null_count = df_cleaned[col].isnull().sum()
                if null_count > 0:
                    logger.info(f"Cleaned column {col}: {null_count} values converted to NaN")

        return df_cleaned


def validate_json_file(file_path: str, validator_func) -> Tuple[bool, List[str]]:
    """Validate a JSON file using the provided validator function.

    Args:
        file_path: Path to JSON file
        validator_func: Function to validate the JSON data

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return validator_func(data)

    except FileNotFoundError:
        return False, [f"File not found: {file_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in file {file_path}: {e}"]
    except Exception as e:
        return False, [f"Error validating file {file_path}: {e}"]