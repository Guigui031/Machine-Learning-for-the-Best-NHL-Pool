import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from player import Player
import data_download
from data_validation import DataValidator, validate_json_file
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_json(season: str = None) -> Optional[Dict[str, Any]]:
    """Load and combine goalie and player points data for a season.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024'). If None, uses current season from config.

    Returns:
        Combined data dictionary or None if loading fails
    """
    if season is None:
        season = config.data.current_season

    try:
        data = {}

        # Load goalies data if available
        goalies_path = config.data.get_season_path(season) / f"goalies_points_{season}.json"
        if goalies_path.exists():
            with open(goalies_path, "r", encoding='utf-8') as f:
                goalies_data = json.load(f)
                data.update(goalies_data)
                logger.info(f"Loaded goalies data for season {season}")
        else:
            logger.warning(f"Goalies data not found for season {season}")

        # Load players data
        players_path = config.data.get_season_path(season) / f"players_points_{season}.json"
        if players_path.exists():
            with open(players_path, "r", encoding='utf-8') as f:
                players_data = json.load(f)
                data.update(players_data)
                logger.info(f"Loaded players data for season {season}")
        else:
            # Try alternative naming
            alt_path = config.data.get_players_points_path(season)
            if alt_path.exists():
                with open(alt_path, "r", encoding='utf-8') as f:
                    players_data = json.load(f)
                    data.update(players_data)
                    logger.info(f"Loaded players data for season {season} (alternative path)")
            else:
                logger.error(f"Players data not found for season {season}")
                return None

        return data if data else None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON data for season {season}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading data for season {season}: {e}")
        return None

def get_season_standings(season: str) -> Optional[Dict[str, Any]]:
    """Get season standings data, downloading if necessary.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')

    Returns:
        Standings data dictionary or None if loading fails
    """
    try:
        # Download if not exists
        if not data_download.download_season_standing(season):
            logger.error(f"Failed to download standings for season {season}")
            return None

        path = config.data.get_standings_path(season)
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Successfully loaded standings for season {season}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in standings file for season {season}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading standings for season {season}: {e}")
        return None


def get_season_teams(season: str) -> List[str]:
    """Get list of team abbreviations for a season.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')

    Returns:
        List of team abbreviations
    """
    try:
        # Download players data if not exists
        if not data_download.download_players_points(season):
            logger.error(f"Failed to download player points for season {season}")
            return []

        path = config.data.get_players_points_path(season)
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)

        teams = get_all_teams_abbrev(data)
        logger.info(f"Found {len(teams)} teams for season {season}: {teams}")
        return teams

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in players file for season {season}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting teams for season {season}: {e}")
        return []


def get_all_player_ids(season: str, team: str) -> List[str]:
    """Get all player IDs for a specific team and season.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        team: Team abbreviation (e.g., 'TOR', 'MTL')

    Returns:
        List of player IDs
    """
    try:
        # Download team roster if not exists
        if not data_download.download_teams_roster(season, team):
            logger.error(f"Failed to download roster for team {team} in season {season}")
            return []

        path = config.data.get_team_path(season, team)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate team data structure
        is_valid, errors = DataValidator.validate_team_data(data)
        if not is_valid:
            logger.warning(f"Team {team} data validation issues: {errors}")

        # Extract unique player IDs (both skaters and goalies)
        ids = []
        if 'skaters' in data:
            for player in data['skaters']:
                if 'playerId' in player:
                    player_id = str(player['playerId'])
                    if player_id not in ids:
                        ids.append(player_id)

        if 'goalies' in data:
            for player in data['goalies']:
                if 'playerId' in player:
                    player_id = str(player['playerId'])
                    if player_id not in ids:
                        ids.append(player_id)

        logger.info(f"Found {len(ids)} players (skaters + goalies) for team {team} in season {season}")
        return ids

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in team file for {team} in season {season}: {e}")
        return []
    except FileNotFoundError:
        logger.error(f"Team file not found for {team} in season {season}")
        return []
    except Exception as e:
        logger.error(f"Error getting player IDs for team {team} in season {season}: {e}")
        return []


# function to normalize the data by the number of games played
def process_data_skaters(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize skater data by games played and season length.

    Args:
        df: DataFrame with skater statistics

    Returns:
        Processed DataFrame with normalized statistics
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to process_data_skaters")
        return df

    df_processed = df.copy()

    # Define columns that should be normalized by games played
    per_game_columns = [
        ('points_1', 'goals_1', 'assists_1', 'games_1'),
        ('points_2', 'goals_2', 'assists_2', 'games_2')
    ]

    rate_columns = [
        ('goals_1', 'games_1'),
        ('goals_2', 'games_2'),
        ('shots_1', 'games_1'),
        ('shots_2', 'games_2'),
        ('pim_1', 'games_1'),
        ('pim_2', 'games_2')
    ]

    # Calculate points per game first
    for points_col, goals_col, assists_col, games_col in per_game_columns:
        if all(col in df_processed.columns for col in [goals_col, assists_col, games_col]):
            # Avoid division by zero
            mask = df_processed[games_col] > 0
            df_processed.loc[mask, points_col] = (
                df_processed.loc[mask, goals_col] + df_processed.loc[mask, assists_col]
            ) / df_processed.loc[mask, games_col]
            df_processed.loc[~mask, points_col] = 0

    # Normalize rate statistics
    for stat_col, games_col in rate_columns:
        if stat_col in df_processed.columns and games_col in df_processed.columns:
            # Avoid division by zero
            mask = df_processed[games_col] > 0
            df_processed.loc[mask, stat_col] = df_processed.loc[mask, stat_col] / df_processed.loc[mask, games_col]
            df_processed.loc[~mask, stat_col] = 0

    # Process time on ice columns
    for time_col, games_col in [('time_1', 'games_1'), ('time_2', 'games_2')]:
        if time_col in df_processed.columns and games_col in df_processed.columns:
            try:
                # Handle time format (MM:SS -> minutes)
                def parse_time(time_str):
                    if pd.isna(time_str):
                        return 0
                    time_parts = str(time_str).split(':')
                    return int(time_parts[0]) if time_parts[0].isdigit() else 0

                df_processed[time_col] = df_processed[time_col].apply(parse_time)

                # Normalize by games played
                mask = df_processed[games_col] > 0
                df_processed.loc[mask, time_col] = df_processed.loc[mask, time_col] / df_processed.loc[mask, games_col]
                df_processed.loc[~mask, time_col] = 0

            except Exception as e:
                logger.error(f"Error processing time column {time_col}: {e}")
                df_processed[time_col] = 0

    # Normalize games played by season length (82 games)
    for games_col in ['games_1', 'games_2']:
        if games_col in df_processed.columns:
            df_processed[games_col] = df_processed[games_col] / 82  # TODO: adjust for shortened seasons

    # Validate processed data
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed = DataValidator.clean_numeric_data(df_processed, numeric_columns.tolist())

    logger.info(f"Processed {len(df_processed)} skater records")
    return df_processed


# function to extract the stats in the dict for a specific year and return zeros if the year is not there
def get_year_data_skaters(player, year, index):
    if year in player.seasons.keys():
        pl_season = player.seasons[year]
        return {'goals_'+index: pl_season.n_goals,
                'assists_'+index: pl_season.n_assists,
                'pim_'+index: pl_season.n_pim,
                'games_'+index: pl_season.n_games_played,
                'shots_'+index: pl_season.n_shots,
                'time_'+index: pl_season.n_time,
                'plus_minus_'+index: pl_season.n_plusMinus,
                'team_'+index: pl_season.team
                }
    else:
        return {'goals_'+index: 0,
                'assists_'+index: 0,
                'pim_'+index: 0,
                'games_'+index: 0,
                'shots_'+index: 0,
                'time_'+index: 0,
                'plus_minus_'+index: 0,
                'team_'+index: 0
                }

# def get_all_player_ids(data):
#     ids = []
#     for player in data['points']:
#         ids.append(player['id'])
#     for player in data['wins']:
#         ids.append(player['id'])
#     return ids


def get_all_teams_abbrev(players_points_data):
    abbrevs = []
    for player in players_points_data['points']:
        abbrev = player['teamAbbrev']
        if abbrev not in abbrevs:
            abbrevs.append(abbrev)
    return abbrevs


def clean_pl_team_data(pl_team_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and standardize player team data.

    Args:
        pl_team_data: Raw player team data

    Returns:
        Cleaned player team data with default values for missing fields
    """
    if not isinstance(pl_team_data, dict):
        logger.warning(f"Invalid player team data type: {type(pl_team_data)}")
        pl_team_data = {}

    # Required fields with default values
    default_fields = {
        'goals': 0,
        'assists': 0,
        'wins': 0,
        'shutouts': 0,
        'gamesPlayed': 0,
        'points': 0,
        'pim': 0,
        'shots': 0,
        'plusMinus': 0
    }

    cleaned_data = pl_team_data.copy()

    for key, default_value in default_fields.items():
        if key not in cleaned_data or cleaned_data[key] is None:
            cleaned_data[key] = default_value
        else:
            # Convert to appropriate numeric type
            try:
                cleaned_data[key] = float(cleaned_data[key]) if isinstance(cleaned_data[key], str) else cleaned_data[key]
                if np.isnan(cleaned_data[key]):
                    cleaned_data[key] = default_value
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {key}: {cleaned_data[key]}, using default {default_value}")
                cleaned_data[key] = default_value

    return cleaned_data

def get_skater_info_from_team(season: str, team: str, skater_id: str) -> Optional[Dict[str, Any]]:
    """Get skater information from team data.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        team: Team abbreviation
        skater_id: Player ID as string

    Returns:
        Cleaned player data dictionary or None if not found
    """
    try:
        path = config.data.get_team_path(season, team)
        if not path.exists():
            logger.error(f"Team file not found: {path}")
            return None

        with open(path, encoding='utf-8') as f:
            team_data = json.load(f)

        if 'skaters' not in team_data:
            logger.error(f"No skaters data found for team {team} in season {season}")
            return None

        for player in team_data['skaters']:
            if str(player.get('playerId', '')) == str(skater_id):
                return clean_pl_team_data(player)

        logger.warning(f"Skater {skater_id} not found in team {team} for season {season}")
        return None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in team file for {team}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting skater info for {skater_id} from team {team}: {e}")
        return None


def get_player_salary(name: str, season: str = None) -> int:
    """Get player salary from salary data file.

    Args:
        name: Player name in format 'Last, First'
        season: Season to look for salary data. If None, uses current season from config.

    Returns:
        Player salary in dollars, or default max salary if not found
    """
    if season is None:
        season = config.data.current_season
    """Get player salary from salary data file.

    Args:
        name: Player name in format 'Last, First'
        season: Season to look for salary data

    Returns:
        Player salary in dollars, or default max salary if not found
    """
    try:
        salary_path = config.data.get_salary_path(season)
        if not salary_path.exists():
            logger.warning(f"Salary file not found for season {season}, using default salary")
            return config.data.default_salary

        players_salary = pd.read_csv(salary_path, sep='\t')

        # Clean and validate salary data
        if 'name' not in players_salary.columns or 'salary' not in players_salary.columns:
            logger.error(f"Invalid salary file format for season {season}")
            return config.data.default_salary

        results = players_salary[players_salary['name'] == name]

        if len(results.index) == 0:
            logger.info(f"Salary not found for player {name}, using default salary")
            return config.data.default_salary

        salary_str = results.iloc[0]['salary']

        # Clean salary string and convert to integer
        if pd.isna(salary_str):
            logger.warning(f"NaN salary for player {name}, using default")
            return config.data.default_salary

        # Remove currency symbols and commas, then convert to int
        clean_salary = str(salary_str).replace('$', '').replace(',', '').strip()
        salary = int(clean_salary)

        # Validate salary is reasonable
        if salary < 0 or salary > 15000000:  # NHL salary cap considerations
            logger.warning(f"Unrealistic salary for {name}: ${salary:,}, using default")
            return config.data.default_salary

        return salary

    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing salary for {name}: {e}, using default")
        return 88000000
    except Exception as e:
        logger.error(f"Error getting salary for {name}: {e}, using default")
        return 88000000


def load_season_points(pl_class, season):
    pl_team_data = get_skater_info_from_team(season, pl_class.team, pl_class.id)
    pl_class.set_season_points(season)
    pl_class.seasons[season].set_n_goals(pl_team_data['goals'])
    pl_class.seasons[season].set_n_assists(pl_team_data['assists'])
    pl_class.seasons[season].set_n_wins(pl_team_data['wins'])
    pl_class.seasons[season].set_n_shutouts(pl_team_data['shutouts'])
    pl_class.seasons[season].set_n_games_played(pl_team_data['gamesPlayed'])



def load_player(player_id: str, seasons: List[str], season: str = None) -> Optional[Player]:
    """Load player data with multi-season statistics.

    Args:
        player_id: NHL player ID
        seasons: List of seasons to load data for
        season: Season to store player file under (default: current season)

    Returns:
        Player object with loaded data or None if loading fails
    """
    if season is None:
        season = config.data.current_season

    try:
        # Download player data if not exists
        if not data_download.download_player(player_id, season):
            logger.error(f"Failed to download player {player_id}")
            return None

        pl_class = Player(player_id)

        player_path = config.data.get_player_path(season, player_id)
        with open(player_path, encoding='utf-8') as f:
            pl_data = json.load(f)

        # Validate player data
        is_valid, errors = DataValidator.validate_player_data(pl_data)
        if not is_valid:
            logger.warning(f"Player {player_id} data validation issues: {errors}")

        # Set basic player info with error handling
        try:
            first_name = pl_data.get('firstName', {}).get('default', 'Unknown')
            last_name = pl_data.get('lastName', {}).get('default', 'Unknown')
            pl_class.set_name(first_name, last_name)
        except Exception as e:
            logger.error(f"Error setting name for player {player_id}: {e}")
            return None

        pl_class.set_position(pl_data.get('position', 'C'))
        pl_class.set_country(pl_data.get('birthCountry', 'Unknown'))

        if 'birthDate' in pl_data:
            pl_class.set_age(pl_data['birthDate'])
        if 'weightInKilograms' in pl_data:
            pl_class.set_weight(pl_data['weightInKilograms'])
        if 'heightInCentimeters' in pl_data:
            pl_class.set_height(pl_data['heightInCentimeters'])

        pl_class.set_salary(get_player_salary(pl_class.salary_name, season))

        # Load season statistics
        season_totals = pl_data.get('seasonTotals', [])
        if not season_totals:
            logger.warning(f"No season totals found for player {player_id}")

        for target_season in seasons:
            for stats in season_totals:
                if str(stats.get('season', '')) == target_season and stats.get('leagueAbbrev') == 'NHL':
                    try:
                        # Validate season stats
                        is_valid, errors = DataValidator.validate_season_stats(stats, target_season)
                        if not is_valid:
                            logger.warning(f"Season {target_season} stats validation issues for player {player_id}: {errors}")

                        pl_class.set_season_points(target_season)
                        season_obj = pl_class.seasons[target_season]

                        # Common stats (all players)
                        season_obj.set_n_goals(stats.get('goals', 0))
                        season_obj.set_n_assists(stats.get('assists', 0))
                        season_obj.set_n_points(stats.get('points', 0))
                        season_obj.set_n_pim(stats.get('pim', 0))
                        season_obj.set_n_games_played(stats.get('gamesPlayed', 0))

                        # Skater-specific stats
                        if pl_class.role in ['A', 'D']:
                            season_obj.set_n_shots(stats.get('shots', 0))
                            season_obj.set_n_plus_minus(stats.get('plusMinus', 0))
                            season_obj.set_n_powerplay_goals(stats.get('powerPlayGoals', 0))
                            season_obj.set_n_powerplay_points(stats.get('powerPlayPoints', 0))
                            season_obj.set_n_shorthanded_goals(stats.get('shorthandedGoals', 0))
                            season_obj.set_n_shorthanded_points(stats.get('shorthandedPoints', 0))
                            season_obj.set_n_game_winning_goals(stats.get('gameWinningGoals', 0))
                            season_obj.set_n_overtime_goals(stats.get('otGoals', 0))
                            season_obj.set_n_faceoff_percentage(stats.get('faceoffWinningPctg', 0.0))
                            season_obj.set_n_shooting_percentage(stats.get('shootingPctg', 0.0))

                            # Parse time on ice per game (format: "MM:SS" to seconds)
                            avg_toi = stats.get('avgToi', '0:00')
                            if isinstance(avg_toi, str) and ':' in avg_toi:
                                try:
                                    parts = avg_toi.split(':')
                                    toi_seconds = int(parts[0]) * 60 + int(parts[1])
                                    season_obj.set_n_time_on_ice_per_game(toi_seconds)
                                except:
                                    season_obj.set_n_time_on_ice_per_game(0)
                            else:
                                season_obj.set_n_time_on_ice_per_game(0)

                            season_obj.set_n_avg_shifts_per_game(stats.get('avgShiftsPerGame', 0.0))

                        # Goalie-specific stats
                        elif pl_class.role == 'G':
                            season_obj.set_n_games_started(stats.get('gamesStarted', 0))
                            season_obj.set_n_wins(stats.get('wins', 0))
                            season_obj.set_n_losses(stats.get('losses', 0))
                            season_obj.set_n_ties(stats.get('ties', 0))
                            season_obj.set_n_overtime_losses(stats.get('otLosses', 0))
                            season_obj.set_n_shutouts(stats.get('shutouts', 0))
                            season_obj.set_n_goals_against(stats.get('goalsAgainst', 0))
                            season_obj.set_n_goals_against_avg(stats.get('goalsAgainstAvg', 0.0))
                            season_obj.set_n_shots_against(stats.get('shotsAgainst', 0))
                            season_obj.set_n_saves(stats.get('saves', 0))
                            season_obj.set_n_save_percentage(stats.get('savePctg', 0.0))

                            # Total time on ice in seconds
                            toi = stats.get('timeOnIce', 0)
                            if isinstance(toi, str) and ':' in toi:
                                try:
                                    parts = toi.split(':')
                                    toi_seconds = int(parts[0]) * 60 + int(parts[1])
                                    season_obj.set_n_time_on_ice(toi_seconds)
                                except:
                                    season_obj.set_n_time_on_ice(0)
                            else:
                                season_obj.set_n_time_on_ice(toi if isinstance(toi, int) else 0)

                        # Handle team name
                        team_name = 'Unknown'
                        if 'teamCommonName' in stats and isinstance(stats['teamCommonName'], dict):
                            team_name = stats['teamCommonName'].get('default', 'Unknown')
                        elif 'teamAbbrev' in stats:
                            team_name = stats['teamAbbrev']
                        pl_class.seasons[target_season].set_team(team_name)

                    except Exception as e:
                        logger.error(f"Error processing season {target_season} for player {player_id}: {e}")
                    break

        return pl_class

    except FileNotFoundError:
        logger.error(f"Player file not found for {player_id}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON for player {player_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading player {player_id}: {e}")
        return None

# function to normalize the data
def normalize_data(data, standardization):
    for col in ['height', 'weight', 'age', 'plus_minus_1', 'plus_minus_2', 'time_1', 'time_2']:
        data[col] = (data[col] - standardization[col]['mu']) / standardization[col]['sig']
    return data


def main():
    load_player('8483505', ['20232024'])
    # data = get_json()
    # print(data)
    # # n = len(data['points'])
    # # print(n)
    # ids = get_all_player_ids(data)
    # print(len(ids))
    # # teams = get_all_teams_abbrev(data)
    # # print(teams)
    # # load_player('8478402')


if __name__ == "__main__":
    main()
