import os
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
import json

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_players_points(season: str, force_refresh: bool = False) -> bool:
    """Download player points data for a given season.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        force_refresh: If True, re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    path = config.data.get_players_points_path(season)

    if path.exists() and not force_refresh:
        logger.info(f"Player points data for {season} already exists, skipping download")
        return True

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    url = f'{config.api.base_url}/skater-stats-leaders/{season}/2?categories=points&limit=-1'

    for attempt in range(config.api.max_retries):
        try:
            logger.info(f"Downloading player points for season {season} (attempt {attempt + 1}/{config.api.max_retries})")
            response = requests.get(url, headers=config.api.headers, timeout=config.api.timeout)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if not isinstance(data, dict) or 'points' not in data:
                logger.error(f"Invalid response structure for season {season}")
                if attempt == config.api.max_retries - 1:
                    return False
                continue

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully downloaded {len(data.get('points', []))} player records for season {season}")
            time.sleep(config.api.request_delay)
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for season {season}: {e}")
            if attempt < config.api.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for season {season}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading player points for season {season}: {e}")
            return False

    return False

def download_multiple_players(player_ids: List[str], season: str, force_refresh: bool = False) -> Dict[str, bool]:
    """Download multiple players' data with progress tracking.

    Args:
        player_ids: List of NHL player IDs
        season: Season to store the data under
        force_refresh: If True, re-download even if files exist

    Returns:
        Dict mapping player_id to success status
    """
    results = {}
    total_players = len(player_ids)

    logger.info(f"Starting download of {total_players} players for season {season}")

    for i, player_id in enumerate(player_ids, 1):
        logger.info(f"Processing player {i}/{total_players}: {player_id}")
        success = download_player(player_id, season, force_refresh)
        results[player_id] = success

        if not success:
            logger.warning(f"Failed to download data for player {player_id}")

    successful = sum(results.values())
    logger.info(f"Downloaded {successful}/{total_players} players successfully")

    return results


def download_season_standing(season: str, force_refresh: bool = False) -> bool:
    """Download season standings data.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        force_refresh: If True, re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    path = config.data.get_standings_path(season)

    if path.exists() and not force_refresh:
        logger.info(f"Season standings for {season} already exists, skipping download")
        return True

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract year from season for API call
    end_year = season[-4:]
    url = f'{config.api.base_url}/standings/{end_year}-05-01'

    for attempt in range(config.api.max_retries):
        try:
            logger.info(f"Downloading season standings for {season} (attempt {attempt + 1}/{config.api.max_retries})")
            response = requests.get(url, headers=config.api.headers, timeout=config.api.timeout)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if not isinstance(data, dict):
                logger.error(f"Invalid response structure for season standings {season}")
                if attempt == config.api.max_retries - 1:
                    return False
                continue

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully downloaded season standings for {season}")
            time.sleep(config.api.request_delay)
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for season standings {season}: {e}")
            if attempt < config.api.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for season standings {season}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading season standings for {season}: {e}")
            return False

    return False


def download_teams_roster(season: str, team: str, force_refresh: bool = False) -> bool:
    """Download team roster and stats data.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        team: Team abbreviation (e.g., 'TOR', 'MTL')
        force_refresh: If True, re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    path = config.data.get_team_path(season, team)

    if path.exists() and not force_refresh:
        logger.info(f"Team {team} data for {season} already exists, skipping download")
        return True

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    url = f'{config.api.base_url}/club-stats/{team}/{season}/2'

    for attempt in range(config.api.max_retries):
        try:
            logger.info(f"Downloading team {team} data for season {season} (attempt {attempt + 1}/{config.api.max_retries})")
            response = requests.get(url, headers=config.api.headers, timeout=config.api.timeout)
            response.raise_for_status()

            data = response.json()

            # Check if team data is valid (not "Not Found" message)
            if isinstance(data, dict) and data.get('message') == 'Not Found':
                logger.warning(f"Team {team} not found for season {season}")
                return False

            # Validate response structure
            if not isinstance(data, dict) or 'skaters' not in data:
                logger.error(f"Invalid response structure for team {team} in season {season}")
                if attempt == config.api.max_retries - 1:
                    return False
                continue

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully downloaded team {team} data for season {season}")
            time.sleep(config.api.request_delay)
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for team {team} in season {season}: {e}")
            if attempt < config.api.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for team {team} in season {season}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading team {team} data for season {season}: {e}")
            return False

    return False

def download_player(player_id: str, season: str = "20232024", force_refresh: bool = False) -> bool:
    """Download individual player data.

    Args:
        player_id: NHL player ID
        season: Season to store the data under (default: '20232024')
        force_refresh: If True, re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    path = config.data.get_player_path(season, player_id)

    if path.exists() and not force_refresh:
        logger.info(f"Player {player_id} data already exists, skipping download")
        return True

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    url = f'{config.api.base_url}/player/{player_id}/landing'

    for attempt in range(config.api.max_retries):
        try:
            logger.info(f"Downloading player {player_id} data (attempt {attempt + 1}/{config.api.max_retries})")
            response = requests.get(url, headers=config.api.headers, timeout=config.api.timeout)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if not isinstance(data, dict) or 'playerId' not in data:
                logger.error(f"Invalid response structure for player {player_id}")
                if attempt == config.api.max_retries - 1:
                    return False
                continue

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully downloaded player {player_id} data")
            time.sleep(config.api.request_delay)
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for player {player_id}: {e}")
            if attempt < config.api.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for player {player_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading player {player_id} data: {e}")
            return False

    return False
    
def download_all_teams_stats(season: str, teams: Optional[List[str]] = None, force_refresh: bool = False) -> Dict[str, bool]:
    """Download all teams' stats for a given season.

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        teams: List of team abbreviations. If None, will try to get from player data
        force_refresh: If True, re-download even if files exist

    Returns:
        Dict mapping team abbreviation to success status
    """
    if teams is None:
        # Try to get teams from player data if not provided
        from process_data import get_season_teams
        try:
            teams = get_season_teams(season)
        except Exception as e:
            logger.error(f"Could not get teams list for season {season}: {e}")
            return {}

    results = {}
    total_teams = len(teams)

    logger.info(f"Starting download of {total_teams} teams for season {season}")

    for i, team in enumerate(teams, 1):
        logger.info(f"Processing team {i}/{total_teams}: {team}")
        success = download_teams_roster(season, team, force_refresh)
        results[team] = success

        if not success:
            logger.warning(f"Failed to download data for team {team}")

    successful = sum(results.values())
    logger.info(f"Downloaded {successful}/{total_teams} teams successfully")

    return results


def download_season_data(season: str, force_refresh: bool = False) -> bool:
    """Download complete season data (players, standings, teams).

    Args:
        season: Season in format YYYYYYY (e.g., '20232024')
        force_refresh: If True, re-download even if files exist

    Returns:
        bool: True if all downloads successful, False otherwise
    """
    logger.info(f"Starting complete data download for season {season}")

    # Download player points data first
    if not download_players_points(season, force_refresh):
        logger.error(f"Failed to download player points for season {season}")
        return False

    # Download season standings
    if not download_season_standing(season, force_refresh):
        logger.error(f"Failed to download season standings for season {season}")
        return False

    # Download team data
    team_results = download_all_teams_stats(season, force_refresh=force_refresh)
    if not team_results:
        logger.error(f"Failed to download any team data for season {season}")
        return False

    failed_teams = [team for team, success in team_results.items() if not success]
    if failed_teams:
        logger.warning(f"Failed to download data for teams: {failed_teams}")

    logger.info(f"Season {season} data download completed")
    return True


def main():
    """Example usage of the data download functions."""
    season = '20232024'
    success = download_season_data(season, force_refresh=False)
    if success:
        logger.info("Data download completed successfully")
    else:
        logger.error("Data download failed")


if __name__ == '__main__':
    main()