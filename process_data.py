import json
import numpy as np
import pandas

from player import Player
import data_download


def get_json():
    with open("data/20232024/goalies_points_20232024.json", "r") as f:
        data = json.load(f)
    with open("data/20232024/players_points_20232024.json", "r") as f:
        data.update(json.load(f))
    return data

def get_season_standings(season):
    data_download.download_season_standing(season)
    with open(f"data/{season}/{season}_standings.json", "r") as f:
        data = json.load(f)
    return data


def get_season_teams(season):
    data_download.download_players_points(season)
    with open(f"data/{season}/{season}_players_points.json", "r") as f:
        data = json.load(f)
    return get_all_teams_abbrev(data)


def get_all_player_ids(data):
    ids = []
    for player in data['points']:
        ids.append(player['id'])
    for player in data['wins']:
        ids.append(player['id'])
    return ids


def get_all_teams_abbrev(players_points_data):
    abbrevs = []
    for player in players_points_data['points']:
        abbrev = player['teamAbbrev']
        if abbrev not in abbrevs:
            abbrevs.append(abbrev)
    return abbrevs


def clean_pl_team_data(pl_team_data):
    for key in ['goals', 'assists', 'wins', 'shutouts', 'gamesPlayed']:
        if key not in pl_team_data.keys():
            pl_team_data[key] = np.nan
    return pl_team_data

def get_skater_info_from_team(season, team, skater_id):
    with open(f"data/{season}/teams/{team}.json") as f:
        team_data = json.load(f)
    for player in team_data['skaters']:
        if str(player['playerId']) == skater_id:
            return clean_pl_team_data(player)


def get_player_salary(name):
    players_salary = pandas.read_csv("data/20232024/players_salary.tsv", sep='\t')
    results = players_salary[players_salary['name'] == name]
    if len(results.index) == 0:
        return None
    salary = results.iloc[0]['salary']
    return salary


def load_season_points(pl_class):
    for season in ['20232024']:
        pl_team_data = get_skater_info_from_team(season, pl_class.team, pl_class.id)
        pl_class.set_season_points(season)
        pl_class.seasons[season].set_n_goals(pl_team_data['goals'])
        pl_class.seasons[season].set_n_assists(pl_team_data['assists'])
        pl_class.seasons[season].set_n_wins(pl_team_data['wins'])
        pl_class.seasons[season].set_n_shutouts(pl_team_data['shutouts'])
        pl_class.seasons[season].set_n_games_played(pl_team_data['gamesPlayed'])


def load_player(id):
    pl_class = Player(id)

    with open(f"data/20232024/players/{id}.json") as f:
        pl_data = json.load(f)

    pl_class.set_team(pl_data['currentTeamAbbrev'])
    pl_class.set_age(pl_data['birthDate'])
    pl_class.set_name(pl_data['firstName']['default'], pl_data['lastName']['default'])
    pl_class.set_salary(get_player_salary(pl_class.salary_name))
    pl_class.set_position(pl_data['position'])
    load_season_points(pl_class)

    return pl_class


def main():
    get_season_teams('20232024')
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
