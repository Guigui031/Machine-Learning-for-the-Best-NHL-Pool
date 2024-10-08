import json
import numpy as np
import pandas

from player import Player


def get_json():
    with open("data/20232024/players_points_20232024.json", "r") as f:
        data = json.load(f)
        return data


def get_all_player_ids(data):
    ids = []
    for player in data['points']:
        ids.append(player['id'])
    return ids


def get_all_teams_abbrev(data):
    abbrevs = []
    for player in data['points']:
        abbrev = player['teamAbbrev']
        if abbrev not in abbrevs:
            abbrevs.append(abbrev)
    return abbrevs


def get_skater_info_from_team(team, skater_id):
    with open(f"data/20232024/teams/{team}.json") as f:
        team_data = json.load(f)
    for player in team_data['skaters']:
        if str(player['playerId']) == skater_id:
            return player


def get_player_salary(name):
    players_salary = pandas.read_csv("data/20232024/players_salary.tsv", sep='\t')
    results = players_salary[players_salary['name'] == name]
    if len(results.index) == 0:
        return None
    salary = results.iloc[0]['salary']
    return salary


def load_player(id):
    pl_class = Player(id)

    with open(f"data/20232024/players/{id}.json") as f:
        pl_data = json.load(f)

    pl_class.set_team(pl_data['currentTeamAbbrev'])
    pl_team_data = get_skater_info_from_team(pl_class.team, id)
    pl_class.set_age(pl_data['birthDate'])
    pl_class.set_name(pl_data['firstName']['default'], pl_data['lastName']['default'])
    pl_class.set_points(pl_team_data['points'])
    pl_class.set_salary(get_player_salary(pl_class.salary_name))
    pl_class.set_position(pl_data['position'])
    return pl_class


def main():
    data = get_json()
    n = len(data['points'])
    print(n)
    ids = get_all_player_ids(data)
    print(ids)
    teams = get_all_teams_abbrev(data)
    print(teams)
    load_player('8478402')


if __name__ == "__main__":
    main()
