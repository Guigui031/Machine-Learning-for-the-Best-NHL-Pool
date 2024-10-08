import json
import numpy as np


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


def main():
    data = get_json()
    n = len(data['points'])
    print(n)
    ids = get_all_player_ids(data)
    print(ids)
    teams = get_all_teams_abbrev(data)
    print(teams)

if __name__ == "__main__":
    main()
