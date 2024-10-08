import json
import numpy as np


def get_json():
    with open("data/20232024/players_points_20232024.json", "r") as f:
        data = json.load(f)
        return data


def get_ids(data):
    ids = []
    for player in data['points']:
        ids.append(player['id'])
    return ids


def main():
    data = get_json()
    n = len(data['points'])
    print(n)
    ids = get_ids(data)
    print(ids)

if __name__ == "__main__":
    main()
