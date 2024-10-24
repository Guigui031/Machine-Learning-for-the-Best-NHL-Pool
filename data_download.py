import os
import time

import requests
import json

headers = {'Accept': 'application/json'}


def download_players_points(season):
    path = f"data/{season}/{season}_players_points.json"
    if not os.path.exists(path):
        if not os.path.exists(f"data/{season}"):
            os.mkdir(f"data/{season}")
        r = requests.get(f'https://api-web.nhle.com/v1/skater-stats-leaders/{season}/2?categories=points&limit=-1', headers=headers)
        print(f"Response: {r.json()}")
        with open(path, 'w') as f:
            json.dump(r.json(), f)

# def download_players(season):
#     data = get_json()
#     ids = get_all_player_ids(data)
#     for i in ids:
#         path = f"data/{season}/players/{i}.json"
#         if os.path.exists(path):
#             continue
#         r = requests.get(f'https://api-web.nhle.com/v1/player/{i}/landing', headers=headers)
#         print(f"Response: {r.json()}")
#         with open(f"data/{season}/players/{i}.json", 'w') as f:
#             json.dump(r.json(), f)
#         time.sleep(2)


def download_season_standing(season):
    path = f"data/{season}/{season}_standings.json"
    if not os.path.exists(path):
        if not os.path.exists(f"data/{season}"):
            os.mkdir(f"data/{season}")
        r = requests.get(f'https://api-web.nhle.com/v1/standings/{season[-4:]}-05-01', headers=headers)
        print(f"Response: {r.json()}")
        with open(path, 'w') as f:
            json.dump(r.json(), f)


# def dowload_teams_stats(season):
#     data = get_json()
#     teams = get_all_teams_abbrev(data)
#     for team in teams:
#         path = f"data/{season}/teams/{team}.json"
#         if os.path.exists(path):
#             continue
#         r = requests.get(f'https://api-web.nhle.com/v1/club-stats/{team}/{season}/2', headers=headers)
#         print(f"Response: {r.json()}")
#         # TODO: if message = Not Found then skip
#         with open(f"data/{season}/teams/{team}.json", 'w') as f:
#             json.dump(r.json(), f)
#         time.sleep(2)


# def main():
#     download_players('20232024')
#     # dowload_teams_stats('20202021')

#
# if __name__ == '__main__':
#     main()