import os.path
import time

from process_data import get_json, get_ids
import requests
import json

headers = {'Accept': 'application/json'}


def download_players(season):
    data = get_json()
    ids = get_ids(data)
    for i in ids:
        path = f"data/{season}/players/{i}.json"
        if os.path.exists(path):
            continue
        r = requests.get(f'https://api-web.nhle.com/v1/player/{i}/landing', headers=headers)
        print(f"Response: {r.json()}")
        with open(f"data/{season}/players/{i}.json", 'w') as f:
            json.dump(r.json(), f)
        time.sleep(2)

def main():
    # download_players('20232024')
    pass



if __name__ == '__main__':
    main()