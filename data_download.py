import os.path
import time

from process_data import get_json, get_ids
import requests
import json

headers = {'Accept': 'application/json'}




def main():
    data = get_json()
    ids = get_ids(data)
    for i in ids:
        path = f"data/20232024/players/{i}.json"
        if os.path.exists(path):
            continue
        r = requests.get(f'https://api-web.nhle.com/v1/player/{i}/landing', headers=headers)
        print(f"Response: {r.json()}")
        with open(f"data/20232024/players/{i}.json", 'w') as f:
            json.dump(r.json(), f)
        time.sleep(2)


if __name__ == '__main__':
    main()