from bs4 import BeautifulSoup
import requests
import pandas

nhl_teams = [
    "anaheim_ducks",
    "arizona_coyotes",
    "boston_bruins",
    "buffalo_sabres",
    "calgary_flames",
    "carolina_hurricanes",
    "chicago_blackhawks",
    "colorado_avalanche",
    "columbus_blue_jackets",
    "dallas_stars",
    "detroit_red_wings",
    "edmonton_oilers",
    "florida_panthers",
    "los_angeles_kings",
    "minnesota_wild",
    "montreal_canadiens",
    "nashville_predators",
    "new_jersey_devils",
    "new_york_islanders",
    "new_york_rangers",
    "ottawa_senators",
    "philadelphia_flyers",
    "pittsburgh_penguins",
    "san_jose_sharks",
    "seattle_kraken",
    "st_louis_blues",
    "tampa_bay_lightning",
    "toronto_maple_leafs",
    "vancouver_canucks",
    "vegas_golden_knights",
    "washington_capitals",
    "winnipeg_jets",
    "utah_hockey_club"
]

print(nhl_teams)

final_df = pandas.DataFrame(columns=['name', 'salary'])

for team in nhl_teams:
    forwards = {}
    defense = {}
    goalies = {}
    url = f"https://capwages.com/teams/{team}"
    w = requests.get(url)
    soup = BeautifulSoup(w.content)
    masterTables = soup.find_all('table',{"class": "teamProfileRosterSection__table"})
    i = 0
    for table in masterTables:
        if i == 3:
            break
        tableList = table.find("tbody")
        for tr in tableList.find_all("tr"):
            all_td = tr.find_all("td")
            name = all_td[0].find("a").get_text(strip=True)
            salary = all_td[9].find("div", {"class": "w-full"}).get_text(strip=True)
            print(name, salary)
            if not name or not salary:
                continue
            if i == 0:
                forwards[name] = salary
            elif i == 1:
                defense[name] = salary
            elif i == 2:
                goalies[name] = salary

        i += 1

    forwards_df = pandas.DataFrame(forwards, columns=['name', 'salary'])
    defense_df = pandas.DataFrame(defense, columns=['name', 'salary'])
    goalies_df = pandas.DataFrame(goalies, columns=['name', 'salary'])
    final_df = pandas.concat([final_df, forwards_df, defense_df, goalies_df])
    break

final_df.to_csv("data/20232024/players_salary.csv")