import datetime

import numpy as np


class Player:
    def __init__(self, id):
        self.last_name = None
        self.first_name = None
        self.id = id
        self.seasons = {}
        self.role = None
        self.salary_name = None
        self.position = None
        self.salary = np.inf
        self.points = None
        self.age = None
        self.name = None
        self.birth_date = None
        self.predict_points = 0
        self.height = None
        self.weight = None
        self.country = None

    def set_name(self, first_name, last_name):
        self.name = first_name + ' ' + last_name
        self.salary_name = last_name + ', ' + first_name
        self.first_name = first_name
        self.last_name = last_name

    def set_age(self, birth_date):
        self.birth_date = birth_date
        today = datetime.date.today()
        born = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
        self.age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def set_salary(self, salary):
        self.salary = salary
        
    def set_country(self, country):
        self.country = country

    def set_weight(self, weight):
        self.weight = weight

    def set_height(self, height):
        self.height = height
        
    def set_position(self, position_code):
        self.position = position_code
        if position_code in ['C', 'L', 'R']:
            self.role = 'A'
        elif position_code == 'D':
            self.role = 'D'
        elif position_code == 'G':
            self.role = 'G'

    def set_season_points(self, season):
        self.seasons[season] = Season(season)

    def get_season_points(self, season):
        return self.seasons[season].get_marqueur_points(self.role)

    def get_ratio_season_points(self, season):
        return self.seasons[season].get_ratio_marqueur_points(self.role)
    
    
class Season:
    def __init__(self, season_id):
        self.season_id = season_id
        self.team = None
        self.n_games_played = None
        self.n_assists = None
        self.n_wins = None
        self.n_shutouts = None
        self.n_goals = None
        self.n_pim = None
        self.n_shots = None
        self.n_plusMinus = None
        self.n_time = None

    def set_team(self, team):
        self.team = team
        # TODO: get abbrev from name

    def set_n_goals(self, n_goals):
        self.n_goals = n_goals

    def set_n_assists(self, n_assists):
        self.n_assists = n_assists

    def set_n_pim(self, n_pim):
        self.n_pim = n_pim

    def set_n_shots(self, n_shots):
        self.n_shots = n_shots

    def set_n_time(self, n_time):
        self.n_time = n_time

    def set_n_plusMinus(self, n_plusMinus):
        self.n_plusMinus = n_plusMinus

    def set_n_wins(self, n_wins):
        self.n_wins = n_wins

    def set_n_shutouts(self, n_shutouts):
        self.n_shutouts = n_shutouts
    
    def set_n_games_played(self, n_games_played):
        self.n_games_played = n_games_played

    def get_marqueur_points(self, role):
        if role == "A":
            return 2 * self.n_goals + 1 * self.n_assists
        elif role == "D":
            return 3 * self.n_goals + 1 * self.n_assists
        elif role == "G":
            return 3 * self.n_wins + 5 * self.n_shutouts + 3 * self.n_goals + 1 * self.n_assists

    def get_ratio_marqueur_points(self, role):
        return self.get_marqueur_points(role) / self.n_games_played

    def get_points(self):
        return self.n_goals + self.n_assists
