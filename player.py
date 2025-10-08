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
        self.salary = 88000000
        self.points = None
        self.age = None
        self.name = None
        self.birth_date = None
        self.predict_points = 0
        self.height = None
        self.weight = None
        self.country = None
        self.team = None

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

    def set_team(self, team):
        self.team = team
        
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

        # Common stats (skaters and goalies)
        self.n_games_played = None
        self.n_goals = None
        self.n_assists = None
        self.n_points = None
        self.n_pim = None

        # Skater-specific stats
        self.n_shots = None
        self.n_plus_minus = None
        self.n_time_on_ice_per_game = None
        self.n_powerplay_goals = None
        self.n_powerplay_points = None
        self.n_shorthanded_goals = None
        self.n_shorthanded_points = None
        self.n_game_winning_goals = None
        self.n_overtime_goals = None
        self.n_faceoff_percentage = None
        self.n_shooting_percentage = None
        self.n_avg_shifts_per_game = None

        # Goalie-specific stats
        self.n_games_started = None
        self.n_wins = None
        self.n_losses = None
        self.n_ties = None
        self.n_overtime_losses = None
        self.n_shutouts = None
        self.n_goals_against = None
        self.n_goals_against_avg = None
        self.n_shots_against = None
        self.n_saves = None
        self.n_save_percentage = None
        self.n_time_on_ice = None  # Total TOI for goalies

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

    def set_n_points(self, n_points):
        self.n_points = n_points

    def set_n_games_played(self, n_games_played):
        self.n_games_played = n_games_played

    # Skater setters
    def set_n_plus_minus(self, n_plus_minus):
        self.n_plus_minus = n_plus_minus

    def set_n_time_on_ice_per_game(self, n_time_on_ice_per_game):
        self.n_time_on_ice_per_game = n_time_on_ice_per_game

    def set_n_powerplay_goals(self, n_powerplay_goals):
        self.n_powerplay_goals = n_powerplay_goals

    def set_n_powerplay_points(self, n_powerplay_points):
        self.n_powerplay_points = n_powerplay_points

    def set_n_shorthanded_goals(self, n_shorthanded_goals):
        self.n_shorthanded_goals = n_shorthanded_goals

    def set_n_shorthanded_points(self, n_shorthanded_points):
        self.n_shorthanded_points = n_shorthanded_points

    def set_n_game_winning_goals(self, n_game_winning_goals):
        self.n_game_winning_goals = n_game_winning_goals

    def set_n_overtime_goals(self, n_overtime_goals):
        self.n_overtime_goals = n_overtime_goals

    def set_n_faceoff_percentage(self, n_faceoff_percentage):
        self.n_faceoff_percentage = n_faceoff_percentage

    def set_n_shooting_percentage(self, n_shooting_percentage):
        self.n_shooting_percentage = n_shooting_percentage

    def set_n_avg_shifts_per_game(self, n_avg_shifts_per_game):
        self.n_avg_shifts_per_game = n_avg_shifts_per_game

    # Goalie setters
    def set_n_games_started(self, n_games_started):
        self.n_games_started = n_games_started

    def set_n_wins(self, n_wins):
        self.n_wins = n_wins

    def set_n_losses(self, n_losses):
        self.n_losses = n_losses

    def set_n_ties(self, n_ties):
        self.n_ties = n_ties

    def set_n_overtime_losses(self, n_overtime_losses):
        self.n_overtime_losses = n_overtime_losses

    def set_n_shutouts(self, n_shutouts):
        self.n_shutouts = n_shutouts

    def set_n_goals_against(self, n_goals_against):
        self.n_goals_against = n_goals_against

    def set_n_goals_against_avg(self, n_goals_against_avg):
        self.n_goals_against_avg = n_goals_against_avg

    def set_n_shots_against(self, n_shots_against):
        self.n_shots_against = n_shots_against

    def set_n_saves(self, n_saves):
        self.n_saves = n_saves

    def set_n_save_percentage(self, n_save_percentage):
        self.n_save_percentage = n_save_percentage

    def set_n_time_on_ice(self, n_time_on_ice):
        self.n_time_on_ice = n_time_on_ice

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
