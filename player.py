import datetime


class Player:
    def __init__(self, id):
        self.seasons = {}
        self.role = None
        self.salary_name = None
        self.position = None
        self.team = None
        self.salary = None
        self.points = None
        self.age = None
        self.name = None
        self.birth_date = None
        self.id = id

    def set_name(self, first_name, last_name):
        self.name = first_name + ' ' + last_name
        self.salary_name = last_name + ', ' + first_name

    def set_age(self, birth_date):
        self.birth_date = birth_date
        today = datetime.date.today()
        born = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
        self.age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def set_salary(self, salary):
        self.salary = salary
        
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

    def get_ratio_points_salary(self):
        return self.points / self.salary
    
    
class Season:
    def __init__(self, season_id):
        self.n_games_played = None
        self.n_assists = None
        self.n_wins = None
        self.n_shutouts = None
        self.n_goals = None
        self.season_id = season_id

    def set_n_goals(self, n_goals):
        self.n_goals = n_goals

    def set_n_assists(self, n_assists):
        self.n_assists = n_assists

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

    def get_points(self):
        return self.n_goals + self.n_assists
