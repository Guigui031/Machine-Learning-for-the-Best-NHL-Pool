from datetime import date

class Player:
    def __init__(self, id):
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
        today = date.today()
        self.age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    def set_points(self, points):
        self.points = points

    def set_salary(self, salary):
        self.salary = salary
        
    def set_team(self, team):
        self.team = team
        
    def set_position(self, position_code):
        self.position = position_code

    def get_ratio_points_salary(self):
        return self.points / self.salary
    
    

