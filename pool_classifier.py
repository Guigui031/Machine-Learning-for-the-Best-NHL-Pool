from process_data import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

budget = 88000000
n_goalies = 2
n_atk = 12
n_def = 6


def load_players():
    # load players
    data = get_json()
    ids = get_all_player_ids(data)
    players = []
    for i in ids:
        players.append(load_player(str(i)))
    return players


def create_dataset(players):
    # compile dataset
    dataset = np.zeros((len(players),4))
    seasons = ['20202021', '20212022', '20222023', '20232024']
    len_seasons = len(seasons)
    for i in range(len(players)):
        for j in range(len_seasons):
            dataset[i,j] = players[i].get_ratio_season_points(seasons[j])

    return dataset


def regression_on_player(ratios_points_par_match):
    saisons = np.array([1, 2, 3, 4]).reshape(-1, 1)
    model = LinearRegression()
    # Ajuster le modèle sur les données de saisons précédentes
    model.fit(saisons, ratios_points_par_match)

    # Prédire le ratio pour la saison suivante (saison 5)
    saison_suivante = np.array([[5]])
    ratio_predit = model.predict(saison_suivante)
    return ratio_predit


# Fonction d'évaluation (borne supérieure)
def calcul_borne_sup(solution, joueurs_restants):
    total_valeur = sum(joueur.predict_points for joueur in solution)
    borne_sup = total_valeur
    for joueur in joueurs_restants:
        borne_sup += joueur.predict_points
    return borne_sup


# Fonction pour vérifier la faisabilité
def est_faisable(solution):
    total_salaire = sum(joueur.salary for joueur in solution)
    nb_gardiens = len([j for j in solution if j.role == "G"])
    nb_defenseurs = len([j for j in solution if j.role == "D"])
    nb_attaquants = len([j for j in solution if j.role == "A"])

    return (total_salaire <= budget and
            # nb_gardiens == n_goalies and
            0 <= nb_defenseurs <= n_def and
            0 <= nb_attaquants <= n_atk)


# Algorithme Branch and Bound
def branch_and_bound(joueurs):
    meilleure_solution = []
    meilleure_valeur = 0  # ppg de l'équipe

    # Pile pour explorer les solutions
    pile = [(0, [])]  # (indice du joueur à explorer, solution partielle)

    while pile:
        indice, solution_actuelle = pile.pop()

        # Si on a exploré tous les joueurs, on vérifie si c'est faisable et meilleur
        if indice == len(joueurs):
            if est_faisable(solution_actuelle):
                valeur_solution = sum(joueur.predict_points for joueur in solution_actuelle)
                if valeur_solution > meilleure_valeur:
                    meilleure_solution = solution_actuelle[:]
                    meilleure_valeur = valeur_solution
            continue

        # Joueur actuel
        joueur_actuel = joueurs[indice]

        # Générer la borne supérieure pour cette branche
        borne_sup = calcul_borne_sup(solution_actuelle, joueurs[indice:])

        # Si la borne supérieure est plus petite que la meilleure valeur actuelle, on abandonne cette branche
        if borne_sup <= meilleure_valeur:
            continue

        # Branch 1 : on inclut le joueur actuel dans la solution
        nouvelle_solution = solution_actuelle + [joueur_actuel]
        pile.append((indice + 1, nouvelle_solution))

        # Branch 2 : on n'inclut pas le joueur actuel
        pile.append((indice + 1, solution_actuelle))

    return meilleure_solution, meilleure_valeur


def team_optimization_branch_and_bound(players, budget=budget, max_roles={"A": n_atk, "D": n_def, "G": n_goalies}):
    n = len(players)
    best_team = []
    best_ppg = 0

    # Sort players by ppg/salary ratio for efficiency
    players.sort(key=lambda p: p.predict_points / p.salary, reverse=True)

    def bound(index, current_ppg, current_salary, role_counts):
        """
        Compute the upper bound of the maximum PPG for the current branch.
        Assumes we can pick the remaining players up to the budget limit.
        """
        remaining_ppg = current_ppg
        remaining_salary = budget - current_salary
        remaining_role_counts = role_counts.copy()
        for i in range(index, n):
            player = players[i]
            if player.salary <= remaining_salary and remaining_role_counts[player.role] < max_roles[player.role]:
                remaining_ppg += player.predict_points
                remaining_salary -= player.salary
            else:
                break
        return remaining_ppg

    def branch_and_bound(index, current_team, current_ppg, current_salary, role_counts):
        nonlocal best_team, best_ppg

        # Prune if over budget or invalid role counts
        if current_salary > budget:
            return
        for role in max_roles.keys():
            if role_counts[role] > max_roles[role]:
                return

        # Update the best solution
        if current_ppg > best_ppg:
            best_ppg = current_ppg
            best_team = current_team[:]

        # Compute the upper bound
        if index >= n:
            return
        upper_bound = bound(index, current_ppg, current_salary, role_counts)
        if upper_bound <= best_ppg:
            return  # Prune the branch

        # Branch: include the current player
        if index < n:
            player = players[index]
            role_counts[player.role] += 1
            current_team.append(player)
            branch_and_bound(
                index + 1,
                current_team,
                current_ppg + player.predict_points,
                current_salary + player.salary,
                role_counts,
            )
            # Backtrack: exclude the player
            current_team.pop()
            role_counts[player.role] -= 1

        # Branch: exclude the current player
        branch_and_bound(index + 1, current_team, current_ppg, current_salary, role_counts)

    # Initialize the algorithm
    branch_and_bound(0, [], 0, 0, {"A": 0, "D": 0, "G": 0})

    return best_team, best_ppg


def solve_problem(players):
    # Define the problem
    prob = LpProblem("Maximize_PPG", LpMaximize)

    # Create decision variables (1 if player is selected, 0 otherwise)
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(len(players))}

    # Objective function: Maximize the total PPG
    prob += lpSum(players[i].predict_points * x[i] for i in range(len(players)))

    # Constraint: Budget limit
    prob += lpSum(players[i].salary * x[i] for i in range(len(players))) <= 88000000

    # Constraints: Role limits
    prob += lpSum(x[i] for i in range(len(players)) if players[i].role == "A") <= 12
    prob += lpSum(x[i] for i in range(len(players)) if players[i].role == "D") <= 6
    prob += lpSum(x[i] for i in range(len(players)) if players[i].role == "G") <= 2

    # Solve the problem
    prob.solve()

    # Display the results
    selected_players = [players[i].name for i in range(len(players)) if x[i].value() == 1]
    total_ppg = sum(players[i].predict_points for i in range(len(players)) if x[i].value() == 1)
    total_salary = sum(players[i].salary for i in range(len(players)) if x[i].value() == 1)

    return selected_players, total_ppg, total_salary



def create_team():
    players = load_players()
    dataset = create_dataset(players)

    # régression
    for i in range(len(players)):
        players[i].predict_points = regression_on_player(dataset[i,:])

    # algo
    meilleure_solution, meilleure_valeur = branch_and_bound(players)
    print('meilleure_valeur', meilleure_valeur)
    print('meilleure_solution')
    results = []
    for joueur in meilleure_solution:
        result = (joueur.role, joueur.name, joueur.salary. joueur.team)
        print(result)
        results.append(result)
    pandas.DataFrame(results, columns=['role', 'name', 'salary', 'team']).to_csv('meilleure_solution.csv')


if __name__ == '__main__':
    create_team()