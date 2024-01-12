import copy
import math
import random
from math import log, sqrt, inf
from random import randrange
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress
import pandas as pd
import classes.logic as logic

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file

class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player
        

    def start(self):
        """
        This function select a tile from the board.

        @returns    (x, y) A tuple of integer corresponding to a valid
                    and free tile on the board.
        """
        raise NotImplementedError

class Node(object):
    """
    This class implements the main object that you will manipulate : nodes.
    Nodes include the state of the game (i.e. the 2D board), children (i.e. other children nodes), a list of
    untried moves, etc...
    """
    def __init__(self, board, move=(None, None),
                 wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Random(PlayerStrat):
    def start(self):
        available_tiles = []
        board_size = len(self.root_state)  # Obtenir la taille du tableau

        for x in range(board_size):
            for y in range(board_size):
                if self.root_state[x, y] == 0:  # Vérifier si la tuile est vide (représentée par 0)
                    available_tiles.append((x, y))

        if available_tiles:
            return random.choice(available_tiles)
        else:
            return None



class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, max_depth=100):
        super().__init__(_board_state, player)
        self.max_depth = max_depth

    def start(self):
        best_move = self.minimax(self.root_state, True)
        print(f"Je suis le Joueur {self.player}, mon placement {best_move}")
        return best_move

    def minimax(self, board, max_player):
        if logic.is_game_over(self.player, board) or logic.is_game_over(3 - self.player, board):
            return None, self.evaluate(board)

        if max_player:
            best_value = -math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = self.player
                _, value = self.minimax(new_board, False)
                new_board[move[0], move[1]] = 0 
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_move
        else:
            best_value = math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = 3 - self.player
                _, value = self.minimax(new_board, True)
                new_board[move[0], move[1]] = 0  
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_move

    def evaluate(self, board):
        if logic.is_game_over(self.player, board):
            return 1
        elif logic.is_game_over(3 - self.player, board):
            return -1
        else:
            return 0
        
class MiniMaxAlphaBeta(PlayerStrat):
    def __init__(self, _board_state, player, max_depth=100):
        super().__init__(_board_state, player)
        self.max_depth = max_depth

    def start(self):
        best_move, best_value = self.alphabeta(self.root_state ,-math.inf, math.inf, True)
        print(f"Je suis le Joueur {self.player}, mon placement {best_move}")
        return best_move

    def alphabeta(self, board, alpha, beta, max_player):
        if logic.is_game_over(self.player, board) or logic.is_game_over(3 - self.player, board):
            return None, self.evaluate(board) 

        if max_player:
            best_value = -math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = self.player
                _, value = self.alphabeta(new_board, alpha, beta, False)
                new_board[move[0], move[1]] = 0 
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_move, best_value
        else:
            best_value = math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = 3 - self.player
                _, value = self.alphabeta(new_board, alpha, beta, True)
                new_board[move[0], move[1]] = 0  
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_move, best_value

    def evaluate(self, board):
            if logic.is_game_over(self.player, board):
                return 1
            elif logic.is_game_over(3 - self.player, board):
                return -1
            else:
                return 0

class MyStrategyPlayer(PlayerStrat):
    def __init__(self, _board_state, player, max_depth=4):
        super().__init__(_board_state, player)
        self.max_depth = max_depth

    def start(self):
        best_move, _ = self.alphabeta(self.root_state, self.max_depth, -math.inf, math.inf, True)
        print(f"Player {self.player}: Best move at {best_move}")
        return best_move

    def alphabeta(self, board, depth, alpha, beta, max_player):
        if depth == 0 or depth < self.max_depth / 2:
            return None, self.utility(board)

        if logic.is_game_over(self.player, board) or logic.is_game_over(3 - self.player, board):
            return None, self.evaluate(board)

        best_value = -math.inf if max_player else math.inf
        best_move = None

        for move in logic.get_possible_moves(board):
            new_board = np.copy(board)
            new_board[move[0], move[1]] = self.player if max_player else 3 - self.player
            _, value = self.alphabeta(new_board, depth - 1, alpha, beta, not max_player)
            new_board[move[0], move[1]] = 0

            if (max_player and value > best_value) or (not max_player and value < best_value):
                best_value = value
                best_move = move

            alpha = max(alpha, value) if max_player else alpha
            beta = min(beta, value) if not max_player else beta

            if beta <= alpha:
                break

        return best_move, best_value

    def utility(self, board):
        centre = (len(board) - 1) // 2  # Position de la colonne centrale
        score_centre = sum(board[i][centre] == self.player for i in range(len(board)))

        score_adjacent_pieces = 0
        directions = [(0, 1), (1, 0), (1, 1), (-1, -1), (0, -1), (-1, 0), (-1, 1), (1, -1)]

        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == self.player:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(board) and 0 <= ny < len(board[nx]) and board[nx][ny] == self.player:
                            score_adjacent_pieces += 1

        weight_centre = 1
        weight_adjacent_pieces = 2

        combined_score = weight_centre * score_centre + weight_adjacent_pieces * score_adjacent_pieces

        return combined_score


    def evaluate(self, board):
        if logic.is_game_over(self.player, board):
            return 1
        elif logic.is_game_over(3 - self.player, board):
            return -1
        else:
            return 0


            

# Définition de la classe McPlayer héritant de PlayerStrat
class McPlayer(PlayerStrat):
    def __init__(self, _board_state, player, num_simulations=1000):
        super().__init__(_board_state, player)
        self.num_simulations = num_simulations
        self.move_counts = {}
        self.move_scores = {} 

    # Méthode pour commencer la simulation MC et sélectionner le meilleur mouvement
    def start(self):
        for _ in range(self.num_simulations):
            board_copy = np.copy(self.root_state)
            self.run_simulation(board_copy)
        best_move = self.select_best_move()
        print(f"Je suis le Joueur {self.player}, mon placement {best_move}")
        return best_move

    # Méthode pour exécuter une seule simulation Monte Carlo
    def run_simulation(self, board):
        move_history = []
        while not logic.is_game_over(self.player, board) and not logic.is_game_over(3 - self.player, board):
            possible_moves = logic.get_possible_moves(board)
            move = random.choice(possible_moves)
            board[move[0], move[1]] = self.player if len(move_history) % 2 == 0 else 3 - self.player
            move_history.append(move)
        winner = self.player if logic.is_game_over(self.player, board) else 3 - self.player
        self.update_move_scores(move_history, winner)

    # Méthode pour mettre à jour les scores des mouvements en fonction des résultats des simulations
    def update_move_scores(self, move_history, winner):
        for move in move_history:
            if move not in self.move_counts:
                self.move_counts[move] = 0
                self.move_scores[move] = 0  
            if (winner == self.player and move_history.index(move) % 2 == 0) or (winner != self.player and move_history.index(move) % 2 == 1):
                self.move_counts[move] += 1
                self.move_scores[move] += 1  

    # Méthode pour sélectionner le meilleur mouvement en fonction des scores des mouvements
    def select_best_move(self):
        best_move = max(self.move_scores, key=self.move_scores.get)
        return best_move

            
str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
        "my_new_ai_strat":MyStrategyPlayer,
        "minimax_ab": MiniMaxAlphaBeta,
         "mc": McPlayer,
}
